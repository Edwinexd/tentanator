//! Examination grade scheme engine.
//!
//! General, config-driven grading: an examination defines per-question config
//! (max points, type, group, an expression `var`), tunable constants, named
//! aggregate variables, and an ordered list of guarded rules `when <bool> ->
//! <grade>` (first match wins). Nothing about sections or question counts is
//! hardcoded; PVT's SE/HCI ECTS rules are just one configuration.
//!
//! Expressions are evaluated with `evalexpr` (arithmetic, comparisons, &&/||,
//! built-ins). A `groupsum("tag")` function sums the points of questions whose
//! `group` equals the tag, so sectioned exams need no explicit long sums.

use std::collections::HashMap;

use evalexpr::*;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// Per-question grading config (decoupled from grading I/O so the engine is
/// pure and testable). `var` is the identifier used in scheme expressions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QuestionConfig {
    pub var: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub group: String,
    #[serde(default)]
    pub qtype: String,
    #[serde(default)]
    pub max_points: f64,
    #[serde(default)]
    pub position: i32,
    /// Optional expression to estimate points when the question is ungraded
    /// (e.g. `hci_mc / 18 * 7`). Flagged in the result when used.
    #[serde(default)]
    pub estimate: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct SchemeConst {
    pub name: String,
    pub value: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct SchemeVar {
    pub name: String,
    pub expr: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct GradeRule {
    pub when: String,
    pub grade: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct GradeScheme {
    #[serde(default)]
    pub constants: Vec<SchemeConst>,
    #[serde(default)]
    pub vars: Vec<SchemeVar>,
    #[serde(default)]
    pub rules: Vec<GradeRule>,
    /// Which variable to surface as the headline "total" (default "total").
    #[serde(default)]
    pub total_var: String,
    /// Grade when no rule matches (default "F").
    #[serde(default)]
    pub default_grade: String,
}

#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct StudentResult {
    pub id: String,
    pub grade: String,
    pub total: f64,
    pub vars: HashMap<String, f64>,
    /// Question vars whose value was estimated (ungraded + estimator applied).
    pub estimated: Vec<String>,
    /// True when every question had a real grade (no estimates / missing).
    pub complete: bool,
}

/// Round to 2 decimals (banker-free, half-away-from-zero via `f64::round`).
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Append `.0` to bare integer literals so `evalexpr` never does integer
/// division: `7 / 18` would otherwise evaluate to `0` (i64 division), silently
/// breaking estimators like `7 / 18 * mc`. Numbers that are part of an
/// identifier (`hci_mc`), already floats (`3.5`), or inside a string literal
/// (`groupsum("2024")`) are left untouched.
fn floatify_int_literals(expr: &str) -> String {
    let chars: Vec<char> = expr.chars().collect();
    let mut out = String::with_capacity(expr.len() + 8);
    let mut i = 0;
    let mut in_string = false;
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            out.push(c);
            if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            i += 1;
            continue;
        }
        let prev = i.checked_sub(1).map(|p| chars[p]);
        let starts_number = c.is_ascii_digit()
            && !matches!(prev, Some(p) if p.is_alphanumeric() || p == '_' || p == '.');
        if starts_number {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            for &d in &chars[start..i] {
                out.push(d);
            }
            // Skip a real float (`3.5`) or an identifier suffix (`1e5`); only a
            // pure integer literal gets the `.0`.
            let next = chars.get(i).copied();
            let is_float_or_ident =
                matches!(next, Some(n) if n == '.' || n.is_alphabetic() || n == '_');
            if !is_float_or_ident {
                out.push_str(".0");
            }
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn install_groupsum(
    ctx: &mut HashMapContext,
    questions: &[QuestionConfig],
    points: &HashMap<String, f64>,
) {
    let mut sums: HashMap<String, f64> = HashMap::new();
    for q in questions {
        if q.group.is_empty() {
            continue;
        }
        let p = points.get(&q.var).copied().unwrap_or(0.0);
        *sums.entry(q.group.clone()).or_insert(0.0) += p;
    }
    let _ = ctx.set_function(
        "groupsum".to_string(),
        Function::new(move |arg| {
            let key = arg.as_string()?;
            Ok(Value::Float(sums.get(&key).copied().unwrap_or(0.0)))
        }),
    );
}

/// Compute one student's result. `points[var] = Some(p)` for a graded question,
/// `None` (or absent) for ungraded. Expression errors degrade to 0 / false
/// rather than failing the whole computation.
pub fn compute_student(
    id: &str,
    scheme: &GradeScheme,
    questions: &[QuestionConfig],
    points: &HashMap<String, Option<f64>>,
) -> StudentResult {
    let mut ctx = HashMapContext::new();
    for c in &scheme.constants {
        let _ = ctx.set_value(c.name.clone(), Value::Float(c.value));
    }

    // Bind question vars (ungraded -> 0 for now); collect ungraded.
    let mut var_points: HashMap<String, f64> = HashMap::new();
    let mut ungraded: Vec<&QuestionConfig> = Vec::new();
    for q in questions {
        match points.get(&q.var).copied().flatten() {
            Some(p) => {
                var_points.insert(q.var.clone(), p);
            }
            None => {
                var_points.insert(q.var.clone(), 0.0);
                ungraded.push(q);
            }
        }
    }
    for (k, v) in &var_points {
        let _ = ctx.set_value(k.clone(), Value::Float(*v));
    }
    install_groupsum(&mut ctx, questions, &var_points);

    // Estimate ungraded questions that carry an estimator.
    let mut estimated = Vec::new();
    for q in &ungraded {
        let Some(expr) = q.estimate.as_ref().filter(|e| !e.trim().is_empty()) else {
            continue;
        };
        if let Ok(v) = eval_number_with_context(&floatify_int_literals(expr), &ctx) {
            var_points.insert(q.var.clone(), v);
            let _ = ctx.set_value(q.var.clone(), Value::Float(v));
            estimated.push(q.var.clone());
        }
    }
    if !estimated.is_empty() {
        install_groupsum(&mut ctx, questions, &var_points);
    }

    // Named aggregate vars, in order. Round each to 2 decimals before it feeds
    // later vars and the rule guards, so binary-float noise can't drop a value
    // below a band threshold (mirrors the legacy `round(total, 2)`).
    let mut vars_out: HashMap<String, f64> = HashMap::new();
    for sv in &scheme.vars {
        let v = round2(eval_number_with_context(&floatify_int_literals(&sv.expr), &ctx).unwrap_or(0.0));
        let _ = ctx.set_value(sv.name.clone(), Value::Float(v));
        vars_out.insert(sv.name.clone(), v);
    }

    let total_var = if scheme.total_var.is_empty() {
        "total"
    } else {
        scheme.total_var.as_str()
    };
    let total = round2(
        vars_out
            .get(total_var)
            .copied()
            .or_else(|| eval_number_with_context(total_var, &ctx).ok())
            .unwrap_or(0.0),
    );

    // Guarded rules, first match wins.
    let mut grade = if scheme.default_grade.is_empty() {
        "F".to_string()
    } else {
        scheme.default_grade.clone()
    };
    for rule in &scheme.rules {
        if eval_boolean_with_context(&floatify_int_literals(&rule.when), &ctx).unwrap_or(false) {
            grade = rule.grade.clone();
            break;
        }
    }

    StudentResult {
        id: id.to_string(),
        grade,
        total,
        vars: vars_out,
        estimated,
        complete: ungraded.is_empty(),
    }
}

/// Validate that every variable referenced by the scheme's expressions resolves
/// against the known set (constants + question vars + earlier scheme vars), so a
/// typo'd identifier is rejected at config time rather than silently grading
/// everyone wrong. `question_vars` are the effective vars of the questions.
pub fn validate_scheme(scheme: &GradeScheme, question_vars: &[String]) -> Result<(), String> {
    let mut available: std::collections::HashSet<String> = std::collections::HashSet::new();
    for c in &scheme.constants {
        available.insert(c.name.clone());
    }
    for v in question_vars {
        available.insert(v.clone());
    }
    let check = |expr: &str, avail: &std::collections::HashSet<String>| -> Result<(), String> {
        let tree = build_operator_tree::<DefaultNumericTypes>(expr)
            .map_err(|e| format!("invalid expression `{expr}`: {e}"))?;
        for id in tree.iter_variable_identifiers() {
            if !avail.contains(id) {
                return Err(format!("expression `{expr}` references unknown variable `{id}`"));
            }
        }
        Ok(())
    };
    for v in &scheme.vars {
        check(&v.expr, &available)?;
        available.insert(v.name.clone());
    }
    for r in &scheme.rules {
        check(&r.when, &available)?;
    }
    Ok(())
}

/// Grade distribution (letter -> count) over a set of results.
pub fn distribution(results: &[StudentResult]) -> HashMap<String, usize> {
    let mut d = HashMap::new();
    for r in results {
        *d.entry(r.grade.clone()).or_insert(0) += 1;
    }
    d
}

/// Summary statistics over student total points.
#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct GradeStats {
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub stdev: f64,
}

/// Summary statistics over student totals. `None` when there are no results.
pub fn stats(results: &[StudentResult]) -> Option<GradeStats> {
    if results.is_empty() {
        return None;
    }
    let mut totals: Vec<f64> = results.iter().map(|r| r.total).collect();
    totals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = totals.len();
    let sum: f64 = totals.iter().sum();
    let mean = sum / n as f64;
    let median = if n % 2 == 1 {
        totals[n / 2]
    } else {
        (totals[n / 2 - 1] + totals[n / 2]) / 2.0
    };
    let variance = totals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    Some(GradeStats {
        mean: round2(mean),
        median: round2(median),
        min: totals[0],
        max: totals[n - 1],
        stdev: round2(variance.sqrt()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The PVT scheme expressed in the general engine (see reference/pvt
    // 02_compute_grades.py): SE/HCI section totals, -2 lenience, section minima,
    // both-essays>0 gate for grade > D, ECTS A-F.
    fn pvt_scheme() -> GradeScheme {
        let rule = |w: &str, g: &str| GradeRule { when: w.into(), grade: g.into() };
        GradeScheme {
            constants: vec![
                SchemeConst { name: "lenience".into(), value: 2.0 },
                SchemeConst { name: "secmin".into(), value: 10.0 },
                SchemeConst { name: "emin".into(), value: 25.0 },
            ],
            vars: vec![
                SchemeVar { name: "se_total".into(), expr: "se_mc + se_essay".into() },
                SchemeVar { name: "hci_total".into(), expr: "hci_mc + hci_essay".into() },
                SchemeVar { name: "total".into(), expr: "se_total + hci_total".into() },
            ],
            rules: vec![
                rule("se_total < secmin || hci_total < secmin", "F"),
                rule("total < emin - lenience", "F"),
                rule("total >= 45 - lenience && se_essay > 0 && hci_essay > 0", "A"),
                rule("total >= 40 - lenience && se_essay > 0 && hci_essay > 0", "B"),
                rule("total >= 35 - lenience && se_essay > 0 && hci_essay > 0", "C"),
                rule("total >= 30 - lenience", "D"),
                rule("total >= emin - lenience", "E"),
            ],
            total_var: "total".into(),
            default_grade: "F".into(),
        }
    }

    fn pvt_questions() -> Vec<QuestionConfig> {
        ["se_mc", "hci_mc", "se_essay", "hci_essay"]
            .iter()
            .enumerate()
            .map(|(i, v)| QuestionConfig {
                var: v.to_string(),
                position: i as i32,
                ..Default::default()
            })
            .collect()
    }

    fn run(se_mc: f64, hci_mc: f64, se_essay: f64, hci_essay: f64) -> String {
        let points = HashMap::from([
            ("se_mc".to_string(), Some(se_mc)),
            ("hci_mc".to_string(), Some(hci_mc)),
            ("se_essay".to_string(), Some(se_essay)),
            ("hci_essay".to_string(), Some(hci_essay)),
        ]);
        compute_student("s", &pvt_scheme(), &pvt_questions(), &points).grade
    }

    #[test]
    fn reproduces_pvt_oracle_grades() {
        // (se_mc, hci_mc, se_essay, hci_essay) -> expected grade, from
        // reference/pvt/data/grades.json (anonymized; scores only).
        assert_eq!(run(7.66, 14.17, 0.0, 0.0), "F"); // section minimum fail
        assert_eq!(run(10.32, 14.33, 2.0, 3.0), "D");
        assert_eq!(run(13.0, 15.33, 5.0, 5.5), "B");
        assert_eq!(run(13.34, 15.34, 3.5, 3.0), "C");
        assert_eq!(run(9.68, 12.33, 0.5, 4.0), "E");
        assert_eq!(run(16.34, 15.34, 6.5, 6.5), "A");
    }

    #[test]
    fn both_essays_gate_caps_at_d() {
        // High total but one essay is 0 -> cannot exceed D.
        assert_eq!(run(18.0, 18.0, 7.0, 0.0), "D");
    }

    #[test]
    fn estimation_fills_ungraded() {
        // hci_essay ungraded with an estimator = hci_mc/18*7.
        let mut qs = pvt_questions();
        qs[3].estimate = Some("hci_mc / 18 * 7".into());
        let points = HashMap::from([
            ("se_mc".to_string(), Some(15.0)),
            ("hci_mc".to_string(), Some(18.0)),
            ("se_essay".to_string(), Some(6.0)),
            ("hci_essay".to_string(), None), // -> estimated to 7.0
        ]);
        let r = compute_student("s", &pvt_scheme(), &qs, &points);
        assert!(r.estimated.contains(&"hci_essay".to_string()));
        assert!(!r.complete);
        assert!((r.vars["hci_total"] - 25.0).abs() < 1e-9); // 18 + 7
    }

    #[test]
    fn groupsum_aggregates_by_tag() {
        let qs = vec![
            QuestionConfig { var: "q1".into(), group: "SE".into(), max_points: 1.0, ..Default::default() },
            QuestionConfig { var: "q2".into(), group: "SE".into(), max_points: 1.0, ..Default::default() },
            QuestionConfig { var: "q3".into(), group: "HCI".into(), max_points: 1.0, ..Default::default() },
        ];
        let scheme = GradeScheme {
            vars: vec![SchemeVar { name: "total".into(), expr: "groupsum(\"SE\")".into() }],
            rules: vec![GradeRule { when: "total >= 2".into(), grade: "PASS".into() }],
            total_var: "total".into(),
            default_grade: "FAIL".into(),
            ..Default::default()
        };
        let points = HashMap::from([
            ("q1".to_string(), Some(1.0)),
            ("q2".to_string(), Some(1.0)),
            ("q3".to_string(), Some(1.0)),
        ]);
        let r = compute_student("s", &scheme, &qs, &points);
        assert_eq!(r.total, 2.0);
        assert_eq!(r.grade, "PASS");
    }

    #[test]
    fn floatify_only_touches_bare_integers() {
        assert_eq!(floatify_int_literals("7 / 18 * hci_mc"), "7.0 / 18.0 * hci_mc");
        assert_eq!(floatify_int_literals("total >= 45 - lenience"), "total >= 45.0 - lenience");
        assert_eq!(floatify_int_literals("q1 + q2 >= 3.5"), "q1 + q2 >= 3.5");
        assert_eq!(floatify_int_literals("groupsum(\"2024\")"), "groupsum(\"2024\")");
    }

    #[test]
    fn estimator_uses_float_division() {
        // `7 / 18 * hci_mc` is integer division (-> 0) under raw evalexpr; floatify
        // makes it 7/18*18 = 7, so hci_total is 18 + 7 = 25.
        let mut qs = pvt_questions();
        qs[3].estimate = Some("7 / 18 * hci_mc".into());
        let points = HashMap::from([
            ("se_mc".to_string(), Some(15.0)),
            ("hci_mc".to_string(), Some(18.0)),
            ("se_essay".to_string(), Some(6.0)),
            ("hci_essay".to_string(), None),
        ]);
        let r = compute_student("s", &pvt_scheme(), &qs, &points);
        assert!(r.estimated.contains(&"hci_essay".to_string()));
        assert!((r.vars["hci_total"] - 25.0).abs() < 1e-9, "got {}", r.vars["hci_total"]);
    }

    #[test]
    fn validate_rejects_unknown_identifiers() {
        let qvars = vec!["q1".to_string(), "q2".to_string()];
        let bad = GradeScheme {
            vars: vec![SchemeVar { name: "total".into(), expr: "q1 + q2".into() }],
            rules: vec![GradeRule { when: "totl >= 5".into(), grade: "A".into() }], // typo
            ..Default::default()
        };
        let err = validate_scheme(&bad, &qvars).unwrap_err();
        assert!(err.contains("totl"), "{err}");

        let good = GradeScheme {
            vars: vec![SchemeVar { name: "total".into(), expr: "q1 + q2".into() }],
            rules: vec![GradeRule { when: "total >= 5".into(), grade: "A".into() }],
            ..Default::default()
        };
        assert!(validate_scheme(&good, &qvars).is_ok());

        // groupsum() is a function, not a variable - must not be flagged.
        let grp = GradeScheme {
            vars: vec![SchemeVar { name: "total".into(), expr: "groupsum(\"SE\")".into() }],
            rules: vec![GradeRule { when: "total >= 1".into(), grade: "P".into() }],
            ..Default::default()
        };
        assert!(validate_scheme(&grp, &qvars).is_ok());
    }
}
