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

/// Parse an expression once (int literals floatified). `None` on a syntax
/// error, which evaluates as 0 / false like a runtime error does.
fn compile(expr: &str) -> Option<Node> {
    build_operator_tree::<DefaultNumericTypes>(&floatify_int_literals(expr)).ok()
}

/// A scheme with every expression pre-parsed, so repeated evaluation (the
/// second-pass gap search runs hundreds per student) skips re-tokenizing.
struct CompiledScheme<'a> {
    scheme: &'a GradeScheme,
    vars: Vec<Option<Node>>,
    rules: Vec<Option<Node>>,
    total: Option<Node>,
}

/// Outcome of evaluating a scheme over fully-bound question points.
struct Evaluated {
    grade: String,
    total: f64,
    vars: HashMap<String, f64>,
}

impl<'a> CompiledScheme<'a> {
    fn new(scheme: &'a GradeScheme) -> Self {
        CompiledScheme {
            scheme,
            vars: scheme.vars.iter().map(|v| compile(&v.expr)).collect(),
            rules: scheme.rules.iter().map(|r| compile(&r.when)).collect(),
            total: compile(total_var(scheme)),
        }
    }

    /// Evaluate aggregate vars, total and the first matching rule.
    fn evaluate(&self, questions: &[QuestionConfig], var_points: &HashMap<String, f64>) -> Evaluated {
        let mut ctx = base_context(self.scheme);
        for (k, v) in var_points {
            let _ = ctx.set_value(k.clone(), Value::Float(*v));
        }
        install_groupsum(&mut ctx, questions, var_points);

        // Named aggregate vars, in order. Round each to 2 decimals before it feeds
        // later vars and the rule guards, so binary-float noise can't drop a value
        // below a band threshold (mirrors the legacy `round(total, 2)`).
        let mut vars = HashMap::new();
        for (sv, node) in self.scheme.vars.iter().zip(&self.vars) {
            let v = round2(
                node.as_ref()
                    .and_then(|n| n.eval_number_with_context(&ctx).ok())
                    .unwrap_or(0.0),
            );
            let _ = ctx.set_value(sv.name.clone(), Value::Float(v));
            vars.insert(sv.name.clone(), v);
        }

        let total = round2(
            vars.get(total_var(self.scheme))
                .copied()
                .or_else(|| self.total.as_ref().and_then(|n| n.eval_number_with_context(&ctx).ok()))
                .unwrap_or(0.0),
        );

        // Guarded rules, first match wins.
        let grade = self
            .scheme
            .rules
            .iter()
            .zip(&self.rules)
            .find(|(_, node)| {
                node.as_ref()
                    .and_then(|n| n.eval_boolean_with_context(&ctx).ok())
                    .unwrap_or(false)
            })
            .map(|(rule, _)| rule.grade.clone())
            .unwrap_or_else(|| default_grade(self.scheme).to_string());
        Evaluated { grade, total, vars }
    }
}

fn total_var(scheme: &GradeScheme) -> &str {
    if scheme.total_var.is_empty() {
        "total"
    } else {
        &scheme.total_var
    }
}

fn default_grade(scheme: &GradeScheme) -> &str {
    if scheme.default_grade.is_empty() {
        "F"
    } else {
        &scheme.default_grade
    }
}

fn base_context(scheme: &GradeScheme) -> HashMapContext {
    let mut ctx = HashMapContext::new();
    for c in &scheme.constants {
        let _ = ctx.set_value(c.name.clone(), Value::Float(c.value));
    }
    ctx
}

/// Bind question points, estimating ungraded questions that carry an
/// estimator. Returns (points per var, estimated vars, every question graded).
fn resolve_points(
    scheme: &GradeScheme,
    questions: &[QuestionConfig],
    points: &HashMap<String, Option<f64>>,
) -> (HashMap<String, f64>, Vec<String>, bool) {
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
    let mut ctx = base_context(scheme);
    for (k, v) in &var_points {
        let _ = ctx.set_value(k.clone(), Value::Float(*v));
    }
    install_groupsum(&mut ctx, questions, &var_points);

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
    (var_points, estimated, ungraded.is_empty())
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
    let (var_points, estimated, complete) = resolve_points(scheme, questions, points);
    let ev = CompiledScheme::new(scheme).evaluate(questions, &var_points);
    StudentResult {
        id: id.to_string(),
        grade: ev.grade,
        total: ev.total,
        vars: ev.vars,
        estimated,
        complete,
    }
}

// ---------------------------------------------------------------------------
// Second pass: how close is a student to a better grade?
// ---------------------------------------------------------------------------

const ECTS: [&str; 7] = ["A", "B", "C", "D", "E", "FX", "F"];

/// Rank of each grade the scheme can produce (0 = best). The ECTS scale when
/// every grade is on it; otherwise first appearance in the rules with the
/// default grade last. Rule order alone is not grade order: gate rules such as
/// `se_total < secmin -> F` usually come first.
pub fn grade_ranks(scheme: &GradeScheme) -> HashMap<String, usize> {
    let default = default_grade(scheme);
    let mut order: Vec<&str> = Vec::new();
    for g in scheme.rules.iter().map(|r| r.grade.as_str()).chain([default]) {
        if !order.contains(&g) {
            order.push(g);
        }
    }
    if order.iter().all(|g| ECTS.contains(g)) {
        order.sort_by_key(|g| ECTS.iter().position(|e| e == g));
    } else {
        order.retain(|g| *g != default);
        order.push(default);
    }
    order.into_iter().enumerate().map(|(i, g)| (g.to_string(), i)).collect()
}

/// Where the extra points of a lift come from.
#[derive(Clone, Debug, PartialEq, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
#[serde(rename_all = "lowercase")]
pub enum LiftSource {
    /// Spread over any questions with headroom.
    Any,
    /// Spread over one group's questions (`key` is the group tag).
    Group,
    /// One question alone (`key` is its scheme var).
    Question,
}

/// Extra points from one source that reach a better grade.
#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct GradeLift {
    pub source: LiftSource,
    pub key: String,
    /// Smallest extra points (rounded up to 0.01) from this source.
    pub points: f64,
    pub grade: String,
}

/// A student within `margin` points of a better grade.
#[derive(Clone, Debug, Serialize, TS)]
#[ts(export, export_to = "../../web/src/lib/generated/")]
pub struct GradeGap {
    /// The better grade reached by the cheapest lift.
    pub next_grade: String,
    /// Points of the cheapest lift.
    pub points: f64,
    /// Every source that reaches a better grade within the margin, cheapest first.
    pub lifts: Vec<GradeLift>,
}

/// Headroom per question var: how many more points it can take.
type Room<'a> = Vec<(&'a str, f64)>;

/// Find the smallest extra points (at most `margin`) that lift a student to a
/// better grade. Rules are arbitrary expressions, so instead of reading
/// thresholds this simulates: for each source (any question, each group, each
/// question alone) it adds points into questions' headroom (`max_points` minus
/// current, estimated questions excluded) and bisects for the cheapest amount
/// that changes the grade for the better. Assumes more points never worsen the
/// grade, which holds for any sane scheme. `None` when no source reaches a
/// better grade within the margin.
pub fn grade_gap(
    scheme: &GradeScheme,
    questions: &[QuestionConfig],
    points: &HashMap<String, Option<f64>>,
    margin: f64,
) -> Option<GradeGap> {
    let compiled = CompiledScheme::new(scheme);
    let ranks = grade_ranks(scheme);
    let rank = |g: &str| ranks.get(g).copied().unwrap_or(usize::MAX);
    let (base, estimated, _) = resolve_points(scheme, questions, points);
    let current = rank(&compiled.evaluate(questions, &base).grade);

    let headroom: Room = questions
        .iter()
        .filter(|q| q.max_points > 0.0 && !estimated.contains(&q.var))
        .map(|q| (q.var.as_str(), q.max_points - base.get(&q.var).copied().unwrap_or(0.0)))
        .filter(|(_, h)| *h > 1e-9)
        .collect();

    let mut sources: Vec<(LiftSource, String, Room)> =
        vec![(LiftSource::Any, String::new(), headroom.clone())];
    let mut groups: Vec<&str> =
        questions.iter().map(|q| q.group.as_str()).filter(|g| !g.is_empty()).collect();
    groups.sort();
    groups.dedup();
    for g in groups {
        let in_group: Room = headroom
            .iter()
            .copied()
            .filter(|(v, _)| questions.iter().any(|q| q.var == *v && q.group == g))
            .collect();
        sources.push((LiftSource::Group, g.to_string(), in_group));
    }
    for &(v, h) in &headroom {
        sources.push((LiftSource::Question, v.to_string(), vec![(v, h)]));
    }

    // Grade after pouring `delta` into `room`, filling each question in turn.
    let lifted = |room: &[(&str, f64)], delta: f64| -> String {
        let mut pts = base.clone();
        let mut left = delta;
        for &(v, h) in room {
            let add = left.min(h);
            *pts.entry(v.to_string()).or_insert(0.0) += add;
            left -= add;
            if left <= 0.0 {
                break;
            }
        }
        compiled.evaluate(questions, &pts).grade
    };

    let mut lifts: Vec<GradeLift> = Vec::new();
    for (source, key, room) in sources {
        let cap = margin.min(room.iter().map(|(_, h)| h).sum());
        if cap <= 0.0 || rank(&lifted(&room, cap)) >= current {
            continue;
        }
        let (mut lo, mut hi) = (0.0, cap);
        while hi - lo > 1e-4 {
            let mid = (lo + hi) / 2.0;
            if rank(&lifted(&room, mid)) < current {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let points = ((hi * 100.0).ceil() / 100.0).min(cap);
        lifts.push(GradeLift { source, key, points, grade: lifted(&room, points) });
    }
    lifts.sort_by(|a, b| a.points.partial_cmp(&b.points).unwrap_or(std::cmp::Ordering::Equal));
    let best = lifts.first()?;
    Some(GradeGap { next_grade: best.grade.clone(), points: best.points, lifts })
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

    fn pvt_with_max() -> Vec<QuestionConfig> {
        let mut qs = pvt_questions();
        for (q, max) in qs.iter_mut().zip([18.0, 18.0, 7.0, 7.0]) {
            q.max_points = max;
        }
        qs
    }

    fn gap(pts: [f64; 4], margin: f64) -> Option<GradeGap> {
        let points: HashMap<String, Option<f64>> = pvt_with_max()
            .iter()
            .zip(pts)
            .map(|(q, p)| (q.var.clone(), Some(p)))
            .collect();
        grade_gap(&pvt_scheme(), &pvt_with_max(), &points, margin)
    }

    #[test]
    fn ranks_ignore_gate_rule_order() {
        // PVT lists its F gates first; F must still rank last.
        let r = grade_ranks(&pvt_scheme());
        assert!(r["A"] < r["B"] && r["D"] < r["E"] && r["E"] < r["F"]);
        // Non-ECTS scale: first appearance, default last.
        let rule = |w: &str, g: &str| GradeRule { when: w.into(), grade: g.into() };
        let s = GradeScheme {
            rules: vec![rule("total >= 30", "VG"), rule("total >= 20", "G")],
            default_grade: "U".into(),
            ..Default::default()
        };
        let r = grade_ranks(&s);
        assert!(r["VG"] < r["G"] && r["G"] < r["U"]);
    }

    #[test]
    fn gap_finds_threshold_distance() {
        // total 22 < emin - lenience (23): F, one point short of E.
        let g = gap([10.0, 10.0, 1.0, 1.0], 1.0).unwrap();
        assert_eq!(g.next_grade, "E");
        assert!((g.points - 1.0).abs() < 1e-9, "{}", g.points);
        assert!(gap([10.0, 10.0, 1.0, 1.0], 0.5).is_none());
    }

    #[test]
    fn gap_sees_gates_not_just_totals() {
        // High total capped at D by the zero hci essay: a hundredth of a point
        // on that essay alone reaches A; the full questions offer no headroom.
        let g = gap([18.0, 18.0, 7.0, 0.0], 1.0).unwrap();
        assert_eq!(g.next_grade, "A");
        assert!(g.points <= 0.011);
        assert!(g
            .lifts
            .iter()
            .any(|l| l.source == LiftSource::Question && l.key == "hci_essay"));
        assert!(!g.lifts.iter().any(|l| l.key == "se_mc"));
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
