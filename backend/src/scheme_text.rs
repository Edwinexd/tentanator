//! Readable round-trip for the grade scheme. The backend owns this grammar so
//! both clients stay thin and cannot drift on the format. One statement per
//! line:
//!
//! ```text
//! const <name> = <number>     a tunable constant
//! <name> = <expr>             a named aggregate variable
//! when <cond> -> <grade>      a guarded rule (first match wins)
//! total_var: <name>           headline total var (optional)
//! default_grade: <grade>      grade when no rule matches (optional)
//! ```
//!
//! Blank lines and `#` comments are ignored. `emit_scheme` is the exact inverse
//! of `parse_scheme`, so loading then saving a scheme is a no-op. Ported from
//! the former web client `schemeToText` / `textToScheme`.

use crate::scheme::{GradeRule, GradeScheme, SchemeConst, SchemeVar};

/// Serialize a scheme to the readable DSL.
pub fn emit_scheme(s: &GradeScheme) -> String {
    let mut sections: Vec<String> = Vec::new();

    let mut meta: Vec<String> = Vec::new();
    if !s.total_var.is_empty() {
        meta.push(format!("total_var: {}", s.total_var));
    }
    if !s.default_grade.is_empty() {
        meta.push(format!("default_grade: {}", s.default_grade));
    }
    if !meta.is_empty() {
        sections.push(meta.join("\n"));
    }
    if !s.constants.is_empty() {
        sections.push(
            s.constants
                .iter()
                .map(|c| format!("const {} = {}", c.name, fmt_num(c.value)))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    if !s.vars.is_empty() {
        sections.push(
            s.vars
                .iter()
                .map(|v| format!("{} = {}", v.name, v.expr))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    if !s.rules.is_empty() {
        sections.push(
            s.rules
                .iter()
                .map(|r| format!("when {} -> {}", r.when, r.grade))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    sections.join("\n\n")
}

/// Format a constant value the way JS `${value}` did: integers print without a
/// decimal point, fractions keep their shortest form. Rust's `f64` Display
/// already matches this (`2.0 -> "2"`, `2.5 -> "2.5"`).
fn fmt_num(v: f64) -> String {
    format!("{v}")
}

/// Parse the readable DSL into a scheme, mirroring the former client parser
/// (including its `line N: ...` error messages).
pub fn parse_scheme(text: &str) -> Result<GradeScheme, String> {
    let mut s = GradeScheme::default();
    for (i, raw) in text.split('\n').enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = format!("line {}", i + 1);

        if let Some(rest) = line.strip_prefix("total_var:") {
            s.total_var = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("default_grade:") {
            s.default_grade = rest.trim().to_string();
        } else if let Some(body) = line.strip_prefix("const ") {
            let (name, value) = match body.split_once('=') {
                Some((n, v)) => (n.trim(), v.trim().parse::<f64>().ok()),
                None => ("", None),
            };
            match value {
                Some(value) if !name.is_empty() => {
                    s.constants.push(SchemeConst { name: name.to_string(), value });
                }
                _ => {
                    return Err(format!(
                        "{at}: expected 'const <name> = <number>', got '{line}'"
                    ))
                }
            }
        } else if let Some(body) = line.strip_prefix("when ") {
            match body.split_once("->") {
                Some((when, grade)) if !when.trim().is_empty() && !grade.trim().is_empty() => {
                    s.rules.push(GradeRule {
                        when: when.trim().to_string(),
                        grade: grade.trim().to_string(),
                    });
                }
                _ => {
                    return Err(format!(
                        "{at}: expected 'when <cond> -> <grade>', got '{line}'"
                    ))
                }
            }
        } else if line.contains("->") {
            return Err(format!("{at}: a rule must start with 'when', got '{line}'"));
        } else if let Some((name, expr)) = line.split_once('=') {
            let (name, expr) = (name.trim(), expr.trim());
            if name.is_empty() || expr.is_empty() {
                return Err(format!("{at}: expected '<name> = <expr>', got '{line}'"));
            }
            s.vars.push(SchemeVar { name: name.to_string(), expr: expr.to_string() });
        } else {
            return Err(format!("{at}: unrecognized statement '{line}'"));
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pvt_scheme() -> GradeScheme {
        GradeScheme {
            constants: vec![
                SchemeConst { name: "lenience".into(), value: 2.0 },
                SchemeConst { name: "secmin".into(), value: 10.0 },
            ],
            vars: vec![
                SchemeVar { name: "total".into(), expr: "se_total + hci_total".into() },
            ],
            rules: vec![
                GradeRule { when: "se_total < secmin".into(), grade: "F".into() },
                GradeRule { when: "total >= 45 - lenience".into(), grade: "A".into() },
            ],
            total_var: "total".into(),
            default_grade: "F".into(),
        }
    }

    #[test]
    fn emit_then_parse_round_trips() {
        let s = pvt_scheme();
        let parsed = parse_scheme(&emit_scheme(&s)).unwrap();
        assert_eq!(parsed.constants.len(), 2);
        assert_eq!(parsed.constants[0].name, "lenience");
        assert_eq!(parsed.constants[1].value, 10.0);
        assert_eq!(parsed.vars.len(), 1);
        assert_eq!(parsed.vars[0].expr, "se_total + hci_total");
        assert_eq!(parsed.rules.len(), 2);
        assert_eq!(parsed.rules[0].when, "se_total < secmin");
        assert_eq!(parsed.rules[1].grade, "A");
        assert_eq!(parsed.total_var, "total");
        assert_eq!(parsed.default_grade, "F");
    }

    #[test]
    fn integer_constants_have_no_decimal() {
        let s = GradeScheme {
            constants: vec![SchemeConst { name: "secmin".into(), value: 10.0 }],
            ..Default::default()
        };
        assert_eq!(emit_scheme(&s), "const secmin = 10");
    }

    #[test]
    fn ignores_blanks_and_comments() {
        let parsed = parse_scheme("\n# a comment\n  \ntotal = q1 + q2\n").unwrap();
        assert_eq!(parsed.vars.len(), 1);
        assert_eq!(parsed.vars[0].name, "total");
    }

    #[test]
    fn rule_without_when_is_rejected() {
        let err = parse_scheme("total >= 5 -> A").unwrap_err();
        assert!(err.contains("must start with 'when'"), "{err}");
    }

    #[test]
    fn bad_const_is_rejected() {
        let err = parse_scheme("const x = abc").unwrap_err();
        assert!(err.contains("const <name>"), "{err}");
    }
}
