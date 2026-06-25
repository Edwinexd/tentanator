//! Grade parsing: a grade is a plain number or a signed sum of subpart scores,
//! e.g. "5", "7.5", "-0.5", "2+1.5+2.5", "2+2.5-0.5". Whitespace is ignored.
//! Ported from `evaluate_grade` / `validate_grade` in the legacy `tentanator.py`.

/// Evaluate a grade expression into its numeric total, or `None` if malformed.
pub fn evaluate_grade(grade_str: &str) -> Option<f64> {
    if grade_str.is_empty() {
        return None;
    }
    let s: String = grade_str.chars().filter(|c| !c.is_whitespace()).collect();
    if s.is_empty() {
        return None;
    }
    let bytes = s.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    let mut total = 0.0f64;
    let mut matched_any = false;

    while i < n {
        let start = i;
        let mut sign = 1.0f64;
        if bytes[i] == b'+' || bytes[i] == b'-' {
            if bytes[i] == b'-' {
                sign = -1.0;
            }
            i += 1;
        }
        let num_start = i;
        let mut has_int = false;
        while i < n && bytes[i].is_ascii_digit() {
            i += 1;
            has_int = true;
        }
        let mut has_frac = false;
        if i < n && bytes[i] == b'.' {
            i += 1;
            while i < n && bytes[i].is_ascii_digit() {
                i += 1;
                has_frac = true;
            }
        }
        // Valid number term: `\d+\.?\d*` (needs an integer part) or `\.\d+`.
        let starts_with_dot = num_start < n && bytes[num_start] == b'.';
        let valid = has_int || (starts_with_dot && has_frac);
        if !valid || i == start {
            return None;
        }
        let mut numpart = s[num_start..i].to_string();
        if numpart.starts_with('.') {
            numpart.insert(0, '0');
        }
        if numpart.ends_with('.') {
            numpart.push('0');
        }
        match numpart.parse::<f64>() {
            Ok(v) => total += sign * v,
            Err(_) => return None,
        }
        matched_any = true;
    }

    if !matched_any || i != n {
        return None;
    }
    Some(total)
}

/// Validate a grade string. Returns the evaluated total on success, or a
/// human-readable error message.
pub fn validate_grade(grade_str: &str) -> Result<f64, String> {
    if grade_str.trim().is_empty() {
        return Err("Grade cannot be empty".to_string());
    }
    evaluate_grade(grade_str).ok_or_else(|| {
        format!(
            "Invalid grade '{grade_str}' - must be numeric or a signed sum \
             (e.g. 0, 7.5, 2+1.5+2.5, 2+2.5-0.5)"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Option<f64>, b: f64) -> bool {
        a.map(|v| (v - b).abs() < 1e-9).unwrap_or(false)
    }

    #[test]
    fn plain_numbers() {
        assert!(approx(evaluate_grade("5"), 5.0));
        assert!(approx(evaluate_grade("7.5"), 7.5));
        assert!(approx(evaluate_grade("-0.5"), -0.5));
        assert!(approx(evaluate_grade("0"), 0.0));
        assert!(approx(evaluate_grade(".5"), 0.5));
    }

    #[test]
    fn signed_sums() {
        assert!(approx(evaluate_grade("2+1.5+2.5"), 6.0));
        assert!(approx(evaluate_grade("2+2.5+2.5-0.5"), 6.5));
        assert!(approx(evaluate_grade(" 2 + 2.5 - 0.5 "), 4.0));
    }

    #[test]
    fn malformed() {
        assert_eq!(evaluate_grade(""), None);
        assert_eq!(evaluate_grade("abc"), None);
        assert_eq!(evaluate_grade("5/2"), None);
        assert_eq!(evaluate_grade("2++2"), None);
        assert_eq!(evaluate_grade("5 points"), None);
    }

    #[test]
    fn validate() {
        assert!(validate_grade("2+1.5").is_ok());
        assert!(validate_grade("").is_err());
        assert!(validate_grade("nope").is_err());
    }
}
