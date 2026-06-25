//! Embedding-based sample selection. Only two strategies are supported:
//! `random` and `maximin` (max-spread / farthest-first), ported from
//! `maximin_sampling` in the legacy `sampling.py`.

use rand::seq::SliceRandom;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    Random,
    Maximin,
}

impl Algorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            Algorithm::Random => "random",
            Algorithm::Maximin => "maximin",
        }
    }
}

/// Random selection of `n` ids (all if `n >= len`).
pub fn random_sample(ids: &[String], n: usize) -> Vec<String> {
    if n >= ids.len() {
        return ids.to_vec();
    }
    let mut rng = rand::thread_rng();
    let mut chosen: Vec<String> = ids
        .choose_multiple(&mut rng, n)
        .cloned()
        .collect();
    chosen.sort();
    chosen
}

/// Maximin (farthest-first) selection over feature vectors.
///
/// Mirrors the legacy implementation: z-score standardize each dimension, seed
/// with the point closest to the (scaled) centroid, then greedily add the point
/// that maximizes the minimum distance to the already-selected set.
pub fn maximin_sample(data: &[(String, Vec<f32>)], n: usize) -> Vec<String> {
    let n_total = data.len();
    if n == 0 {
        return Vec::new();
    }
    if n >= n_total {
        return data.iter().map(|(id, _)| id.clone()).collect();
    }

    let dim = data.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
    if dim == 0 {
        return data.iter().take(n).map(|(id, _)| id.clone()).collect();
    }

    // Matrix of f64, rows padded/truncated to `dim`.
    let mut x: Vec<Vec<f64>> = data
        .iter()
        .map(|(_, v)| {
            let mut row = vec![0.0f64; dim];
            for (j, val) in v.iter().take(dim).enumerate() {
                row[j] = *val as f64;
            }
            row
        })
        .collect();

    // StandardScaler (population std, scale=1 where std==0).
    for j in 0..dim {
        let mean = x.iter().map(|r| r[j]).sum::<f64>() / n_total as f64;
        let var = x.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / n_total as f64;
        let std = var.sqrt();
        let scale = if std == 0.0 { 1.0 } else { std };
        for r in &mut x {
            r[j] = (r[j] - mean) / scale;
        }
    }

    // Seed: point closest to centroid (≈origin after scaling).
    let centroid: Vec<f64> = (0..dim)
        .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n_total as f64)
        .collect();
    let first = (0..n_total)
        .min_by(|&a, &b| {
            dist(&x[a], &centroid)
                .partial_cmp(&dist(&x[b], &centroid))
                .unwrap()
        })
        .unwrap();

    let mut selected = vec![first];
    while selected.len() < n {
        let mut best_idx = usize::MAX;
        let mut best_min = f64::NEG_INFINITY;
        for idx in 0..n_total {
            if selected.contains(&idx) {
                continue;
            }
            let min_d = selected
                .iter()
                .map(|&s| dist(&x[idx], &x[s]))
                .fold(f64::INFINITY, f64::min);
            if min_d > best_min {
                best_min = min_d;
                best_idx = idx;
            }
        }
        if best_idx == usize::MAX {
            break;
        }
        selected.push(best_idx);
    }

    selected.into_iter().map(|i| data[i].0.clone()).collect()
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_caps_at_len() {
        let ids: Vec<String> = (0..3).map(|i| i.to_string()).collect();
        assert_eq!(random_sample(&ids, 10).len(), 3);
        assert_eq!(random_sample(&ids, 2).len(), 2);
    }

    #[test]
    fn maximin_returns_n_and_spreads() {
        // Three tight clusters around 0, 10, 20 in 1D; expect one per cluster.
        let data: Vec<(String, Vec<f32>)> = vec![
            ("a".into(), vec![0.0]),
            ("b".into(), vec![0.1]),
            ("c".into(), vec![10.0]),
            ("d".into(), vec![10.1]),
            ("e".into(), vec![20.0]),
            ("f".into(), vec![20.1]),
        ];
        let picked = maximin_sample(&data, 3);
        assert_eq!(picked.len(), 3);
        // Should span the low, mid and high clusters rather than duplicate one.
        let vals: Vec<f32> = picked
            .iter()
            .map(|id| data.iter().find(|(d, _)| d == id).unwrap().1[0])
            .collect();
        let has_low = vals.iter().any(|&v| v < 5.0);
        let has_mid = vals.iter().any(|&v| (5.0..15.0).contains(&v));
        let has_high = vals.iter().any(|&v| v >= 15.0);
        assert!(has_low && has_mid && has_high, "got {vals:?}");
    }

    #[test]
    fn maximin_all_when_n_ge_len() {
        let data: Vec<(String, Vec<f32>)> =
            vec![("a".into(), vec![1.0]), ("b".into(), vec![2.0])];
        assert_eq!(maximin_sample(&data, 5).len(), 2);
    }
}
