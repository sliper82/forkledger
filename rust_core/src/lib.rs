//! ForkLedger Rust Core — v0.4.0
//! Performance-critical scoring, regret, and clustering functions.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::collections::HashMap;

#[inline]
fn numeric_proximity(a: f64, b: f64) -> f64 {
    if (a - b).abs() < 1e-10 { return 1.0; }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    (1.0 - diff / scale).max(0.0)
}

fn map_overlap_rust(a: &HashMap<String, String>, b: &HashMap<String, String>,
                    a_num: &HashMap<String, f64>, b_num: &HashMap<String, f64>) -> f64 {
    let mut all_keys: Vec<&str> = a.keys().map(String::as_str).collect();
    for k in b.keys() {
        if !a.contains_key(k.as_str()) { all_keys.push(k.as_str()); }
    }
    if all_keys.is_empty() { return 0.0; }
    let mut score = 0.0f64;
    for key in &all_keys {
        let in_a = a.contains_key(*key);
        let in_b = b.contains_key(*key);
        if !in_a || !in_b { continue; }
        match (a_num.get(*key), b_num.get(*key)) {
            (Some(&fa), Some(&fb)) => score += numeric_proximity(fa, fb),
            _ => if a.get(*key) == b.get(*key) { score += 1.0; }
        }
    }
    score / all_keys.len() as f64
}

/// Compute state similarity with numeric proximity support.
#[pyfunction]
fn state_overlap(a: &Bound<'_, PyDict>, b: &Bound<'_, PyDict>) -> PyResult<f64> {
    let a_map: HashMap<String,String> = a.iter().map(|(k,v)|(k.to_string(),v.to_string())).collect();
    let b_map: HashMap<String,String> = b.iter().map(|(k,v)|(k.to_string(),v.to_string())).collect();
    let a_num: HashMap<String,f64> = a.iter().filter_map(|(k,v)| v.extract::<f64>().ok().map(|f|(k.to_string(),f))).collect();
    let b_num: HashMap<String,f64> = b.iter().filter_map(|(k,v)| v.extract::<f64>().ok().map(|f|(k.to_string(),f))).collect();
    Ok(map_overlap_rust(&a_map, &b_map, &a_num, &b_num))
}

/// Parallel batch scoring of N record states vs 1 query state.
#[pyfunction]
fn batch_state_overlap(py: Python<'_>, record_states: &Bound<'_, PyList>, query_state: &Bound<'_, PyDict>) -> PyResult<Vec<f64>> {
    let q_map: HashMap<String,String> = query_state.iter().map(|(k,v)|(k.to_string(),v.to_string())).collect();
    let q_num: HashMap<String,f64> = query_state.iter().filter_map(|(k,v)| v.extract::<f64>().ok().map(|f|(k.to_string(),f))).collect();

    let states: Vec<(HashMap<String,String>, HashMap<String,f64>)> = record_states.iter().map(|item| {
        match item.downcast::<PyDict>() {
            Ok(d) => {
                let s: HashMap<String,String> = d.iter().map(|(k,v)|(k.to_string(),v.to_string())).collect();
                let n: HashMap<String,f64> = d.iter().filter_map(|(k,v)| v.extract::<f64>().ok().map(|f|(k.to_string(),f))).collect();
                (s, n)
            }
            Err(_) => (HashMap::new(), HashMap::new()),
        }
    }).collect();

    let scores: Vec<f64> = py.allow_threads(move || {
        states.par_iter()
            .map(|(s, n)| map_overlap_rust(s, &q_map, n, &q_num))
            .collect()
    });
    Ok(scores)
}

/// Compute regret vector for one record.
#[pyfunction]
fn compute_regret_fast(chosen_branch: &str, realized_value: f64, estimated: &Bound<'_, PyDict>) -> PyResult<HashMap<String, f64>> {
    let mut vals: HashMap<String,f64> = estimated.iter()
        .map(|(k,v)| Ok((k.to_string(), v.extract::<f64>()?)))
        .collect::<PyResult<_>>()?;
    vals.insert(chosen_branch.to_string(), realized_value);
    if vals.is_empty() { return Ok(HashMap::new()); }
    let best = vals.values().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(vals.into_iter().map(|(b,v)| (b, ((best-v)*1e6).round()/1e6)).collect())
}

/// CFR-style weighted regret accumulation.
#[pyfunction]
fn accumulate_regret_weighted(records: &Bound<'_, PyList>, weights: Vec<f64>) -> PyResult<HashMap<String, f64>> {
    let mut acc: HashMap<String,f64> = HashMap::new();
    for (item, &w) in records.iter().zip(weights.iter()) {
        let d = item.downcast::<PyDict>()?;
        for (branch, regret) in d.iter() {
            let r: f64 = regret.extract()?;
            *acc.entry(branch.to_string()).or_insert(0.0) += r * w;
        }
    }
    Ok(acc.into_iter().map(|(k,v)| (k, (v*1e6).round()/1e6)).collect())
}

/// Greedy fuzzy clustering of JSON state strings.
#[pyfunction]
fn fuzzy_cluster_states(states: Vec<String>, threshold: f64) -> PyResult<Vec<usize>> {
    let parsed: Vec<HashMap<String,String>> = states.iter()
        .map(|s| serde_json::from_str(s).unwrap_or_default())
        .collect();
    let empty: HashMap<String,f64> = HashMap::new();
    let mut centroids: Vec<HashMap<String,String>> = Vec::new();
    let mut assignments: Vec<usize> = Vec::new();
    for state in &parsed {
        let state_num: HashMap<String,f64> = HashMap::new();
        let mut best_idx = usize::MAX;
        let mut best_sim = -1.0f64;
        for (i, c) in centroids.iter().enumerate() {
            let sim = map_overlap_rust(state, c, &state_num, &empty);
            if sim > best_sim { best_sim = sim; best_idx = i; }
        }
        if best_sim >= threshold && best_idx != usize::MAX {
            assignments.push(best_idx);
            let updated: HashMap<String,String> = centroids[best_idx].iter()
                .filter(|(k,v)| state.get(k.as_str()) == Some(v))
                .map(|(k,v)| (k.clone(),v.clone()))
                .collect();
            centroids[best_idx] = updated;
        } else {
            assignments.push(centroids.len());
            centroids.push(state.clone());
        }
    }
    Ok(assignments)
}

#[pymodule]
fn forkledger_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(state_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(batch_state_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(compute_regret_fast, m)?)?;
    m.add_function(wrap_pyfunction!(accumulate_regret_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy_cluster_states, m)?)?;
    m.add("__version__", "0.4.0")?;
    Ok(())
}
