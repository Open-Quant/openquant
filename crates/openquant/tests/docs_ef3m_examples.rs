//! The Rust example on the ef3m docs page, run as a test. `check:examples` only compiles page
//! snippets; this executes it.
//!
//! The body is the page's snippet, reformatted by rustfmt. If you change one, change the other.

#[test]
fn ef3m_page() -> Result<(), Box<dyn std::error::Error>> {
    use openquant::ef3m::{centered_moment, most_likely_parameters, raw_moment, M2N};

    // The exact raw moments of 0.7 N(-1, 1) + 0.3 N(2, 0.5^2). Parameters are ordered
    // [mu_1, mu_2, sigma_1, sigma_2, p_1].
    let truth = [-1.0, 2.0, 1.0, 0.5, 0.7];
    let moments = M2N::with_defaults(vec![]).get_moments(&truth, true).unwrap();
    assert!((moments[0] + 0.1).abs() < 1e-12); // 0.7 * -1 + 0.3 * 2

    // Raw and centred moments convert both ways. The first centred moment is 0.
    let central = (1..=5).map(|k| centered_moment(&moments, k)).collect::<Result<Vec<_>, _>>()?;
    assert!(central[0].abs() < 1e-12);
    assert!((central[1] - 2.665).abs() < 1e-12); // the variance
    let round_trip = raw_moment(&central, moments[0]);
    assert!(round_trip.iter().zip(&moments).all(|(a, b)| (a - b).abs() < 1e-9));

    // 25 runs of the five-moment fit (variant 2), then the mode of each parameter.
    let m2n = M2N::new(moments.clone(), 1e-4, 5.0, 25, 2, 100_000, 1);
    let rows = m2n.mp_fit()?;
    assert_eq!(rows.len(), 25);
    let fit = most_likely_parameters(&rows, None, 10_000);
    for (name, want) in
        [("mu_1", -1.0), ("mu_2", 2.0), ("sigma_1", 1.0), ("sigma_2", 0.5), ("p_1", 0.7)]
    {
        assert!((fit[name] - want).abs() < 0.02, "{name} = {}", fit[name]);
    }

    // Each row's error is the squared moment error of that row's own parameters.
    let mut check = M2N::with_defaults(moments.clone());
    for row in &rows {
        let implied = check
            .get_moments(&[row.mu_1, row.mu_2, row.sigma_1, row.sigma_2, row.p_1], true)
            .unwrap();
        let error: f64 = moments.iter().zip(&implied).map(|(a, b)| (a - b).powi(2)).sum();
        assert!((error - row.error).abs() < 1e-9);
    }
    Ok(())
}
