use openquant::ef3m::*;

#[test]
fn test_m2n_constructor() {
    let moments = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let m2n = M2N::with_defaults(moments.clone());
    assert_eq!(m2n.moments, moments);
    assert_eq!(m2n.new_moments, vec![0.0; 5]);
    assert_eq!(m2n.parameters, vec![0.0; 5]);
    assert!((m2n.error - 55.0).abs() < 1e-12);
}

#[test]
fn test_get_moments() {
    let (u1, u2, s1, s2, p1): (f64, f64, f64, f64, f64) = (2.1, 4.3, 1.1, 0.7, 0.3);
    let p2 = 1.0 - p1;
    let m1 = p1 * u1 + p2 * u2;
    let m2 = p1 * (s1 * s1 + u1 * u1) + p2 * (s2 * s2 + u2 * u2);
    let m3 = p1 * (3.0 * s1 * s1 * u1 + u1.powi(3)) + p2 * (3.0 * s2 * s2 * u2 + u2.powi(3));
    let m4 = p1 * (3.0 * s1.powi(4) + 6.0 * s1 * s1 * u1 * u1 + u1.powi(4))
        + p2 * (3.0 * s2.powi(4) + 6.0 * s2 * s2 * u2 * u2 + u2.powi(4));
    let m5 = p1 * (15.0 * s1.powi(4) * u1 + 10.0 * s1 * s1 * u1.powi(3) + u1.powi(5))
        + p2 * (15.0 * s2.powi(4) * u2 + 10.0 * s2 * s2 * u2.powi(3) + u2.powi(5));
    let test_params = vec![u1, u2, s1, s2, p1];
    let expected = vec![m1, m2, m3, m4, m5];

    let mut m2n = M2N::with_defaults(expected.clone());
    let _ = m2n.get_moments(&test_params, false);
    for (a, b) in m2n.new_moments.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-12);
    }

    let got = m2n.get_moments(&test_params, true).unwrap();
    for (a, b) in got.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn test_iter_4_checks_and_success() {
    assert!(M2N::with_defaults(vec![1.0, 2.0, 3.0, 4.0, 5.0]).iter_4(3.0, 1.0).is_empty());
    assert!(M2N::with_defaults(vec![2.0, 2.0, 3.0, 4.0, 5.0]).iter_4(1.0, 0.8).is_empty());
    assert!(M2N::with_defaults(vec![1.5, 2.0, 3.0, 4.0, 5.0]).iter_4(2.0, 0.7).is_empty());
    assert!(M2N::with_defaults(vec![0.0, 0.1, 0.0, 0.0, 5.0]).iter_4(0.1, 0.5).is_empty());
    assert!(M2N::with_defaults(vec![0.0, 0.1, 0.0, 0.0, 5.0]).iter_4(0.1, 0.25).is_empty());

    let ok = M2N::with_defaults(vec![0.7, 2.6, 0.4, 25.0, -59.8]).iter_4(1.0, 0.2);
    assert_eq!(ok.len(), 5);
}

#[test]
fn test_iter_5_checks_and_success() {
    assert!(M2N::with_defaults(vec![0.0; 5]).iter_5(0.0, 0.05).is_empty());
    assert!(M2N::with_defaults(vec![0.0; 5]).iter_5(0.1, 0.05).is_empty());
    assert!(M2N::with_defaults(vec![0.0, 0.0, 0.1, 0.0, 0.0]).iter_5(0.1, 0.2).is_empty());
    assert!(M2N::with_defaults(vec![0.0, 0.1, 0.0, 0.0, 0.0]).iter_5(0.1, 0.99999).is_empty());
    assert!(M2N::with_defaults(vec![0.0, 0.1, 0.0, 0.0, 0.0]).iter_5(0.1, 0.95).is_empty());
    assert!(M2N::with_defaults(vec![0.0, 0.1, 0.1, 0.0, 0.2]).iter_5(0.4, 0.95).is_empty());
    assert!(M2N::with_defaults(vec![
        1.7486117351052706,
        12.30094642908807,
        44.14804719610457,
        301.66990880582324,
        1389.7073066865096
    ])
    .iter_5(8.927498436080297, -1910484717784700.2)
    .is_empty());
    assert!(M2N::with_defaults(vec![
        1.7465392043495434,
        12.32010406019726,
        44.3090981635415,
        302.3152423573811,
        1403.0640473698527
    ])
    .iter_5(1.8733475857864539, 0.019291066689915537)
    .is_empty());

    let ok = M2N::with_defaults(vec![0.7, 2.6, 0.4, 25.0, -59.8])
        .iter_5(0.8642146104188053, 0.03296760034110158);
    assert_eq!(ok.len(), 5);
}

#[test]
fn test_fit_variants_and_paths() {
    let moments = vec![0.7, 2.6, 0.4, 25.0, -59.8];

    let mut m1 = M2N::new(moments.clone(), 1e-5, 5.0, 5, 1, 10_000, -1);
    assert!(m1.fit(1.0).is_ok());
    assert_eq!(m1.parameters.len(), 5);

    let mut m2 = M2N::new(moments.clone(), 1e-5, 5.0, 5, 2, 10_000, -1);
    assert!(m2.fit(1.0).is_ok());
    assert_eq!(m2.parameters.len(), 5);

    let mut bad = M2N::new(moments.clone(), 1e-5, 5.0, 5, 3, 10_000, -1);
    assert!(bad.fit(1.0).is_err());

    let mut via_error = M2N::new(moments.clone(), 1e-5, 5.0, 5, 1, 10_000, -1);
    via_error.error = 1e6;
    assert!(via_error.fit(1.0).is_ok());
    assert_eq!(via_error.parameters.len(), 5);

    let mut via_eps = M2N::new(moments.clone(), 1e12, 5.0, 5, 1, 10_000, -1);
    assert!(via_eps.fit(1.0).is_ok());
    assert_eq!(via_eps.parameters.len(), 5);

    let mut via_iter = M2N::new(moments, 1e-12, 5.0, 5, 1, 1, -1);
    assert!(via_iter.fit(1.0).is_ok());
    assert_eq!(via_iter.parameters.len(), 5);
}

#[test]
fn test_single_fit_loop_and_mp_fit_types() {
    let moments = vec![0.7, 2.6, 0.4, 25.0, -59.8];

    let mut s = M2N::new(moments.clone(), 1e-2, 5.0, 3, 2, 1000, 1);
    let out = s.single_fit_loop(None);
    assert!(out.len() <= 1);

    let m = M2N::new(moments, 1e-2, 5.0, 3, 2, 1000, 1);
    let out_mp = m.mp_fit();
    assert!(out_mp.len() <= 3);
}

#[test]
fn test_centered_moment_result() {
    let raw = vec![0.701756, 2.591815, 0.450519, 24.689030, -57.756735];
    let mut centered_5th_correct = 0.0;
    for j in 0..=5 {
        let add_on = if j == 5 { 1.0 } else { raw[5 - j - 1] };
        centered_5th_correct += (-1.0f64).powi(j as i32) * add_on * raw[0].powi(j as i32) * {
            let mut c = 1.0;
            for i in 0..j {
                c *= (5 - i) as f64 / (i + 1) as f64;
            }
            c
        };
    }
    let centered_5th = centered_moment(&raw, 5);
    assert!((centered_5th - centered_5th_correct).abs() < 1e-7);
}

#[test]
fn test_raw_moment_result() {
    let centered = vec![0.0, 2.11, -4.373999999999999, 30.803699999999996, -153.58572];
    let raw = raw_moment(&centered, 0.7);
    let expected = [0.7, 2.6, 0.4, 25.0, -59.8];
    assert_eq!(raw.len(), expected.len());
    for (a, b) in raw.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-7);
    }
}

#[test]
fn test_most_likely_parameters_result() {
    let rows = vec![
        FitResultRow {
            mu_1: -2.074149682208028,
            mu_2: 0.9958122958772418,
            sigma_1: 1.9764097851543956,
            sigma_2: 1.002964090440232,
            p_1: 0.09668610445835334,
            error: 0.1,
        },
        FitResultRow {
            mu_1: -2.1464760973734522,
            mu_2: 0.9927128514876395,
            sigma_1: 1.9516780127056625,
            sigma_2: 1.0054392587806025,
            p_1: 0.09379917992315062,
            error: 0.2,
        },
        FitResultRow {
            mu_1: -1.7318027625411423,
            mu_2: 1.013574632526087,
            sigma_1: 2.080573657129795,
            sigma_2: 0.9872577865302316,
            p_1: 0.11351960785118335,
            error: 0.3,
        },
        FitResultRow {
            mu_1: -1.7799163398785354,
            mu_2: 1.0065707257309104,
            sigma_1: 2.071328499049906,
            sigma_2: 0.9909001363163131,
            p_1: 0.10993400151299484,
            error: 0.4,
        },
        FitResultRow {
            mu_1: -1.9766582333677596,
            mu_2: 1.009533655971151,
            sigma_1: 1.9988591140726848,
            sigma_2: 0.9971327048101786,
            p_1: 0.10264463363929438,
            error: 0.5,
        },
    ];

    let out = most_likely_parameters(&rows, None, 10_000);
    assert!(out.contains_key("mu_1"));
    assert!(out.contains_key("mu_2"));
    assert!(out.contains_key("sigma_1"));
    assert!(out.contains_key("sigma_2"));
    assert!(out.contains_key("p_1"));
}

#[test]
fn test_most_likely_parameters_ignore_columns_list() {
    let rows = vec![
        FitResultRow { mu_1: -2.0, mu_2: 1.0, sigma_1: 2.0, sigma_2: 1.0, p_1: 0.1, error: 0.1 },
        FitResultRow {
            mu_1: -2.1,
            mu_2: 1.01,
            sigma_1: 1.95,
            sigma_2: 1.01,
            p_1: 0.09,
            error: 0.2,
        },
    ];
    let out = most_likely_parameters(&rows, Some(&["error"]), 1_000);
    assert!(!out.contains_key("error"));
    assert!(out.contains_key("mu_1"));
    assert!(out.contains_key("p_1"));
}
