use openquant::etf_trick::{EtfTrick, Table};

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/etf_trick")
}

fn fixture_path(name: &str) -> String {
    fixture_dir().join(name).to_string_lossy().to_string()
}

fn table(name: &str) -> Table {
    Table::from_csv(&fixture_dir().join(name)).unwrap()
}

#[test]
fn test_etf_trick_costs_defined() {
    let open_path = fixture_path("open_df.csv");
    let close_path = fixture_path("close_df.csv");
    let alloc_path = fixture_path("alloc_df.csv");
    let costs_path = fixture_path("costs_df.csv");
    let rates_path = fixture_path("rates_df.csv");

    let mut csv_etf_trick =
        EtfTrick::from_csv(&open_path, &close_path, &alloc_path, &costs_path, Some(&rates_path))
            .unwrap();
    let in_memory_etf_trick = EtfTrick::from_tables(
        table("open_df.csv"),
        table("close_df.csv"),
        table("alloc_df.csv"),
        table("costs_df.csv"),
        Some(table("rates_df.csv")),
    )
    .unwrap();

    let in_memory = in_memory_etf_trick.get_etf_series(100_000).unwrap();
    let csv_4 = csv_etf_trick.get_etf_series(4).unwrap();
    csv_etf_trick.reset();
    let csv_100 = csv_etf_trick.get_etf_series(100).unwrap();
    csv_etf_trick.reset();
    let csv_all = csv_etf_trick.get_etf_series(1_000_000).unwrap();

    assert_eq!(in_memory.len(), csv_4.len());
    assert_eq!(in_memory.len(), csv_100.len());
    assert_eq!(in_memory.len(), csv_all.len());

    // AFML 2.4.1, from tests/fixtures/etf_trick/generate.py (reference.json, with_rates[20]).
    assert!((in_memory[20].1 - 0.9910955906772763).abs() < 1e-12);
    assert_eq!(in_memory[0].1, 1.0);
    assert_eq!(csv_4[0].1, 1.0);
    assert_eq!(csv_100[0].1, 1.0);
    assert_eq!(csv_all[0].1, 1.0);

    assert_eq!(in_memory.last().unwrap().1, csv_4.last().unwrap().1);
    assert_eq!(in_memory.last().unwrap().1, csv_100.last().unwrap().1);
    assert_eq!(in_memory.last().unwrap().1, csv_all.last().unwrap().1);

    for i in 0..in_memory.len() {
        assert_eq!(in_memory[i].1, csv_4[i].1);
        assert_eq!(in_memory[i].1, csv_100[i].1);
        assert_eq!(in_memory[i].1, csv_all[i].1);
    }
}

#[test]
fn test_etf_trick_rates_not_defined() {
    let open_path = fixture_path("open_df.csv");
    let close_path = fixture_path("close_df.csv");
    let alloc_path = fixture_path("alloc_df.csv");
    let costs_path = fixture_path("costs_df.csv");

    let mut csv_etf_trick =
        EtfTrick::from_csv(&open_path, &close_path, &alloc_path, &costs_path, None).unwrap();
    let in_memory_etf_trick = EtfTrick::from_tables(
        table("open_df.csv"),
        table("close_df.csv"),
        table("alloc_df.csv"),
        table("costs_df.csv"),
        None,
    )
    .unwrap();

    let in_memory = in_memory_etf_trick.get_etf_series(100_000).unwrap();
    let csv_4 = csv_etf_trick.get_etf_series(4).unwrap();
    csv_etf_trick.reset();
    let csv_100 = csv_etf_trick.get_etf_series(100).unwrap();
    csv_etf_trick.reset();
    let csv_all = csv_etf_trick.get_etf_series(1_000_000).unwrap();

    assert_eq!(in_memory.len(), csv_4.len());
    assert_eq!(in_memory.len(), csv_100.len());
    assert_eq!(in_memory.len(), csv_all.len());

    // AFML 2.4.1, from tests/fixtures/etf_trick/generate.py (reference.json, without_rates[20]).
    assert!((in_memory[20].1 - 0.9911139304610538).abs() < 1e-12);
    assert_eq!(in_memory[0].1, 1.0);
    assert_eq!(csv_4[0].1, 1.0);
    assert_eq!(csv_100[0].1, 1.0);
    assert_eq!(csv_all[0].1, 1.0);

    assert_eq!(in_memory.last().unwrap().1, csv_4.last().unwrap().1);
    assert_eq!(in_memory.last().unwrap().1, csv_100.last().unwrap().1);
    assert_eq!(in_memory.last().unwrap().1, csv_all.last().unwrap().1);

    for i in 0..in_memory.len() {
        assert_eq!(in_memory[i].1, csv_4[i].1);
        assert_eq!(in_memory[i].1, csv_100[i].1);
        assert_eq!(in_memory[i].1, csv_all[i].1);
    }
}

#[test]
fn test_input_exceptions() {
    let modified_open = {
        let mut t = table("open_df.csv");
        t.index.push("2020-01-01".to_string());
        t.values.push(vec![4.0; t.columns.len()]);
        t
    };

    let result = EtfTrick::from_tables(
        modified_open,
        table("close_df.csv"),
        table("alloc_df.csv"),
        table("costs_df.csv"),
        None,
    );
    assert!(result.is_err());

    let open_path = fixture_path("open_df.csv");
    let close_path = fixture_path("close_df.csv");
    let alloc_path = fixture_path("alloc_df.csv");
    let costs_path = fixture_path("costs_df.csv");
    let csv_etf_trick =
        EtfTrick::from_csv(&open_path, &close_path, &alloc_path, &costs_path, None).unwrap();
    assert!(csv_etf_trick.get_etf_series(2).is_err());
}

/// The numbers printed on docs-site/src/content/docs/modules/etf-trick.md. The docs gate only
/// compiles Rust examples, so the output shown there is pinned here instead.
#[test]
fn test_docs_page_example_values() {
    use chrono::NaiveDate;
    use openquant::etf_trick::{get_futures_roll_series, FuturesRollRow};

    let days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"];
    let build = |values: [[f64; 2]; 6]| Table {
        index: days.iter().map(|d| d.to_string()).collect(),
        columns: vec!["CL".to_string(), "NG".to_string()],
        values: values.iter().map(|v| v.to_vec()).collect(),
    };
    let etf = EtfTrick::from_tables(
        build([[70.0, 2.50], [70.5, 2.52], [71.4, 2.49], [71.0, 2.55], [72.2, 2.60], [72.0, 2.58]]),
        build([[70.4, 2.51], [71.2, 2.50], [71.1, 2.54], [72.0, 2.61], [72.1, 2.57], [72.6, 2.59]]),
        build([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]),
        build([[0.0, 0.0]; 6]),
        None,
    )
    .unwrap();
    let series = etf.get_etf_series(100).unwrap();
    let values: Vec<f64> = series.iter().map(|(_, k)| *k).collect();
    assert_eq!(series[0].0, "2024-01-03");
    for (got, want) in values.iter().zip([1.0, 1.007939, 1.028298, 1.024786]) {
        assert!((got - want).abs() < 5e-7, "got {got}, page says {want}");
    }
    // The hand calculation on the page: bought at the 01-04 opens, so 01-04 earns open-to-close.
    let h1 = [0.5 / 71.4, 0.5 / 2.49];
    let by_hand = 1.0 + h1[0] * (71.1 - 71.4) + h1[1] * (2.54 - 2.49);
    assert!((values[1] - by_hand).abs() < 1e-12);
    // 01-05 is not after a rebalance: the same holdings earn close-to-close.
    let k2 = by_hand + h1[0] * (72.0 - 71.1) + h1[1] * (2.61 - 2.54);
    assert!((values[2] - k2).abs() < 1e-12);
    // 01-05 rebalances to 80/20, sized at the 01-08 opens; 01-08 earns open-to-close.
    let h3 = [0.8 * k2 / 72.2, 0.2 * k2 / 2.60];
    let k3 = k2 + h3[0] * (72.1 - 72.2) + h3[1] * (2.57 - 2.60);
    assert!((values[3] - k3).abs() < 1e-12);

    let chain: Vec<FuturesRollRow> = [
        (2, 70.0, 70.4, "CLG4"),
        (3, 70.5, 71.2, "CLG4"),
        (4, 71.4, 71.1, "CLG4"),
        (5, 72.0, 72.9, "CLH4"),
        (8, 73.1, 73.0, "CLH4"),
    ]
    .into_iter()
    .map(|(day, open, close, contract)| FuturesRollRow {
        date: NaiveDate::from_ymd_opt(2024, 1, day).unwrap(),
        open,
        close,
        security: contract.to_string(),
        current_security: contract.to_string(),
    })
    .collect();
    let gaps = get_futures_roll_series(&chain, "absolute", true).unwrap();
    for (got, want) in gaps.iter().zip([-0.9, -0.9, -0.9, 0.0, 0.0]) {
        assert!((got - want).abs() < 1e-9, "got {got}, page says {want}");
    }
}

fn reference() -> serde_json::Value {
    let path = fixture_dir().join("reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The ETF trick as AFML 2.4.1 writes it, computed in pandas by tests/fixtures/etf_trick/generate.py.
/// Issue #164: the holdings that earn bar t are h_{t-1} = w_{t-1} K_{t-1} / (o_t phi_{t-1} sum|w|).
#[test]
fn etf_series_matches_afml_reference() {
    let reference = reference();
    let dates: Vec<&str> = reference["etf_trick"]["dates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for (rates, key) in [(Some(table("rates_df.csv")), "with_rates"), (None, "without_rates")] {
        let etf = EtfTrick::from_tables(
            table("open_df.csv"),
            table("close_df.csv"),
            table("alloc_df.csv"),
            table("costs_df.csv"),
            rates,
        )
        .unwrap();
        let got = etf.get_etf_series(100_000).unwrap();
        let want = reference["etf_trick"][key].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{key}");
        for (i, ((date, k), w)) in got.iter().zip(want).enumerate() {
            assert_eq!(date, dates[i], "{key} row {i}");
            let w = w.as_f64().unwrap();
            assert!((k - w).abs() < 1e-12, "{key} row {i} ({date}): got {k}, AFML {w}");
        }
    }
}
