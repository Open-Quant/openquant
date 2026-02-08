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

    assert!((in_memory[20].1 - 0.9933502).abs() < 1e-6);
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

    assert!((in_memory[20].1 - 0.9933372).abs() < 1e-6);
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
