//! The example on docs-site/src/content/docs/modules/etf-trick.md, kept here so that its output
//! is produced by running it: `cargo run -p openquant --example docs_etf_trick`.

use chrono::NaiveDate;
use openquant::etf_trick::{get_futures_roll_series, EtfTrick, FuturesRollRow, Table};

fn table(columns: &[&str], rows: &[(&str, [f64; 2])]) -> Table {
    Table {
        index: rows.iter().map(|(day, _)| day.to_string()).collect(),
        columns: columns.iter().map(|c| c.to_string()).collect(),
        values: rows.iter().map(|(_, v)| v.to_vec()).collect(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Two futures, six sessions. The basket is 50/50 until the fourth session, then 80/20.
    let cols = ["CL", "NG"];
    let days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"];
    let open = [[70.0, 2.50], [70.5, 2.52], [71.4, 2.49], [71.0, 2.55], [72.2, 2.60], [72.0, 2.58]];
    let close =
        [[70.4, 2.51], [71.2, 2.50], [71.1, 2.54], [72.0, 2.61], [72.1, 2.57], [72.6, 2.59]];
    let alloc = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]];
    let zeros = [[0.0, 0.0]; 6];

    let rows = |v: &[[f64; 2]; 6]| days.iter().copied().zip(v.iter().copied()).collect::<Vec<_>>();
    let etf = EtfTrick::from_tables(
        table(&cols, &rows(&open)),
        table(&cols, &rows(&close)),
        table(&cols, &rows(&alloc)),
        table(&cols, &rows(&zeros)), // carry / dividends, in price units; none here
        None,                        // no FX: every contract is quoted in the account currency
    )?;
    for (day, value) in etf.get_etf_series(100)? {
        println!("{day}  K = {value:.6}");
    }

    // One contract chain: CLG4 until the roll on 01-05, CLH4 after. The new contract opens
    // 0.90 above the old one's last close, and that gap is not a return.
    let d = |day| NaiveDate::from_ymd_opt(2024, 1, day).unwrap();
    let chain: Vec<FuturesRollRow> = [
        (2, 70.0, 70.4, "CLG4"),
        (3, 70.5, 71.2, "CLG4"),
        (4, 71.4, 71.1, "CLG4"),
        (5, 72.0, 72.9, "CLH4"),
        (8, 73.1, 73.0, "CLH4"),
    ]
    .into_iter()
    .map(|(day, open, close, contract)| FuturesRollRow {
        date: d(day),
        open,
        close,
        security: contract.to_string(),
        current_security: contract.to_string(),
    })
    .collect();

    let gaps = get_futures_roll_series(&chain, "absolute", true)?;
    println!();
    for (row, gap) in chain.iter().zip(&gaps) {
        println!(
            "{}  {}  close {:.2}  gap {:+.2}  rolled {:.2}",
            row.date,
            row.security,
            row.close,
            gap,
            row.close - gap
        );
    }
    Ok(())
}
