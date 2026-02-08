use nalgebra::DMatrix;
use openquant::portfolio_optimization::{allocate_inverse_variance, allocate_max_sharpe, allocate_min_vol};

let prices = DMatrix::from_row_slice(
    6,
    3,
    &[
        100.0, 50.0, 30.0,
        100.2, 50.1, 30.1,
        100.1, 50.3, 30.0,
        100.4, 50.4, 30.2,
        100.5, 50.6, 30.3,
        100.7, 50.5, 30.4,
    ],
);
let _ivp = allocate_inverse_variance(&prices).unwrap();
let _mv = allocate_min_vol(&prices, None, None).unwrap();
let msr = allocate_max_sharpe(&prices, 0.0, None, None).unwrap();
assert!((msr.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
msr
