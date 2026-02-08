use openquant::filters::{cusum_filter_indices, Threshold};

let close = vec![80.0, 80.3, 80.1, 80.6, 80.2, 80.8, 80.5, 80.9];
let events = cusum_filter_indices(&close, Threshold::Scalar(0.001));
assert!(!events.is_empty());
events
