use openquant::sampling::{get_ind_matrix, seq_bootstrap};

let label_endtime = vec![(0, 3), (2, 5), (4, 7)];
let bars = vec![0, 1, 2, 3, 4, 5, 6, 7];
let ind = get_ind_matrix(&label_endtime, &bars);
let sample_idx = seq_bootstrap(&ind, Some(5), Some(vec![0, 1]));
assert_eq!(sample_idx.len(), 5);
sample_idx
