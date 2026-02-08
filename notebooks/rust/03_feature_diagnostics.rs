use openquant::feature_importance::feature_pca_analysis;

let x = vec![
    vec![0.1, 1.2, 0.7],
    vec![0.2, 1.1, 0.9],
    vec![0.3, 1.0, 0.8],
    vec![0.4, 0.9, 1.0],
];
let fi = vec![0.5, 0.3, 0.2];
let pca = feature_pca_analysis(&x, &fi, 0.95).unwrap();
pca
