#

sub unmasking(questioned, known, comparison[], 
		learning_fn, feature_importance_fn, 
		num_features_to_drop, min_features)

-->
Array of Arrays of cross-validation accuracy values (between 0 and 1)
[ Q vs. K,
  Q vs. C[0],
  Q vs. C[1],
  ...
  Q vs. C[n] ]
