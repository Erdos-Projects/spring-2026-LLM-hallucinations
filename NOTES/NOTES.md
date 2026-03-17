# PROBLEMS WITH RUNNING STATISTICS TESTS OR TRAINING ON DOMAINS

In our benchmarks, the domain labels are assigned by the JUDGE, and are not ground-truth metadata. 

This causes four concrete issues:

## Label noise: 
14–36% of questions have inconsistent domain labels across their 20 responses. Majority-vote may help but does not eliminate noise.

## Uniform sampling: 
We draw 500 questions uniformly per benchmark, not stratified by domain. The domain distribution simply reflects whatever topics happen to be in each benchmark's pool.

## Domain quantity: 
TruthfulQA has 90 unique raw domain strings. Many domains have fewer than 10 questions, making per-domain classifiers unreliable.

## Spurious correlations: 
the judge may assign certain domain labels (e.g. "Science") when the question is ambiguous or the response is uncertain, creating an artificial correlation between domain and hallucination rate that has nothing to do with the actual subject matter.

# CONCLUSION: 
If we fit classifiers on domain sub-groups, we are fitting on noise. A result like "History has AUC=0.75 but Food has AUC=0.52" may say more about the noise structure of the domain labels than about genuine domain difficulty. This is why the project runs all inferential tests at the benchmark level and combined level

# PROBLEMS WITH THE REFUSED LABELS
frac_refused has been excluded from training => this is because refusals are part of the label, and are not a feature.  This would imply data leakage.