# MLChemAD
Applicability domain definitions for cheminformatics modelling.

# Getting Started

## Install
```
pip install mlchemad
```

## Example Usage

- With molecular fingerprints, prefer the use of the `KNNApplicabilityDomain` with `k=1`, `scaling=None`, `hard_threshold=0.3`, and `dist='jaccard'`.
- Otherwise, the use of the `TopKatApplicabilityDomain` is recommended.

```python
from mlchemad import TopKatApplicabilityDomain, KNNApplicabilityDomain, data

# Create the applicability domain using TopKat's definition
app_domain = TopKatApplicabilityDomain()
# Fit it to the training set
app_domain.fit(data.mekenyan1993.training)

# Determine outliers from multiple samples (rows) ...
print(app_domain.contains(data.mekenyan1993.test))

# ... or a unique sample
sample = data.mekenyan1993.test.iloc[5] # Obtain the 5th row as a pandas.Series object 
print(app_domain.contains(sample))

# Now with Morgan fingerprints
app_domain = KNNApplicabilityDomain(k=1, scaling=None, hard_threshold=0.3, dist='jaccard')
app_domain.fit(data.broccatelli2011.training.drop(columns='Activity'))
print(app_domain.contains(data.broccatelli2011.test.drop(columns='Activity')))
```

Depending on the definition of the applicability domain, some samples of the training set might be outliers themselves.

# Applicability domains
The applicability domain defined by MLChemAD as the following:
- Bounding Box
- PCA Bounding Box
- Convex Hull<br/>
  ***(does not scale well)***
- TOPKAT's Optimum Prediction Space<br/>
  ***(recommended with molecular descriptors)***
- Leverage
- Hotelling T²
- Distance to Centroids
- k-Nearest Neighbors<br/>
  ***(recommended with molecular fingerprints with the use of `dist='rogerstanimoto'`, `scaling=None` and `hard_threshold=0.75` for ECFP fingerprints)***
- Isolation Forests
- Non-parametric Kernel Densities
