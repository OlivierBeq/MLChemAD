# MLChemAD
Applicability domain definitions for cheminformatics modelling.

# Getting Started

## Install
```
pip install mlchemad
```

## Example Usage

```python
from mlchemad import TopKatApplicabilityDomain, data

# Create the applicability domain
app_domain = TopKatApplicabilityDomain()
# Fit it to the training set
app_domain.fit(data.mekenyan1993.training)

# Determine outliers from multiple samples (rows) ...
print(app_domain.contains(data.mekenyan1993.test))

# ... or a unique sample
sample = data.test[5] # Obtain the 5th row as a pandas.Series object 
print(app_domain.contains(sample))
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
