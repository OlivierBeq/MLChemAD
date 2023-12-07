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
app_domain.fit(data.training)

# Determine outliers from multiple samples (rows) ...
print(app_domain.contains(data.test))

# ... or a unique sample
print(app_domain.contains(data.test[5]))
```

Depending on the definition of the applicability domain, some samples of the training set might be outliers themselves.

# Applicability domains
The applicability domain defined by MLChemAD as the following:
- Bounding Box
- PCA Bounding Box
- Convex Hull ***(does not scale well)***
- TOPKAT's Optimum Prediction Space ***(recommended)***
- Leverage
- Hotelling T²
- Distance to Centroids
- k-Nearest Neighbors
- Isolation Forests
- Non-parametric Kernel Densities
