# MindForge v1.0.8-alpha Release Notes

**Release Date**: December 1, 2025

## Overview

MindForge v1.0.8-alpha introduces powerful Feature Selection capabilities, Principal Component Analysis (PCA) for dimensionality reduction, and Model Persistence for saving and loading trained models. This release adds 152 new tests and significantly expands the library's preprocessing and utility features.

## New Features

### Feature Selection

Three new feature selection methods to identify the most relevant features for your models:

#### VarianceThreshold
Removes features with variance below a specified threshold. Useful for removing constant or near-constant features.

```java
VarianceThreshold selector = new VarianceThreshold(0.1);
double[][] X_selected = selector.fitTransform(X);

// Get feature information
double[] variances = selector.getVariances();
int[] selectedIndices = selector.getSelectedFeatureIndices();
boolean[] support = selector.getSupport();
```

**Key Features:**
- Remove zero-variance (constant) features with default threshold
- Custom variance threshold for fine-grained control
- Unsupervised - doesn't require target variable
- Full transparency with variance and selection information

#### SelectKBest
Selects the k best features based on statistical tests between features and target.

```java
// ANOVA F-value for classification
SelectKBest selector = new SelectKBest(ScoreFunction.F_CLASSIF, 5);
selector.fit(X, y);
double[][] X_best = selector.transform(X);

// Get scores and p-values
double[] scores = selector.getScores();
double[] pValues = selector.getPValues();
```

**Scoring Functions:**
- **F_CLASSIF**: ANOVA F-value for classification tasks
- **CHI2**: Chi-squared statistic (requires non-negative features)
- **MUTUAL_INFO**: Mutual information for discrete targets

#### RFE (Recursive Feature Elimination)
Recursively eliminates features based on their importance, keeping the most relevant ones.

```java
RFE rfe = new RFE(5);        // Keep 5 features
RFE rfe = new RFE(5, 2);     // Keep 5, remove 2 per iteration
rfe.fit(X, y);

// Get rankings and importances
int[] ranking = rfe.getRanking();           // 1 = selected
double[] importances = rfe.getFeatureImportances();
```

**Key Features:**
- Configurable number of features to eliminate per step
- Feature ranking showing elimination order
- Feature importance scores
- Efficient recursive algorithm

### PCA (Principal Component Analysis)

Dimensionality reduction using Singular Value Decomposition (SVD).

```java
PCA pca = new PCA(3);                        // Keep 3 components
double[][] X_reduced = pca.fitTransform(X);

// Explained variance analysis
double[] ratios = pca.getExplainedVarianceRatio();
double[] cumulative = pca.getCumulativeExplainedVarianceRatio();

// Reconstruct original data
double[][] X_reconstructed = pca.inverseTransform(X_reduced);

// Access components
double[][] components = pca.getComponents();
double[] singularValues = pca.getSingularValues();
```

**Key Features:**
- Power iteration SVD implementation
- Explained variance and cumulative variance ratios
- Inverse transform for data reconstruction
- Orthonormal principal components
- Feature mean centering

### Model Persistence

Save and load trained models to disk or byte arrays.

```java
// Save to file
ModelPersistence.save(model, "model.bin");

// Load from file
MyModel loaded = ModelPersistence.load("model.bin");

// Type-safe loading
MyModel typed = ModelPersistence.load("model.bin", MyModel.class);

// Get metadata without full loading
ModelMetadata meta = ModelPersistence.getMetadata("model.bin");
System.out.println(meta.getSimpleClassName());
System.out.println(meta.getFileSize());

// Validate file format
boolean valid = ModelPersistence.isValidModelFile("model.bin");

// Byte array serialization (for network/database)
byte[] bytes = ModelPersistence.toBytes(model);
MyModel fromBytes = ModelPersistence.fromBytes(bytes);
```

**Key Features:**
- File-based persistence with magic header validation
- Byte array serialization for network transfer
- Model metadata inspection without full deserialization
- Type-safe loading with class verification
- Custom ModelPersistenceException for error handling

## Statistics

### Test Coverage
| Category | Tests |
|----------|-------|
| Feature Selection (VarianceThreshold) | 24 |
| Feature Selection (SelectKBest) | 27 |
| Feature Selection (RFE) | 33 |
| PCA | 33 |
| Model Persistence | 35 |
| **New Tests in v1.0.8** | **152** |
| **Total Tests** | **437** |

### File Changes
| Type | Count |
|------|-------|
| New Java Source Files | 6 |
| New Test Files | 4 |
| Modified Files | 2 |

### New Source Files
- `com.mindforge.feature.VarianceThreshold` (252 lines)
- `com.mindforge.feature.SelectKBest` (505 lines)
- `com.mindforge.feature.RFE` (370 lines)
- `com.mindforge.decomposition.PCA` (475 lines)
- `com.mindforge.persistence.ModelPersistence` (286 lines)
- `com.mindforge.persistence.ModelPersistenceException` (35 lines)

## Package Structure

```
com.mindforge.feature/
├── VarianceThreshold.java    # Variance-based selection
├── SelectKBest.java          # Statistical test selection
└── RFE.java                  # Recursive feature elimination

com.mindforge.decomposition/
└── PCA.java                  # Principal Component Analysis

com.mindforge.persistence/
├── ModelPersistence.java     # Save/load utility
└── ModelPersistenceException.java
```

## API Reference

### VarianceThreshold
```java
VarianceThreshold()                          // Default threshold 0.0
VarianceThreshold(double threshold)          // Custom threshold
VarianceThreshold fit(double[][] X)          // Fit to data
double[][] transform(double[][] X)           // Transform data
double[][] fitTransform(double[][] X)        // Fit and transform
double[] getVariances()                      // Get computed variances
int[] getSelectedFeatureIndices()            // Get selected indices
boolean[] getSupport()                       // Get selection mask
int getNumberOfSelectedFeatures()            // Count selected
```

### SelectKBest
```java
SelectKBest(int k)                           // With F_CLASSIF
SelectKBest(ScoreFunction func, int k)       // Custom scoring
SelectKBest fit(double[][] X, int[] y)       // Fit to data
double[][] transform(double[][] X)           // Transform data
double[][] fitTransform(double[][] X, int[] y)
double[] getScores()                         // Feature scores
double[] getPValues()                        // P-values
int[] getSelectedFeatureIndices()            // Selected indices
```

### RFE
```java
RFE(int nFeaturesToSelect)                   // Default step=1
RFE(int nFeaturesToSelect, int step)         // Custom step
RFE fit(double[][] X, int[] y)               // Fit to data
double[][] transform(double[][] X)           // Transform data
int[] getRanking()                           // Feature rankings
double[] getFeatureImportances()             // Importance scores
boolean[] getSupport()                       // Selection mask
```

### PCA
```java
PCA()                                        // Keep all components
PCA(int nComponents)                         // Specify components
PCA fit(double[][] X)                        // Fit to data
double[][] transform(double[][] X)           // Project to PC space
double[][] fitTransform(double[][] X)        // Fit and transform
double[][] inverseTransform(double[][] X)    // Reconstruct
double[][] getComponents()                   // Principal components
double[] getExplainedVariance()              // Variance per PC
double[] getExplainedVarianceRatio()         // Ratio per PC
double[] getCumulativeExplainedVarianceRatio()
double[] getSingularValues()                 // SVD singular values
```

### ModelPersistence
```java
static void save(Serializable model, String path)
static <T> T load(String path)
static <T> T load(String path, Class<T> type)
static ModelMetadata getMetadata(String path)
static boolean isValidModelFile(String path)
static byte[] toBytes(Serializable model)
static <T> T fromBytes(byte[] bytes)
```

## Breaking Changes

None. This release is fully backward compatible with v1.0.7-alpha.

## Dependencies

No new dependencies added. Same as v1.0.7-alpha:
- Apache Commons Math 3.6.1
- ND4J 1.0.0-M2.1
- JUnit 5.10.1
- SLF4J 2.0.9

## Upgrade Guide

1. Update your dependency to v1.0.8-alpha
2. No code changes required for existing functionality
3. Start using new features:
   - Import `com.mindforge.feature.*` for feature selection
   - Import `com.mindforge.decomposition.PCA` for dimensionality reduction
   - Import `com.mindforge.persistence.ModelPersistence` for model save/load

## Known Issues

- PCA uses power iteration SVD which may be slower than optimized LAPACK implementations for very large matrices
- Model persistence requires models to implement `java.io.Serializable`

## Coming Next (v1.0.9)

Planned features:
- AdaBoost classifier
- SVM kernel functions (RBF, Polynomial)
- Polynomial feature transformation
- One-Hot Encoder

---

**Full Changelog**: https://github.com/yasmramos/MindForge/compare/v1.0.7-alpha...v1.0.8-alpha

**Download**: [mindforge-1.0.8-alpha.jar](https://github.com/yasmramos/MindForge/releases/download/v1.0.8-alpha/mindforge-1.0.8-alpha.jar)
