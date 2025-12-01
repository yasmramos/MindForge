## 🚀 MindForge v1.0.1-alpha - Data Preprocessing Package

### ✨ New Features

This version introduces a **complete data preprocessing package** (`com.mindforge.preprocessing`) that provides essential tools to prepare data before training machine learning models.

#### 📊 Core Components

**1. MinMaxScaler** - Feature Normalization
- Scales features to the range [0, 1] or a custom range
- Preserves relationships between values
- Methods: `fit()`, `transform()`, `fitTransform()`, `inverseTransform()`

**2. StandardScaler** - Standardization
- Transforms data to mean = 0 and standard deviation = 1
- Ideal for scale-sensitive algorithms (KNN, SVM, neural networks)
- Methods: `fit()`, `transform()`, `fitTransform()`, `inverseTransform()`

**3. SimpleImputer** - Missing Value Handling
- Available strategies:
  - `MEAN`: Mean imputation
  - `MEDIAN`: Median imputation
  - `MOST_FREQUENT`: Mode imputation
  - `CONSTANT`: Constant value imputation
- Supports `NaN` and `null` values

**4. LabelEncoder** - Categorical Label Encoding
- Converts text labels to integers
- Bidirectional: `encode()` and `decode()`
- Maintains consistent mapping for inverse transformations

**5. DataSplit** - Dataset Splitting
- `trainTestSplit()`: Random train/test split
- `stratifiedSplit()`: Stratified split preserving class distribution
- Supports both integer and continuous labels
- Random control with `randomState`

### 📈 Test Coverage

- **73 tests** passing successfully
- Complete coverage of all preprocessing components
- Edge case and error handling tests

### 📖 Documentation

- Updated README with usage examples
- Inline code documentation
- Detailed guide in `PREPROCESSING_README.md`

### 🔧 Quick Start

```java
import com.mindforge.preprocessing.*;

// Normalization
MinMaxScaler scaler = new MinMaxScaler();
scaler.fit(data);
double[][] normalized = scaler.transform(data);

// Standardization
StandardScaler standardScaler = new StandardScaler();
double[][] standardized = standardScaler.fitTransform(data);

// Imputation
SimpleImputer imputer = new SimpleImputer(SimpleImputer.Strategy.MEAN);
double[][] cleaned = imputer.fitTransform(dataWithNaN);

// Encoding
LabelEncoder encoder = new LabelEncoder();
int[] encoded = encoder.encode(labels);

// Data splitting
DataSplit.Split split = DataSplit.trainTestSplit(X, y, 0.25, 42);
```

### 📦 Installation

```xml
<dependency>
    <groupId>com.mindforge</groupId>
    <artifactId>mindforge</artifactId>
    <version>1.0.1-alpha</version>
</dependency>
```

### 🔗 Resources

- [Main README](README.md)
- [Preprocessing Guide](PREPROCESSING_README.md)
- [Repository](https://github.com/yasmramos/MindForge)

---

**Author**: MindForge Team  
**Version**: 1.0.1-alpha  
**Date**: December 2025
