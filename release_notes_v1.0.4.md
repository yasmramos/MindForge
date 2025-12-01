# MindForge v1.0.4-alpha Release Notes

**Release Date**: December 1, 2025  
**Version**: 1.0.4-alpha

## 🎉 New Features

### Logistic Regression Classifier
We're excited to introduce **Logistic Regression**, a powerful classification algorithm for binary and multiclass problems.

#### Key Features:
- **Multiple Solvers**:
  - Gradient Descent (GD): Full batch gradient descent
  - Stochastic Gradient Descent (SGD): Mini-batch online learning
  - Newton-CG: Newton-Conjugate Gradient for faster convergence
  
- **Regularization Options**:
  - L1 (Lasso): Promotes sparsity in coefficients
  - L2 (Ridge): Prevents overfitting with weight decay
  - Elastic Net: Combination of L1 and L2
  - No regularization: Standard logistic regression

- **Multiclass Classification**:
  - One-vs-Rest (OvR) strategy for handling multiple classes
  - Probability predictions for all classes
  
- **Training Features**:
  - Configurable learning rates
  - Early stopping with convergence tolerance
  - Loss history tracking
  - Random state for reproducibility
  - Batch size configuration for SGD

#### Usage Example

```java
import io.mindforge.classification.LogisticRegression;

// Training data
double[][] X_train = {{1.0, 2.0}, {2.0, 3.0}, {8.0, 8.0}, {9.0, 10.0}};
int[] y_train = {0, 0, 1, 1};

// Create Logistic Regression model with L2 regularization
LogisticRegression lr = new LogisticRegression.Builder()
    .penalty("l2")                 // Regularization type
    .C(1.0)                        // Regularization strength
    .solver("gradient_descent")    // Optimization solver
    .learningRate(0.1)             // Learning rate
    .maxIter(1000)                 // Maximum iterations
    .tol(1e-4)                     // Convergence tolerance
    .randomState(42)               // For reproducibility
    .build();

// Train the model
lr.fit(X_train, y_train);

// Make predictions
double[][] X_test = {{5.0, 5.0}, {2.0, 2.5}};
int[] predictions = lr.predict(X_test);

// Get probability predictions
double[][] probabilities = lr.predictProba(X_test);

// Access model parameters
double[][] coefficients = lr.getCoefficients();
double[] intercepts = lr.getIntercepts();
int[] classes = lr.getClasses();

// View training progress
List<Double> lossHistory = lr.getLossHistory();
```

## 📊 API Reference

### LogisticRegression.Builder

Configuration options for building a Logistic Regression model:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `penalty` | String | `"l2"` | Regularization type: `"l1"`, `"l2"`, `"elasticnet"`, `"none"` |
| `C` | double | `1.0` | Inverse of regularization strength (smaller = stronger regularization) |
| `l1Ratio` | double | `0.5` | Elastic Net mixing parameter (0 = L2, 1 = L1) |
| `solver` | String | `"gradient_descent"` | Optimization solver: `"gradient_descent"`, `"sgd"`, `"newton_cg"` |
| `maxIter` | int | `1000` | Maximum number of iterations |
| `tol` | double | `1e-4` | Convergence tolerance |
| `learningRate` | double | `0.01` | Learning rate for gradient-based solvers |
| `batchSize` | int | `32` | Batch size for SGD solver |
| `fitIntercept` | boolean | `true` | Whether to fit an intercept term |
| `randomState` | int | `42` | Random seed for reproducibility |
| `verbose` | int | `0` | Verbosity level (0=silent, 1=convergence, 2=detailed) |

### LogisticRegression Methods

| Method | Return Type | Description |
|--------|-------------|-------------|
| `fit(double[][] X, int[] y)` | void | Train the model on data |
| `predict(double[][] X)` | int[] | Predict class labels |
| `predictProba(double[][] X)` | double[][] | Predict class probabilities |
| `getCoefficients()` | double[][] | Get model coefficients/weights |
| `getIntercepts()` | double[] | Get model intercepts |
| `getClasses()` | int[] | Get unique class labels |
| `getLossHistory()` | List&lt;Double&gt; | Get loss values during training |

## 🧪 Test Coverage

- **32 comprehensive tests** for Logistic Regression:
  - Binary and multiclass classification
  - All three solvers (GD, SGD, Newton-CG)
  - All regularization types (L1, L2, Elastic Net, None)
  - Probability predictions
  - Convergence behavior
  - Model attributes and parameters
  - Edge cases and error handling
  - Reproducibility
  - Different hyperparameter configurations
  
- **Total project tests**: 154 (all passing)

## 📈 Performance Characteristics

### Solver Comparison

- **Gradient Descent**: Most stable, best for small-medium datasets
- **SGD**: Fastest for large datasets, supports online learning
- **Newton-CG**: Fastest convergence when applicable, best for smaller feature spaces

### Regularization Guidelines

- **L2 (Ridge)**: Default choice, works well in most cases
- **L1 (Lasso)**: Use for feature selection, produces sparse models
- **Elastic Net**: Combines benefits of L1 and L2
- **None**: Use only with sufficient data and no multicollinearity

## 🔄 Improvements & Updates

- Updated README.md with Logistic Regression documentation
- Added comprehensive usage examples
- Marked Logistic Regression as completed in roadmap
- Updated project version to 1.0.4-alpha

## 📦 Installation

### Maven Dependency

```xml
<dependency>
    <groupId>com.mindforge</groupId>
    <artifactId>mindforge</artifactId>
    <version>1.0.4-alpha</version>
</dependency>
```

### Download JAR

Download the compiled JAR from the [releases page](https://github.com/yasmramos/MindForge/releases/tag/v1.0.4-alpha):
- `mindforge-1.0.4-alpha.jar`

## 🔗 Links

- **Repository**: https://github.com/yasmramos/MindForge
- **Issues**: https://github.com/yasmramos/MindForge/issues
- **Previous Release**: [v1.0.3-alpha](https://github.com/yasmramos/MindForge/releases/tag/v1.0.3-alpha)

## 🎯 What's Next?

Coming in future releases:
- Naive Bayes classifiers
- Support Vector Machines (SVM)
- Cross-validation utilities
- Feature selection methods
- Gradient Boosting

## 🙏 Acknowledgments

Special thanks to the scikit-learn and Apache Commons Math projects for inspiration on API design and implementation patterns.

---

**Full Changelog**: https://github.com/yasmramos/MindForge/compare/v1.0.3-alpha...v1.0.4-alpha
