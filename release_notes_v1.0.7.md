# MindForge v1.0.7-alpha Release Notes

**Release Date:** December 1, 2025  
**Version:** 1.0.7-alpha  
**Status:** Alpha Release

---

## Overview

MindForge v1.0.7-alpha completes the core classification roadmap with the addition of **Gradient Boosting**, a powerful ensemble learning method. This release marks a significant milestone with **10 classification algorithms** now available in the library.

---

## New Features

### Gradient Boosting Classifier (`GradientBoostingClassifier`)

A sequential ensemble method that builds weak learners (decision trees) iteratively, where each new tree corrects errors made by previous trees.

**Features:**
- Configurable number of boosting stages (estimators)
- Learning rate control for regularization
- Maximum depth control for base learners
- Subsampling support for stochastic gradient boosting
- Multi-class classification via One-vs-Rest
- Probability predictions with softmax normalization
- Builder pattern for flexible configuration
- Reproducibility with random state

**Usage Example:**
```java
import com.mindforge.classification.GradientBoostingClassifier;

// Create and configure Gradient Boosting classifier
GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
    .nEstimators(100)           // Number of boosting stages
    .learningRate(0.1)          // Shrinkage parameter
    .maxDepth(3)                // Maximum depth of base trees
    .subsample(1.0)             // Fraction of samples for fitting
    .randomState(42)            // For reproducibility
    .build();

// Train
gb.fit(features, labels);

// Predict
int[] predictions = gb.predict(testFeatures);

// Get probability predictions
double[][] probabilities = gb.predictProba(testFeatures);

// Single sample prediction
int prediction = gb.predict(singleSample);
double[] proba = gb.predictProba(singleSample);
```

**Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| nEstimators | 100 | Number of boosting stages (trees) |
| learningRate | 0.1 | Shrinks the contribution of each tree (0, 1] |
| maxDepth | 3 | Maximum depth of individual trees |
| subsample | 1.0 | Fraction of samples for fitting (stochastic boosting) |
| randomState | null | Random seed for reproducibility |

**Algorithm Details:**
- Uses gradient descent to minimize log loss (cross-entropy)
- Builds decision trees to fit pseudo-residuals
- Applies shrinkage (learning rate) to prevent overfitting
- Supports both binary and multi-class classification
- One-vs-Rest strategy for multi-class problems

---

## Complete Algorithm Reference

MindForge v1.0.7-alpha now includes **10 classifiers**:

| Algorithm | Class | Best For |
|-----------|-------|----------|
| K-Nearest Neighbors | `KNearestNeighbors` | Instance-based learning |
| Decision Tree | `DecisionTreeClassifier` | Interpretable models |
| Random Forest | `RandomForestClassifier` | Ensemble learning (bagging) |
| Perceptron | `Perceptron` | Linear binary classification |
| MLP | `MultiLayerPerceptron` | Complex non-linear patterns |
| Logistic Regression | `LogisticRegression` | Probabilistic classification |
| Gaussian Naive Bayes | `GaussianNaiveBayes` | Continuous features |
| Multinomial Naive Bayes | `MultinomialNaiveBayes` | Text/count data |
| Bernoulli Naive Bayes | `BernoulliNaiveBayes` | Binary features |
| SVM | `SVC` | Maximum margin classification |
| **Gradient Boosting** | `GradientBoostingClassifier` | Sequential ensemble learning |

---

## Testing

This release includes comprehensive test coverage:

- **New Tests Added:** 33 tests for Gradient Boosting
- **Total Project Tests:** 285 tests
- **All Tests Passing:** ✅

### Test Categories for Gradient Boosting:
- Constructor validation (default, custom, builder)
- Parameter validation (edge cases, invalid values)
- Binary classification
- Multi-class classification (3+ classes)
- Probability predictions
- Subsampling (stochastic gradient boosting)
- Reproducibility tests
- Edge cases (min estimators, max learning rate, etc.)

---

## Performance Characteristics

### Gradient Boosting

| Operation | Complexity |
|-----------|------------|
| Training | O(n × m × d × log(n)) |
| Prediction | O(m × d) per sample |

Where: n = samples, m = estimators, d = tree depth

**Advantages:**
- High predictive accuracy
- Handles mixed feature types
- Built-in feature importance
- Robust to outliers
- Captures complex non-linear relationships

**Considerations:**
- Sequential training (not parallelizable)
- Prone to overfitting without regularization
- Slower training than Random Forest

---

## Comparison: Random Forest vs Gradient Boosting

| Aspect | Random Forest | Gradient Boosting |
|--------|--------------|-------------------|
| Training | Parallel (bagging) | Sequential (boosting) |
| Trees | Independent | Dependent |
| Overfitting | Less prone | More prone without tuning |
| Speed | Faster training | Slower training |
| Accuracy | Good | Often better with tuning |
| Interpretability | Moderate | Moderate |

---

## Integration Example

```java
import com.mindforge.classification.GradientBoostingClassifier;
import com.mindforge.validation.CrossValidation;
import com.mindforge.validation.CrossValidationResult;

// Cross-validate Gradient Boosting
CrossValidationResult result = CrossValidation.kFold(
    features, labels, 5,
    (X, y) -> {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(50)
            .learningRate(0.1)
            .maxDepth(3)
            .build();
        gb.fit(X, y);
        return gb;
    },
    (model, X) -> ((GradientBoostingClassifier) model).predict(X)
);

System.out.println("CV Mean Accuracy: " + result.getMean());
System.out.println("CV Std Dev: " + result.getStdDev());
```

---

## Migration Guide

### From v1.0.6-alpha

No breaking changes. Simply update your dependency:

```xml
<dependency>
    <groupId>com.mindforge</groupId>
    <artifactId>mindforge</artifactId>
    <version>1.0.7-alpha</version>
</dependency>
```

---

## Roadmap Status

### Completed ✅
- [x] K-Nearest Neighbors
- [x] Decision Trees
- [x] Random Forest
- [x] Logistic Regression
- [x] Cross-Validation (K-Fold, Stratified, LOOCV, Shuffle Split)
- [x] Naive Bayes (Gaussian, Multinomial, Bernoulli)
- [x] Support Vector Machines (Linear)
- [x] **Gradient Boosting** ⭐ NEW

### Upcoming
- [ ] SVM Kernel Extensions (RBF, Polynomial)
- [ ] Feature Selection Utilities
- [ ] Model Serialization/Persistence
- [ ] XGBoost-style optimizations
- [ ] GPU Acceleration

---

## Summary

MindForge v1.0.7-alpha represents a complete implementation of the core classification roadmap:

| Version | Features Added |
|---------|----------------|
| v1.0.4-alpha | Logistic Regression |
| v1.0.5-alpha | Cross-Validation utilities |
| v1.0.6-alpha | Naive Bayes (3 variants) + SVM |
| v1.0.7-alpha | Gradient Boosting |

**Total Statistics:**
- **10 Classification Algorithms**
- **285 Unit Tests**
- **100% Core Roadmap Complete**

---

## Contributors

- MindForge Development Team

---

## Links

- **GitHub Repository:** https://github.com/yasmramos/MindForge
- **Previous Release:** [v1.0.6-alpha](https://github.com/yasmramos/MindForge/releases/tag/v1.0.6-alpha)
- **Issue Tracker:** https://github.com/yasmramos/MindForge/issues

---

**Thank you for using MindForge!**

*Building intelligent systems, one algorithm at a time.*
