# MindForge v1.0.6-alpha Release Notes

**Release Date:** December 1, 2025  
**Version:** 1.0.6-alpha  
**Status:** Alpha Release

---

## Overview

MindForge v1.0.6-alpha introduces two major algorithm families: **Naive Bayes classifiers** (with three variants) and **Support Vector Machines (SVM)**. This release adds 4 new classifiers, bringing the total to 9 different classification algorithms available in the library.

---

## New Features

### Naive Bayes Classifiers

A complete family of probabilistic classifiers based on Bayes' theorem with strong independence assumptions.

#### 1. Gaussian Naive Bayes (`GaussianNaiveBayes`)

Ideal for continuous features that follow a normal distribution.

**Features:**
- Automatic mean and variance calculation per class
- Probability estimation using Gaussian distribution
- Prior probability support (automatic or custom)
- Numerical stability with variance smoothing
- Full probability predictions (`predictProba`)

**Usage Example:**
```java
import com.mindforge.classifier.GaussianNaiveBayes;

// Create and train classifier
GaussianNaiveBayes gnb = new GaussianNaiveBayes();
gnb.train(features, labels);

// Predict
int prediction = gnb.predict(newSample);

// Get probability distribution
double[] probabilities = gnb.predictProba(newSample);
```

**Mathematical Foundation:**
```
P(x|c) = (1 / √(2πσ²)) × exp(-(x-μ)² / (2σ²))
```

#### 2. Multinomial Naive Bayes (`MultinomialNaiveBayes`)

Optimized for discrete count data, particularly effective for text classification.

**Features:**
- Laplace smoothing (configurable alpha parameter)
- Efficient handling of word frequency vectors
- Ideal for document classification and NLP tasks
- Log-probability calculations for numerical stability

**Usage Example:**
```java
import com.mindforge.classifier.MultinomialNaiveBayes;

// Create with custom smoothing
MultinomialNaiveBayes mnb = new MultinomialNaiveBayes(1.0); // alpha = 1.0

// Train on word count vectors
mnb.train(wordCounts, documentLabels);

// Classify new document
int category = mnb.predict(newDocumentVector);
```

**Use Cases:**
- Spam detection
- Sentiment analysis
- Topic classification
- Document categorization

#### 3. Bernoulli Naive Bayes (`BernoulliNaiveBayes`)

Designed for binary/boolean feature vectors.

**Features:**
- Automatic binarization with configurable threshold
- Handles both presence and absence of features
- Laplace smoothing support
- Efficient for binary feature spaces

**Usage Example:**
```java
import com.mindforge.classifier.BernoulliNaiveBayes;

// Create with binarization threshold
BernoulliNaiveBayes bnb = new BernoulliNaiveBayes(0.5, 1.0);

// Train on binary features
bnb.train(binaryFeatures, labels);

// Predict
int prediction = bnb.predict(newBinarySample);
```

**Use Cases:**
- Binary text classification (word presence/absence)
- Feature selection validation
- Boolean attribute classification

---

### Support Vector Machine (`SVC`)

A powerful maximum-margin classifier using the Sequential Minimal Optimization (SMO) algorithm.

**Features:**
- Linear SVM with SMO optimization
- One-vs-Rest strategy for multiclass classification
- Configurable regularization (C parameter)
- Convergence tolerance control
- Reproducible results with random state
- Builder pattern for flexible configuration

**Usage Example:**
```java
import com.mindforge.classifier.SVC;

// Create SVM with builder pattern
SVC svm = new SVC.Builder()
    .C(1.0)              // Regularization parameter
    .maxIter(1000)       // Maximum iterations
    .tol(1e-4)           // Convergence tolerance
    .randomState(42)     // For reproducibility
    .build();

// Train
svm.fit(features, labels);

// Predict
int[] predictions = svm.predict(testFeatures);

// Access model parameters
double[] weights = svm.getWeights();
double bias = svm.getBias();
```

**Algorithm Details:**
- Uses Sequential Minimal Optimization (SMO) for efficient training
- Implements soft-margin SVM with regularization
- Automatic multiclass extension via One-vs-Rest decomposition

**Hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| C | 1.0 | Regularization parameter (higher = less regularization) |
| maxIter | 1000 | Maximum optimization iterations |
| tol | 1e-4 | Convergence tolerance |
| randomState | null | Random seed for reproducibility |

---

## Complete Algorithm Reference

MindForge v1.0.6-alpha now includes **9 classifiers**:

| Algorithm | Class | Best For |
|-----------|-------|----------|
| K-Nearest Neighbors | `KNNClassifier` | Instance-based learning |
| Decision Tree | `DecisionTreeClassifier` | Interpretable models |
| Random Forest | `RandomForestClassifier` | Ensemble learning |
| Perceptron | `Perceptron` | Linear binary classification |
| MLP | `MultiLayerPerceptron` | Complex non-linear patterns |
| Logistic Regression | `LogisticRegression` | Probabilistic classification |
| **Gaussian Naive Bayes** | `GaussianNaiveBayes` | Continuous features |
| **Multinomial Naive Bayes** | `MultinomialNaiveBayes` | Text/count data |
| **Bernoulli Naive Bayes** | `BernoulliNaiveBayes` | Binary features |
| **SVM** | `SVC` | Maximum margin classification |

---

## Testing

This release includes comprehensive test coverage:

- **New Tests Added:** 77 tests
  - Gaussian Naive Bayes: 23 tests
  - Multinomial Naive Bayes: 19 tests
  - Bernoulli Naive Bayes: 20 tests
  - SVC: 15 tests
- **Total Project Tests:** 252 tests
- **All Tests Passing:** ✅

### Test Categories:
- Binary classification
- Multiclass classification
- Edge cases (single class, empty data)
- Numerical stability
- Probability predictions
- Model persistence
- Cross-validation integration

---

## Performance Characteristics

### Naive Bayes Family

| Variant | Training | Prediction | Memory |
|---------|----------|------------|--------|
| Gaussian | O(n×d) | O(c×d) | O(c×d) |
| Multinomial | O(n×d) | O(c×d) | O(c×d) |
| Bernoulli | O(n×d) | O(c×d) | O(c×d) |

Where: n = samples, d = features, c = classes

**Advantages:**
- Very fast training and prediction
- Works well with high-dimensional data
- Handles missing features gracefully
- Excellent for text classification

### SVM (SVC)

| Operation | Complexity |
|-----------|------------|
| Training | O(n²) to O(n³) |
| Prediction | O(d) per sample |

**Advantages:**
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors)
- Versatile with kernel trick (future enhancement)
- Robust to overfitting

---

## Integration with Cross-Validation

All new classifiers integrate seamlessly with the Cross-Validation utilities from v1.0.5:

```java
import com.mindforge.utils.CrossValidation;
import com.mindforge.classifier.GaussianNaiveBayes;

// K-Fold Cross-Validation with Gaussian Naive Bayes
CrossValidationResult result = CrossValidation.kFold(
    features, labels, 5,
    (X, y) -> {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(X, y);
        return gnb;
    },
    (model, X) -> {
        int[] preds = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            preds[i] = ((GaussianNaiveBayes) model).predict(X[i]);
        }
        return preds;
    }
);

System.out.println("Mean Accuracy: " + result.getMean());
System.out.println("Std Dev: " + result.getStdDev());
```

---

## Migration Guide

### From v1.0.5-alpha

No breaking changes. Simply update your dependency:

```xml
<dependency>
    <groupId>com.mindforge</groupId>
    <artifactId>mindforge</artifactId>
    <version>1.0.6-alpha</version>
</dependency>
```

---

## Known Limitations

1. **SVM Kernel Support:** Currently only linear kernel is implemented. RBF and polynomial kernels planned for future releases.

2. **Large Dataset Performance:** SMO algorithm may be slow for very large datasets (>10,000 samples). Consider using other classifiers or sampling.

3. **Naive Bayes Independence:** Assumes feature independence, which may not hold for correlated features.

---

## What's Next (Roadmap)

- [ ] **Gradient Boosting** - Ensemble method with sequential weak learners
- [ ] SVM Kernel Extensions (RBF, Polynomial)
- [ ] Feature Selection Utilities
- [ ] Model Serialization/Persistence
- [ ] GPU Acceleration

---

## Contributors

- MindForge Development Team

---

## Links

- **GitHub Repository:** https://github.com/yasmramos/MindForge
- **Previous Release:** [v1.0.5-alpha](https://github.com/yasmramos/MindForge/releases/tag/v1.0.5-alpha)
- **Issue Tracker:** https://github.com/yasmramos/MindForge/issues

---

**Thank you for using MindForge!**

*Building intelligent systems, one algorithm at a time.*
