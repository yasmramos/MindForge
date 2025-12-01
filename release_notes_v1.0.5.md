# MindForge v1.0.5-alpha Release Notes

**Release Date**: December 1, 2025  
**Version**: 1.0.5-alpha

## 🎯 Overview

This release introduces comprehensive **Cross-Validation** utilities to MindForge, providing essential tools for robust model evaluation and selection. Cross-validation is a fundamental technique in machine learning for assessing model performance and detecting overfitting.

## ✨ New Features

### Cross-Validation Framework

A complete cross-validation system with multiple strategies:

#### **K-Fold Cross-Validation**
- Splits data into K equal folds
- Trains on K-1 folds and tests on the remaining fold
- Repeats K times with each fold used as test set exactly once
- Supports optional shuffling with reproducible random state

#### **Stratified K-Fold Cross-Validation**
- Similar to K-Fold but maintains class proportions in each fold
- Ensures balanced class distribution across folds
- Particularly useful for imbalanced datasets
- Prevents biased evaluation due to class imbalance

#### **Leave-One-Out Cross-Validation (LOOCV)**
- Uses n-1 samples for training and 1 sample for testing
- Repeats n times, leaving out a different sample each time
- Provides nearly unbiased estimate of model performance
- Recommended for small datasets (computationally expensive for large datasets)

#### **Shuffle Split Cross-Validation**
- Randomly splits data into train and test sets multiple times
- Flexible test size configuration (0.0 to 1.0)
- Unlike K-Fold, samples may appear in multiple test sets or none
- Useful for repeated random subsampling validation

#### **Train-Test Split**
- Simple one-time split of data into train and test sets
- Configurable test size ratio
- Supports optional shuffling with reproducible random state
- Returns structured SplitData object with train and test arrays

## 🔧 Implementation Details

### New Classes

#### `CrossValidationResult`
**Package**: `com.mindforge.validation`

Stores and provides statistical analysis of cross-validation results:

```java
public class CrossValidationResult {
    // Get all fold scores
    public double[] getScores();
    
    // Statistical metrics
    public double getMean();      // Average score across folds
    public double getStdDev();    // Standard deviation
    public double getMin();       // Minimum score
    public double getMax();       // Maximum score
    
    // Metadata
    public String getMetricName();  // Name of evaluation metric
    public int getNumFolds();       // Number of folds/splits
}
```

#### `CrossValidation`
**Package**: `com.mindforge.validation`

Main class providing static methods for cross-validation:

```java
public class CrossValidation {
    // Functional interfaces for model training and prediction
    @FunctionalInterface
    public interface ModelTrainer<M> {
        M train(double[][] X, int[] y);
    }
    
    @FunctionalInterface
    public interface ModelPredictor<M> {
        int[] predict(M model, double[][] X);
    }
    
    // Cross-validation methods
    public static <M> CrossValidationResult kFold(
        ModelTrainer<M> trainer,
        ModelPredictor<M> predictor,
        double[][] X, int[] y, int k, Integer randomState
    );
    
    public static <M> CrossValidationResult stratifiedKFold(
        ModelTrainer<M> trainer,
        ModelPredictor<M> predictor,
        double[][] X, int[] y, int k, Integer randomState
    );
    
    public static <M> CrossValidationResult leaveOneOut(
        ModelTrainer<M> trainer,
        ModelPredictor<M> predictor,
        double[][] X, int[] y
    );
    
    public static <M> CrossValidationResult shuffleSplit(
        ModelTrainer<M> trainer,
        ModelPredictor<M> predictor,
        double[][] X, int[] y,
        int nSplits, double testSize, Integer randomState
    );
    
    public static SplitData trainTestSplit(
        double[][] X, int[] y,
        double testSize, Integer randomState
    );
    
    // Container for split data
    public static class SplitData {
        public final double[][] XTrain;
        public final int[] yTrain;
        public final double[][] XTest;
        public final int[] yTest;
    }
}
```

## 📚 Usage Examples

### K-Fold Cross-Validation

```java
import com.mindforge.classification.KNearestNeighbors;
import com.mindforge.validation.CrossValidation;
import com.mindforge.validation.CrossValidationResult;

// Training data
double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0}, {8.0, 8.0}, {9.0, 10.0}, {10.0, 11.0}};
int[] y = {0, 0, 0, 1, 1, 1};

// Define model trainer and predictor
CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X_train, y_train) -> {
    KNearestNeighbors knn = new KNearestNeighbors(3);
    knn.train(X_train, y_train);
    return knn;
};

CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
    (model, X_test) -> model.predict(X_test);

// Perform 5-fold cross-validation
CrossValidationResult result = CrossValidation.kFold(
    trainer, predictor, X, y, 5, 42  // k=5, random_state=42
);

System.out.println("Mean Accuracy: " + result.getMean());
System.out.println("Std Dev: " + result.getStdDev());
System.out.println("Min: " + result.getMin());
System.out.println("Max: " + result.getMax());
System.out.println("All Scores: " + Arrays.toString(result.getScores()));
```

### Stratified K-Fold for Imbalanced Data

```java
// Imbalanced dataset: 70% class 0, 30% class 1
double[][] X = /* ... */;
int[] y = /* ... */;

// Stratified K-Fold maintains class proportions
CrossValidationResult result = CrossValidation.stratifiedKFold(
    trainer, predictor, X, y, 5, 42
);

System.out.println("Stratified Mean Accuracy: " + result.getMean());
```

### Leave-One-Out Cross-Validation

```java
// Best for small datasets
double[][] X_small = {{1.0, 2.0}, {2.0, 3.0}, {8.0, 8.0}, {9.0, 10.0}};
int[] y_small = {0, 0, 1, 1};

CrossValidationResult result = CrossValidation.leaveOneOut(
    trainer, predictor, X_small, y_small
);

System.out.println("LOOCV Accuracy: " + result.getMean());
System.out.println("Number of folds: " + result.getNumFolds());  // equals dataset size
```

### Shuffle Split for Random Subsampling

```java
// Perform 10 random 80-20 splits
CrossValidationResult result = CrossValidation.shuffleSplit(
    trainer, predictor, X, y,
    10,     // number of splits
    0.2,    // test size (20%)
    42      // random state
);

System.out.println("Mean Accuracy: " + result.getMean());
System.out.println("Std Dev: " + result.getStdDev());
```

### Train-Test Split

```java
import com.mindforge.validation.Metrics;

// Simple 70-30 split
CrossValidation.SplitData split = CrossValidation.trainTestSplit(
    X, y, 0.3, 42  // 30% test, random_state=42
);

// Train model
KNearestNeighbors model = new KNearestNeighbors(3);
model.train(split.XTrain, split.yTrain);

// Evaluate on test set
int[] predictions = model.predict(split.XTest);
double accuracy = Metrics.accuracy(split.yTest, predictions);

System.out.println("Test Accuracy: " + accuracy);
System.out.println("Train size: " + split.XTrain.length);
System.out.println("Test size: " + split.XTest.length);
```

### Using with Logistic Regression

```java
import io.mindforge.classification.LogisticRegression;

// Define trainer for Logistic Regression
CrossValidation.ModelTrainer<LogisticRegression> lrTrainer = (X_train, y_train) -> {
    LogisticRegression lr = new LogisticRegression.Builder()
        .penalty("l2")
        .C(1.0)
        .solver("gradient_descent")
        .learningRate(0.1)
        .maxIter(1000)
        .randomState(42)
        .build();
    lr.fit(X_train, y_train);
    return lr;
};

CrossValidation.ModelPredictor<LogisticRegression> lrPredictor = 
    (model, X_test) -> model.predict(X_test);

// Perform cross-validation
CrossValidationResult result = CrossValidation.kFold(
    lrTrainer, lrPredictor, X, y, 5, 42
);

System.out.println("Logistic Regression CV Accuracy: " + result.getMean());
```

## 🏗️ Design Principles

### Generic Model Support
- Uses Java generics and functional interfaces
- Works with any model type (no specific interface required)
- Flexible trainer and predictor functions
- Compatible with both `com.mindforge` and `io.mindforge` packages

### Reproducibility
- Supports random state for reproducible results
- Consistent shuffling across multiple runs
- Essential for experiment tracking and debugging

### Statistical Rigor
- Provides multiple evaluation metrics (mean, std dev, min, max)
- Supports stratification for imbalanced datasets
- Implements standard cross-validation techniques from literature

### Performance
- Efficient index-based data splitting
- No unnecessary data copying
- Optimized for large datasets (except LOOCV)

## 🧪 Testing

Added comprehensive test suite with **21 test cases**:

### Test Coverage
- ✅ K-Fold basic functionality
- ✅ K-Fold with/without shuffling
- ✅ K-Fold reproducibility
- ✅ K-Fold with different k values (3, 5, 10)
- ✅ Stratified K-Fold basic functionality
- ✅ Stratified K-Fold class distribution preservation
- ✅ Leave-One-Out basic functionality
- ✅ Leave-One-Out with perfect classifier
- ✅ Shuffle Split basic functionality
- ✅ Shuffle Split with different test sizes
- ✅ Train-Test Split basic functionality
- ✅ Train-Test Split reproducibility
- ✅ Train-Test Split no overlap verification
- ✅ CrossValidationResult statistics
- ✅ CrossValidationResult toString
- ✅ Error handling: invalid k values
- ✅ Error handling: mismatched array sizes
- ✅ Error handling: invalid test sizes
- ✅ Error handling: invalid n_splits

**All 21 tests passing** ✅

## 🔄 Breaking Changes

None. This is a new feature addition with no impact on existing APIs.

## 📦 Dependencies

No new dependencies added. Uses existing:
- Apache Commons Math 3.6.1
- JUnit 5.10.1

## 🚀 Next Steps

With cross-validation infrastructure in place, upcoming features:
- **Naive Bayes** classifier (v1.0.6-alpha)
- **Support Vector Machines** (v1.0.7-alpha)
- **Gradient Boosting** (v1.0.8-alpha)
- Grid search and hyperparameter tuning

## 🐛 Known Issues

None.

## 📈 Performance Notes

- **K-Fold**: O(k * training_time) - efficient for most use cases
- **Stratified K-Fold**: O(k * training_time + n log n) - slightly slower due to sorting
- **LOOCV**: O(n * training_time) - use only for small datasets (n < 1000)
- **Shuffle Split**: O(splits * training_time) - very fast for multiple random evaluations
- **Train-Test Split**: O(n) - nearly instant for splitting

## 📄 Documentation

Updated documentation:
- ✅ README.md with comprehensive usage examples
- ✅ Javadoc for all public methods
- ✅ Release notes (this document)
- ✅ Example code in test suite

## 👥 Credits

**Author**: Matrix Agent  
**Repository**: https://github.com/yasmramos/MindForge

---

For questions, issues, or feature requests, please visit the [GitHub repository](https://github.com/yasmramos/MindForge).
