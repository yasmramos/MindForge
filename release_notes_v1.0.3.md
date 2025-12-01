# MindForge v1.0.3-alpha Release Notes

## 🎉 New Features

### Random Forest Classifier
A powerful ensemble learning method that creates multiple decision trees and combines their predictions through majority voting. Random Forest reduces overfitting and improves generalization compared to individual decision trees.

**Key Features:**
- **Bootstrap Aggregating (Bagging)**: Creates diverse trees using random subsamples of training data
- **Random Feature Selection**: Considers random subsets of features at each split for decorrelation
- **Out-of-Bag (OOB) Evaluation**: Provides validation scores without needing a separate test set
- **Feature Importance**: Calculates and reports importance scores for each feature
- **Parallel Tree Building**: Efficient multi-threaded tree construction
- **Flexible Configuration**: Extensive hyperparameter tuning options via Builder pattern

**Hyperparameters:**
- `nEstimators`: Number of trees in the forest (default: 100)
- `maxFeatures`: Number of features to consider for splits ("sqrt", "log2", or integer)
- `maxDepth`: Maximum depth of each tree (default: unlimited)
- `minSamplesSplit`: Minimum samples required to split a node (default: 2)
- `minSamplesLeaf`: Minimum samples required at leaf nodes (default: 1)
- `criterion`: Splitting criterion (GINI or ENTROPY)
- `bootstrap`: Enable/disable bootstrap sampling (default: true)
- `randomState`: Random seed for reproducibility

**Usage Example:**
```java
import com.mindforge.classification.RandomForestClassifier;
import com.mindforge.classification.DecisionTreeClassifier.Criterion;

// Create and configure Random Forest
RandomForestClassifier rf = new RandomForestClassifier.Builder()
    .nEstimators(100)
    .maxFeatures("sqrt")
    .maxDepth(15)
    .criterion(Criterion.GINI)
    .randomState(42)
    .build();

// Train the model
rf.fit(X_train, y_train);

// Make predictions
int[] predictions = rf.predict(X_test);
double[][] probabilities = rf.predictProba(X_test);

// Evaluate model
double oobScore = rf.getOOBScore();
double[] featureImportance = rf.getFeatureImportance();
```

## 🔧 Enhancements

### Decision Tree Classifier Updates
Enhanced DecisionTreeClassifier to support Random Forest functionality:
- **Random Feature Selection**: Added `maxFeatures` parameter for selecting random feature subsets
- **Feature Importance Calculation**: Tracks and reports feature importance based on information gain
- **Batch Prediction Methods**: Added `predict(double[][])` and `fit()` methods for consistency
- **Reproducibility**: Added `randomState` parameter for deterministic behavior

## 📊 Test Coverage

- **Total Tests**: 122 (up from 96)
- **New Tests**: 26 comprehensive Random Forest tests
- **Test Categories**:
  - Binary and multiclass classification
  - Probability predictions and calibration
  - Out-of-bag score calculation
  - Feature importance validation
  - Different `maxFeatures` modes (sqrt, log2, integer)
  - Multiple splitting criteria (Gini, Entropy)
  - Hyperparameter validation
  - Reproducibility verification
  - Edge cases and error handling

## 🐛 Bug Fixes

- Fixed probability normalization in Random Forest to handle trees trained on bootstrap samples with missing classes
- Improved feature importance calculation and normalization in Decision Trees

## 📚 Documentation

- Added Random Forest usage examples to README
- Updated API documentation with Random Forest methods
- Added comprehensive Javadoc for all Random Forest classes and methods
- Updated project roadmap (Random Forest: ✅ Completed)

## 🏗️ Technical Details

**New Files:**
- `RandomForestClassifier.java` (568 lines)
- `RandomForestClassifierTest.java` (537 lines)

**Modified Files:**
- `DecisionTreeClassifier.java` (enhanced with 88 additional lines)
- `README.md` (updated with examples and API docs)
- `pom.xml` (version bump to 1.0.3-alpha)

**Performance:**
- Parallel tree building using Java Streams for improved training speed
- Efficient bootstrap sampling and feature selection algorithms
- Optimized probability aggregation across ensemble

## 📦 Installation

### Maven
```xml
<dependency>
    <groupId>com.mindforge</groupId>
    <artifactId>mindforge</artifactId>
    <version>1.0.3-alpha</version>
</dependency>
```

### Download JAR
Download the compiled JAR from the [releases page](https://github.com/yasmramos/MindForge/releases/tag/v1.0.3-alpha).

## 🔄 Migration Guide

### From v1.0.2-alpha
- No breaking changes
- All existing Decision Tree code remains compatible
- New Random Forest functionality is additive

## 🗺️ Roadmap

**Completed:**
- ✅ K-Nearest Neighbors (KNN)
- ✅ Decision Trees
- ✅ Random Forest
- ✅ Linear Regression
- ✅ K-Means Clustering
- ✅ Data Preprocessing Suite
- ✅ Evaluation Metrics

**Next Milestones:**
- Logistic Regression
- Naive Bayes
- Support Vector Machines (SVM)
- Cross-Validation
- Gradient Boosting

## 🙏 Acknowledgments

Random Forest algorithm based on the seminal work by Leo Breiman (2001). Implementation follows industry best practices from scikit-learn and other leading ML libraries.

## 📝 Full Changelog

- feat: Add Random Forest Classifier with ensemble learning
- feat: Add feature importance calculation to Decision Trees
- feat: Add random feature selection support
- feat: Add out-of-bag score evaluation
- test: Add 26 comprehensive Random Forest tests
- docs: Update README with Random Forest examples
- docs: Add Random Forest API documentation
- chore: Bump version to 1.0.3-alpha

---

**Full Test Results:**
```
Tests run: 122, Failures: 0, Errors: 0, Skipped: 0
BUILD SUCCESS
```

For detailed information, see the [GitHub repository](https://github.com/yasmramos/MindForge).
