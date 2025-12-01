## 🌳 MindForge v1.0.2-alpha - Decision Trees

### ✨ New Features

This release introduces the **Decision Tree Classifier**, a powerful and interpretable machine learning algorithm implementing the CART (Classification and Regression Trees) algorithm.

#### 🎯 Decision Tree Classifier

**Key Features:**
- **Splitting Criteria**: Support for both Gini impurity and Entropy (Information Gain)
- **Flexible Configuration**: Builder pattern for easy customization
- **Hyperparameter Control**:
  - `maxDepth`: Control maximum tree depth to prevent overfitting
  - `minSamplesSplit`: Minimum samples required to split a node
  - `minSamplesLeaf`: Minimum samples required at leaf nodes
- **Probability Predictions**: Get class probabilities with `predictProba()`
- **Tree Introspection**: Query tree depth and number of leaves
- **Non-linear Decision Boundaries**: Can solve problems like XOR that linear models cannot

**Example Usage:**
```java
import com.mindforge.classification.DecisionTreeClassifier;

// Create tree with custom parameters
DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
    .maxDepth(5)
    .minSamplesSplit(2)
    .minSamplesLeaf(1)
    .criterion(DecisionTreeClassifier.Criterion.GINI)
    .build();

// Train the model
tree.train(X_train, y_train);

// Make predictions
int prediction = tree.predict(testPoint);

// Get probability predictions
double[] probabilities = tree.predictProba(testPoint);

// Get tree information
System.out.println("Tree depth: " + tree.getTreeDepth());
System.out.println("Number of leaves: " + tree.getNumLeaves());
```

**Supported Criteria:**
- **Gini Impurity**: Measures the probability of incorrect classification (default)
- **Entropy**: Measures information gain from a split

### 📈 Test Coverage

- **23 new tests** for Decision Tree Classifier
- **Total: 96 tests** passing successfully
- Complete coverage of:
  - Different splitting criteria (Gini and Entropy)
  - Hyperparameter constraints (maxDepth, minSamplesSplit, minSamplesLeaf)
  - Probability predictions
  - Edge cases (single class, XOR problem, multiclass)
  - Error handling and validation

### 📖 Documentation

- Updated README with Decision Tree usage examples
- Complete API documentation
- Inline code documentation with Javadoc
- Builder pattern examples

### 🔧 Improvements

- Version bumped to **1.0.2-alpha**
- Enhanced README with comprehensive examples
- Updated project roadmap

### 📦 Installation

```xml
<dependency>
    <groupId>com.mindforge</groupId>
    <artifactId>mindforge</artifactId>
    <version>1.0.2-alpha</version>
</dependency>
```

### 🎯 Algorithm Details

The Decision Tree implementation uses the CART algorithm with the following characteristics:

1. **Binary Splits**: Each node splits into exactly two children
2. **Greedy Search**: Finds the best split at each node by trying all features and thresholds
3. **Stopping Criteria**: Stops splitting when:
   - Maximum depth is reached
   - Minimum samples for split not met
   - Node is pure (all samples same class)
   - Only one class present in dataset

4. **Threshold Selection**: Tries midpoints between consecutive unique values for each feature

### 🔗 Resources

- [Main README](README.md)
- [Repository](https://github.com/yasmramos/MindForge)

### 📋 Full Changelog

**Added:**
- DecisionTreeClassifier with CART algorithm implementation
- Support for Gini and Entropy splitting criteria
- Builder pattern for flexible tree configuration
- Probability prediction capabilities
- Tree introspection methods (getTreeDepth, getNumLeaves)
- 23 comprehensive unit tests
- Decision Tree usage examples in README
- API documentation for Decision Trees

**Changed:**
- Version updated from 1.0.1-alpha to 1.0.2-alpha
- README updated with new features
- Roadmap updated (Decision Trees marked as complete)
- Test count updated to 96 total tests

---

**Author**: MindForge Team  
**Version**: 1.0.2-alpha  
**Date**: December 2025  
**Commit**: 911c13d
