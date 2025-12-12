# Changelog

All notable changes to MindForge ML Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0-alpha] - 2025-12-04

### Added

#### Neural Networks Module (`io.github.yasmramos.mindforge.neural`)
- **NeuralNetwork** - Flexible feedforward neural network implementation
  - Configurable learning rate, epochs, and batch size
  - Support for multiple layer types
  - Training with mini-batch gradient descent
- **DenseLayer** - Fully connected layer with multiple activations (ReLU, Sigmoid, Tanh, Softmax)
- **DropoutLayer** - Regularization layer with configurable dropout rate
- **BatchNormLayer** - Batch normalization for training stability
- **ActivationFunctions** - Comprehensive activation function library

#### Validation & Metrics (`io.github.yasmramos.mindforge.validation`)
- **ConfusionMatrix** - Binary and multiclass confusion matrix
  - True/False Positives and Negatives
  - Precision, Recall, F1-Score, Accuracy
  - Support for labeled classes
- **ROCCurve** - Receiver Operating Characteristic curve analysis
  - AUC (Area Under Curve) calculation
  - Optimal threshold finding
  - FPR/TPR curve points
- **Enhanced Metrics** - MSE, RMSE, MAE, R2 for regression

#### Dataset Management (`io.github.yasmramos.mindforge.data`)
- **Dataset** - Unified dataset container
  - Classification and regression support
  - Train/test split functionality
  - Shuffle and normalization
  - Subset creation
- **DatasetLoader** - Built-in dataset loading
  - Iris dataset (150 samples, 4 features, 3 classes)
  - Wine dataset (178 samples, 13 features, 3 classes)
  - Breast Cancer dataset (569 samples, 30 features, 2 classes)
  - Synthetic data generators (makeBlobs, makeCircles)

#### Utilities (`io.github.yasmramos.mindforge.util`)
- **ArrayUtils** - Comprehensive array operations
  - Statistical functions (mean, std, variance)
  - Vector operations (dot, add, multiply)
  - Matrix operations (transpose)
  - Normalization and shuffling
- **Configuration** - Configuration management
  - Properties file support
  - JSON configuration
  - Default configuration handling
- **MindForgeLogger** - Logging framework
  - Multiple log levels (DEBUG, INFO, WARN, ERROR)
  - File and console output
  - Formatted logging support

#### Examples (`io.github.yasmramos.mindforge.examples`)
- **NeuralNetworkExample** - Neural network training demo
- **ValidationMetricsExample** - Metrics and evaluation demo
- **DatasetWorkflowExample** - Complete data pipeline demo
- **FeatureEngineeringExample** - Feature selection and PCA demo
- **PipelineExample** - ML pipeline construction demo

### Changed
- Improved test coverage to 1,030 tests
- Enhanced documentation across all modules
- Updated README with new features and examples

### Fixed
- Fixed IOException handling in MindForgeLogger tests
- Corrected API consistency across all modules
- Fixed edge cases in array utility functions

---

## [1.0.8] - 2025-12-01

### Added
- Cross-validation framework with K-fold support
- Stratified sampling for classification
- Model persistence (save/load) functionality
- Additional preprocessing transformers

### Fixed
- Memory optimization in large dataset handling
- Improved numerical stability in regression models

---

## [1.0.7] - 2025-12-01

### Added
- Pipeline and ColumnTransformer classes
- GridSearchCV for hyperparameter tuning
- Enhanced feature selection methods

---

## [1.0.6] - 2025-12-01

### Added
- Recursive Feature Elimination (RFE)
- SelectKBest with multiple scoring functions
- VarianceThreshold feature selector

---

## [1.0.5] - 2025-12-01

### Added
- Principal Component Analysis (PCA)
- Polynomial Features generator
- Target Encoder for categorical variables

---

## [1.0.4] - 2025-12-01

### Added
- One-Hot Encoder for categorical data
- Simple Imputer for missing values
- Label Encoder for classification targets

---

## [1.0.3] - 2025-12-01

### Added
- MinMaxScaler for feature normalization
- StandardScaler for standardization
- Data split utilities (train/test split)

---

## [1.0.2] - 2025-12-01

### Added
- Ensemble methods (VotingClassifier)
- Additional clustering algorithms (DBSCAN, Hierarchical)
- Distance metrics module

---

## [1.0.1] - 2025-12-01

### Added
- Ridge, Lasso, and ElasticNet regression
- K-Means clustering improvements
- Basic metrics for classification

---

## [1.0.0] - 2025-12-01

### Added
- Initial release of MindForge ML Library
- Linear Regression
- Logistic Regression
- K-Means Clustering
- K-Nearest Neighbors
- Decision Trees
- Naive Bayes classifier
- Basic preprocessing utilities
