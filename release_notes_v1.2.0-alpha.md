# MindForge v1.2.0-alpha Release Notes

**Release Date:** December 4, 2025

We are excited to announce MindForge v1.2.0-alpha, a major feature release that introduces neural network support, comprehensive validation metrics, enhanced dataset management, and a rich set of utility functions.

---

## Highlights

- **Neural Networks** - Build and train feedforward neural networks with dropout and batch normalization
- **Advanced Validation** - Confusion matrices, ROC curves, and AUC calculations
- **Dataset Management** - Load built-in datasets (Iris, Wine, Breast Cancer) or generate synthetic data
- **Utility Functions** - Comprehensive array operations, configuration management, and logging

---

## What's New

### Neural Networks Module

MindForge now includes a complete neural network implementation:

```java
// Create a neural network
NeuralNetwork nn = new NeuralNetwork(0.01, 100, 16);

// Add layers
nn.addDenseLayer(4, 16, "relu");      // Input -> Hidden
nn.addDenseLayer(16, 8, "relu");      // Hidden -> Hidden
nn.addDenseLayer(8, 3, "softmax");    // Hidden -> Output

// Train
nn.fit(trainX, trainY);

// Predict
double[] output = nn.forward(testSample);
```

**Features:**
- Dense layers with ReLU, Sigmoid, Tanh, and Softmax activations
- Dropout layers for regularization
- Batch normalization for stable training
- Mini-batch gradient descent

### Validation & Metrics

Comprehensive model evaluation tools:

```java
// Confusion Matrix
ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
System.out.println("Accuracy: " + cm.getAccuracy());
System.out.println("F1-Score: " + cm.getF1Score());

// ROC Curve
ROCCurve roc = new ROCCurve(yTrue, yScores);
System.out.println("AUC: " + roc.getAUC());
System.out.println("Optimal Threshold: " + roc.getOptimalThreshold());
```

**Features:**
- Binary and multiclass confusion matrices
- Precision, Recall, F1-Score calculations
- ROC curve generation with AUC
- Regression metrics (MSE, RMSE, MAE, R2)

### Dataset Management

Easy loading and manipulation of datasets:

```java
// Load built-in datasets
Dataset iris = DatasetLoader.loadIris();
Dataset wine = DatasetLoader.loadWine();
Dataset cancer = DatasetLoader.loadBreastCancer();

// Generate synthetic data
Dataset blobs = DatasetLoader.makeBlobs(200, 3, 2, 1.0);
Dataset circles = DatasetLoader.makeCircles(150, 0.1, 0.5);

// Split and manipulate
Dataset[] split = iris.trainTestSplit(0.2);
Dataset shuffled = iris.shuffle();
Dataset normalized = iris.normalize();
```

### Utility Functions

**ArrayUtils** - Statistical and array operations:
```java
double mean = ArrayUtils.mean(array);
double std = ArrayUtils.std(array);
double dot = ArrayUtils.dot(a, b);
double[][] transposed = ArrayUtils.transpose(matrix);
```

**Configuration** - Manage settings:
```java
Configuration config = Configuration.getDefault();
config.loadFromFile("config.properties");
String value = config.getString("key");
```

**MindForgeLogger** - Flexible logging:
```java
MindForgeLogger logger = MindForgeLogger.getLogger();
logger.setLevel(MindForgeLogger.Level.INFO);
logger.info("Training started with %d samples", trainSize);
```

---

## Example Code

### Complete ML Workflow

```java
// 1. Load data
Dataset iris = DatasetLoader.loadIris();

// 2. Split
Dataset[] split = iris.trainTestSplit(0.2);

// 3. Preprocess
StandardScaler scaler = new StandardScaler();
scaler.fit(split[0].getFeatures());
double[][] trainX = scaler.transform(split[0].getFeatures());
double[][] testX = scaler.transform(split[1].getFeatures());

// 4. Train neural network
NeuralNetwork nn = new NeuralNetwork(0.01, 100, 16);
nn.addDenseLayer(4, 16, "relu");
nn.addDenseLayer(16, 3, "softmax");
nn.fit(trainX, split[0].getLabels());

// 5. Evaluate
int[] predictions = new int[testX.length];
for (int i = 0; i < testX.length; i++) {
    predictions[i] = ArrayUtils.argmax(nn.forward(testX[i]));
}

ConfusionMatrix cm = new ConfusionMatrix(split[1].getLabels(), predictions);
System.out.println("Accuracy: " + cm.getAccuracy());
```

---

## Statistics

| Metric | Value |
|--------|-------|
| Total Tests | 1,030 |
| Test Pass Rate | 100% |
| New Classes | 20+ |
| Example Programs | 5 |
| Java Version | 11+ |

---

## Breaking Changes

None. This release is backward compatible with previous versions.

---

## Deprecations

None.

---

## Known Issues

- Neural network training may be slow on very large datasets (>100,000 samples)
- GPU acceleration not yet available

---

## Dependencies

- JUnit 5.10.0 (testing)
- Maven 3.8+ (build)

---

## Installation

### Maven

```xml
<dependency>
    <groupId>com.mindforge</groupId>
    <artifactId>mindforge</artifactId>
    <version>1.2.0-alpha</version>
</dependency>
```

### Manual

Download the JAR from the [Releases](https://github.com/yasmramos/MindForge/releases) page.

---

## Contributors

- MindForge Development Team

---

## What's Next

For v1.2.0 stable release:
- Performance optimizations
- Additional neural network layers (Conv, LSTM)
- GPU acceleration support
- More comprehensive documentation

---

**Full Changelog:** [v1.0.8...v1.2.0-alpha](https://github.com/yasmramos/MindForge/compare/v1.0.8...v1.2.0-alpha)
