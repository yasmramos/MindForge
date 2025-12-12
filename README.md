# MindForge

A Machine Learning and Artificial Intelligence library for Java, inspired by libraries like Smile, designed to be easy to use and efficient.

[![Build Status](https://github.com/yasmramos/MindForge/actions/workflows/ci.yml/badge.svg)](https://github.com/yasmramos/MindForge/actions/workflows/ci.yml)
[![Java Version](https://img.shields.io/badge/Java-11%2B-blue)](https://www.oracle.com/java/)
[![Maven](https://img.shields.io/badge/Maven-3.6%2B-red)](https://maven.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## ⚡ Quick Start

Get started with MindForge in minutes! Here's a complete example that demonstrates classification, regression, and clustering:

```java
import com.mindforge.classification.*;
import com.mindforge.regression.LinearRegression;
import com.mindforge.clustering.KMeans;
import com.mindforge.preprocessing.StandardScaler;
import com.mindforge.validation.Metrics;

public class QuickStart {
    public static void main(String[] args) {
        // === CLASSIFICATION with KNN ===
        double[][] X_class = {{1,2}, {2,3}, {3,3}, {6,5}, {7,8}, {8,7}};
        int[] y_class = {0, 0, 0, 1, 1, 1};
        
        KNearestNeighbors knn = new KNearestNeighbors(3);
        knn.train(X_class, y_class);
        System.out.println("KNN Accuracy: " + Metrics.accuracy(y_class, knn.predict(X_class)) * 100 + "%");
        
        // === REGRESSION ===
        double[][] X_reg = {{1}, {2}, {3}, {4}, {5}};
        double[] y_reg = {2.1, 4.0, 5.9, 8.1, 10.0};
        
        LinearRegression lr = new LinearRegression();
        lr.train(X_reg, y_reg);
        System.out.println("Prediction for x=6: " + lr.predict(new double[]{6}));
        
        // === CLUSTERING ===
        double[][] data = {{1,2}, {1.5,1.8}, {5,8}, {8,8}, {1,0.6}, {9,11}};
        
        KMeans kmeans = new KMeans(2);
        kmeans.fit(data, 2);
        int[] clusters = kmeans.cluster(data);
        System.out.println("Cluster assignments: " + java.util.Arrays.toString(clusters));
        
        // === PREPROCESSING ===
        StandardScaler scaler = new StandardScaler();
        scaler.fit(X_class);
        double[][] X_scaled = scaler.transform(X_class);
        System.out.println("Scaled first sample: " + java.util.Arrays.toString(X_scaled[0]));
    }
}
```

> 📚 **More Examples**: Check out our [comprehensive examples](examples/README.md) including Regression, Preprocessing, Pipelines, and Cross-Validation!

## 🚀 Features

### Core Algorithms
- **Classification**: K-Nearest Neighbors (KNN), Decision Trees, Random Forest, Logistic Regression, Naive Bayes (Gaussian, Multinomial, Bernoulli), Support Vector Machines (SVM), Gradient Boosting
- **Regression**: Linear Regression, Ridge Regression (L2), Lasso Regression (L1), ElasticNet, Polynomial Regression, Support Vector Regression (SVR) with multiple kernels (Linear, RBF, Polynomial, Sigmoid)
- **Clustering**: K-Means with multiple initialization strategies
- **Dimensionality Reduction**: PCA, Linear Discriminant Analysis (LDA) for supervised dimensionality reduction
- **Neural Networks**: Multi-Layer Perceptron (MLP) with backpropagation, multiple activation functions (Sigmoid, ReLU, Tanh, Softmax, Leaky ReLU, ELU), Dropout and Batch Normalization layers
- **Recurrent Neural Networks**: RNN and LSTM (Long Short-Term Memory) for sequence modeling
- **GPU/CPU Acceleration**: Hardware acceleration for compute-intensive operations

### Data Processing
- **Preprocessing**: MinMaxScaler, StandardScaler, SimpleImputer, LabelEncoder, DataSplit
- **Feature Selection**: VarianceThreshold, SelectKBest (F-test, Chi2, Mutual Info), RFE (Recursive Feature Elimination)
- **Pipelines**: Chain transformers and estimators for streamlined workflows
- **Dataset Management**: Built-in datasets (Iris, Wine, Breast Cancer, Boston Housing), train/test splitting

### Model Management
- **Persistence**: Save/Load models to disk or byte arrays
- **Validation**: Cross-Validation (K-Fold, Stratified K-Fold, LOOCV, Shuffle Split), Train-Test Split
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, AUC, MSE, RMSE, MAE, R²
- **Distance Functions**: Euclidean, Manhattan, Chebyshev, Minkowski

### Utilities
- **Logging**: Comprehensive logging system with multiple levels (DEBUG, INFO, WARN, ERROR, FATAL)
- **Configuration**: YAML and Properties file support for application settings
- **Array Utils**: Matrix operations, statistics, normalization, one-hot encoding
- **Visualization**: Chart generation (line, scatter, bar, heatmap) to PNG files
- **API Server**: REST API for model serving and predictions

### Developer Experience
- **8 Comprehensive Examples**: QuickStart, Clustering, Regression, Preprocessing, Pipelines, Validation, Neural Networks, Visualization
- **Simple and Consistent API**: Intuitive interfaces across all algorithms
- **CI/CD Integration**: Automated testing with GitHub Actions

## 📦 Project Structure

```
MindForge/
├── src/main/java/com/mindforge/
│   ├── classification/     # Classification algorithms
│   │   ├── Classifier.java
│   │   └── KNearestNeighbors.java
│   ├── regression/         # Regression algorithms
│   │   ├── Regressor.java
│   │   └── LinearRegression.java
│   ├── clustering/         # Clustering algorithms
│   │   ├── Clusterer.java
│   │   └── KMeans.java
│   ├── preprocessing/     # Data preprocessing
│   │   ├── MinMaxScaler.java
│   │   ├── StandardScaler.java
│   │   ├── SimpleImputer.java
│   │   ├── LabelEncoder.java
│   │   └── DataSplit.java
│   ├── feature/           # Feature selection
│   │   ├── VarianceThreshold.java
│   │   ├── SelectKBest.java
│   │   └── RFE.java
│   ├── decomposition/     # Dimensionality reduction
│   │   └── PCA.java
│   ├── persistence/       # Model save/load
│   │   └── ModelPersistence.java
│   ├── math/              # Mathematical functions
│   │   └── Distance.java
│   ├── validation/        # Evaluation metrics
│   │   └── Metrics.java
│   ├── neural/            # Neural networks (MLP, layers, activations)
│   ├── data/              # Dataset management and loaders
│   ├── visualization/     # Chart generation
│   ├── api/               # REST API server and client
│   └── util/              # Logging, configuration, array utilities
└── pom.xml
```

## 🔧 Requirements

- **Java 11** or higher
- **Maven 3.6** or higher

## 📥 Installation

### Option 1: GitHub Packages (Recommended)

Add the GitHub Packages repository to your `pom.xml`:

```xml
<repositories>
    <repository>
        <id>github</id>
        <url>https://maven.pkg.github.com/yasmramos/MindForge</url>
    </repository>
</repositories>

<dependencies>
    <dependency>
        <groupId>com.mindforge</groupId>
        <artifactId>mindforge</artifactId>
        <version>1.2.0-alpha</version>
    </dependency>
</dependencies>
```

**Note:** You need to authenticate with GitHub Packages. Add this to your `~/.m2/settings.xml`:
```xml
<servers>
    <server>
        <id>github</id>
        <username>YOUR_GITHUB_USERNAME</username>
        <password>YOUR_GITHUB_TOKEN</password>
    </server>
</servers>
```

### Option 2: Download JAR

Download the latest release from [GitHub Releases](https://github.com/yasmramos/MindForge/releases) and add it to your project:

**Maven (local JAR):**
```bash
mvn install:install-file -Dfile=mindforge-1.2.0-alpha.jar \
  -DgroupId=com.mindforge -DartifactId=mindforge \
  -Dversion=1.2.0-alpha -Dpackaging=jar
```

### Option 2: Build from Source

```bash
git clone https://github.com/yasmramos/MindForge.git
cd MindForge
mvn clean install
```

The JAR will be generated at `target/mindforge-1.2.0-alpha.jar`.

## 💡 Usage Examples

### Classification with K-Nearest Neighbors

```java
import com.mindforge.classification.KNearestNeighbors;
import com.mindforge.validation.Metrics;

// Training data
double[][] X_train = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0}, {6.0, 5.0}, {7.0, 8.0}, {8.0, 7.0}};
int[] y_train = {0, 0, 0, 1, 1, 1};

// Create and train the model
KNearestNeighbors knn = new KNearestNeighbors(3);
knn.train(X_train, y_train);

// Make predictions
double[] testPoint = {5.0, 5.0};
int prediction = knn.predict(testPoint);
System.out.println("Prediction: " + prediction);

// Evaluate the model
int[] predictions = knn.predict(X_train);
double accuracy = Metrics.accuracy(y_train, predictions);
System.out.println("Accuracy: " + accuracy);
```

### Classification with Decision Trees

```java
import com.mindforge.classification.DecisionTreeClassifier;
import com.mindforge.validation.Metrics;

// Training data
double[][] X_train = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0}, {6.0, 5.0}, {7.0, 8.0}, {8.0, 7.0}};
int[] y_train = {0, 0, 0, 1, 1, 1};

// Create and train the model with custom parameters
DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
    .maxDepth(5)
    .minSamplesSplit(2)
    .criterion(DecisionTreeClassifier.Criterion.GINI)
    .build();
tree.train(X_train, y_train);

// Make predictions
double[] testPoint = {5.0, 5.0};
int prediction = tree.predict(testPoint);
System.out.println("Prediction: " + prediction);

// Get probability predictions
double[] probabilities = tree.predictProba(testPoint);
System.out.println("Class probabilities: " + Arrays.toString(probabilities));

// Evaluate the model
int[] predictions = tree.predict(X_train);
double accuracy = Metrics.accuracy(y_train, predictions);
System.out.println("Accuracy: " + accuracy);
System.out.println("Tree depth: " + tree.getTreeDepth());
```

### Random Forest Classification

```java
import com.mindforge.classification.RandomForestClassifier;
import com.mindforge.classification.DecisionTreeClassifier;
import com.mindforge.validation.Metrics;

// Training data
double[][] X_train = {{1.0, 2.0}, {2.0, 3.0}, {8.0, 8.0}, {9.0, 10.0}};
int[] y_train = {0, 0, 1, 1};

// Create and train Random Forest with custom parameters
RandomForestClassifier rf = new RandomForestClassifier.Builder()
    .nEstimators(100)              // Number of trees
    .maxFeatures("sqrt")           // Features to consider at each split
    .maxDepth(10)                  // Maximum tree depth
    .minSamplesSplit(2)            // Minimum samples to split
    .criterion(DecisionTreeClassifier.Criterion.GINI)
    .bootstrap(true)               // Use bootstrap sampling
    .randomState(42)               // For reproducibility
    .build();

rf.fit(X_train, y_train);

// Make predictions
double[] testPoint = {5.0, 5.0};
int[] predictions = rf.predict(new double[][]{testPoint});
System.out.println("Prediction: " + predictions[0]);

// Get probability predictions
double[][] probabilities = rf.predictProba(new double[][]{testPoint});
System.out.println("Class probabilities: " + Arrays.toString(probabilities[0]));

// Evaluate the model
double oobScore = rf.getOOBScore();
System.out.println("Out-of-bag score: " + oobScore);

// Get feature importance
double[] importance = rf.getFeatureImportance();
System.out.println("Feature importance: " + Arrays.toString(importance));
```

### Logistic Regression Classification

```java
import io.mindforge.classification.LogisticRegression;
import com.mindforge.validation.Metrics;

// Training data
double[][] X_train = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0}, {8.0, 8.0}, {9.0, 10.0}, {10.0, 11.0}};
int[] y_train = {0, 0, 0, 1, 1, 1};

// Create and train Logistic Regression with L2 regularization
LogisticRegression lr = new LogisticRegression.Builder()
    .penalty("l2")                 // Regularization: "l1", "l2", "elasticnet", "none"
    .C(1.0)                        // Inverse regularization strength
    .solver("gradient_descent")    // Solver: "gradient_descent", "sgd", "newton_cg"
    .learningRate(0.1)             // Learning rate
    .maxIter(1000)                 // Maximum iterations
    .tol(1e-4)                     // Convergence tolerance
    .randomState(42)               // For reproducibility
    .build();

lr.fit(X_train, y_train);

// Make predictions
double[][] X_test = {{5.0, 5.0}, {2.0, 2.5}};
int[] predictions = lr.predict(X_test);
System.out.println("Predictions: " + Arrays.toString(predictions));

// Get probability predictions
double[][] probabilities = lr.predictProba(X_test);
for (int i = 0; i < probabilities.length; i++) {
    System.out.println("Sample " + i + " probabilities: " + Arrays.toString(probabilities[i]));
}

// Access model parameters
double[][] coefficients = lr.getCoefficients();
System.out.println("Coefficients: " + Arrays.deepToString(coefficients));

// View training history
List<Double> lossHistory = lr.getLossHistory();
System.out.println("Final loss: " + lossHistory.get(lossHistory.size() - 1));
```

### Cross-Validation

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

// K-Fold Cross-Validation
CrossValidationResult kFoldResult = CrossValidation.kFold(
    trainer, predictor, X, y, 5, 42
);
System.out.println("K-Fold Mean Accuracy: " + kFoldResult.getMean());
System.out.println("K-Fold Std Dev: " + kFoldResult.getStdDev());

// Stratified K-Fold (maintains class proportions)
CrossValidationResult stratifiedResult = CrossValidation.stratifiedKFold(
    trainer, predictor, X, y, 5, 42
);
System.out.println("Stratified K-Fold Mean: " + stratifiedResult.getMean());

// Leave-One-Out Cross-Validation
CrossValidationResult loocvResult = CrossValidation.leaveOneOut(
    trainer, predictor, X, y
);
System.out.println("LOOCV Mean: " + loocvResult.getMean());

// Shuffle Split Cross-Validation
CrossValidationResult shuffleResult = CrossValidation.shuffleSplit(
    trainer, predictor, X, y, 10, 0.2, 42
);
System.out.println("Shuffle Split Mean: " + shuffleResult.getMean());

// Train-Test Split
CrossValidation.SplitData split = CrossValidation.trainTestSplit(X, y, 0.3, 42);
KNearestNeighbors model = new KNearestNeighbors(3);
model.train(split.XTrain, split.yTrain);
int[] predictions = model.predict(split.XTest);
double accuracy = Metrics.accuracy(split.yTest, predictions);
System.out.println("Test Accuracy: " + accuracy);
```

### Naive Bayes Classification

```java
import com.mindforge.classification.GaussianNaiveBayes;
import com.mindforge.classification.MultinomialNaiveBayes;
import com.mindforge.classification.BernoulliNaiveBayes;
import com.mindforge.validation.Metrics;

// === Gaussian Naive Bayes (for continuous features) ===
double[][] X_continuous = {{-2.0, -2.0}, {-1.8, -2.2}, {2.0, 2.0}, {1.8, 2.2}};
int[] y_continuous = {0, 0, 1, 1};

GaussianNaiveBayes gnb = new GaussianNaiveBayes();
gnb.train(X_continuous, y_continuous);

int[] pred_gnb = gnb.predict(X_continuous);
double[][] proba_gnb = gnb.predictProba(X_continuous);
System.out.println("Gaussian Predictions: " + Arrays.toString(pred_gnb));

// === Multinomial Naive Bayes (for count features, e.g., word counts) ===
double[][] X_counts = {{5.0, 2.0, 0.0}, {0.0, 3.0, 5.0}, {6.0, 1.0, 0.0}, {0.0, 4.0, 6.0}};
int[] y_counts = {0, 1, 0, 1};

MultinomialNaiveBayes mnb = new MultinomialNaiveBayes(1.0); // alpha=1.0 (Laplace smoothing)
mnb.train(X_counts, y_counts);

int[] pred_mnb = mnb.predict(X_counts);
System.out.println("Multinomial Predictions: " + Arrays.toString(pred_mnb));

// === Bernoulli Naive Bayes (for binary features) ===
double[][] X_binary = {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
int[] y_binary = {0, 1, 0, 1};

BernoulliNaiveBayes bnb = new BernoulliNaiveBayes(1.0); // alpha=1.0 for smoothing
bnb.train(X_binary, y_binary);

int[] pred_bnb = bnb.predict(X_binary);
double[][] proba_bnb = bnb.predictProba(X_binary);
System.out.println("Bernoulli Predictions: " + Arrays.toString(pred_bnb));
```

### Support Vector Machines (SVM)

```java
import com.mindforge.classification.SVC;
import com.mindforge.validation.Metrics;

// Training data
double[][] X_svm = {
    {-2.0, -2.0}, {-1.8, -2.2}, {-2.2, -1.8},
    {2.0, 2.0}, {1.8, 2.2}, {2.2, 1.8}
};
int[] y_svm = {0, 0, 0, 1, 1, 1};

// Create and train SVM with custom parameters
SVC svc = new SVC.Builder()
    .C(1.0)                   // Regularization parameter
    .maxIter(1000)            // Maximum iterations
    .tol(1e-3)                // Tolerance
    .learningRate(0.01)       // Learning rate
    .build();

svc.train(X_svm, y_svm);

// Make predictions
int[] predictions = svc.predict(X_svm);
System.out.println("SVM Predictions: " + Arrays.toString(predictions));

// Get decision scores
double[] scores = svc.decisionFunction(X_svm[0]);
System.out.println("Decision scores: " + Arrays.toString(scores));

// Access model parameters
double[][] weights = svc.getWeights();
double[] bias = svc.getBias();
System.out.println("Number of classes: " + svc.getNumClasses());
```

### Gradient Boosting Classification

```java
import com.mindforge.classification.GradientBoostingClassifier;
import com.mindforge.validation.Metrics;

// Training data
double[][] X_train = {
    {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},  // Class 0
    {1.0, 1.0}, {1.1, 1.1}, {1.2, 1.2}   // Class 1
};
int[] y_train = {0, 0, 0, 1, 1, 1};

// Create and train Gradient Boosting with custom parameters
GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
    .nEstimators(100)           // Number of boosting stages
    .learningRate(0.1)          // Shrinkage parameter
    .maxDepth(3)                // Maximum depth of trees
    .subsample(1.0)             // Fraction of samples for fitting
    .randomState(42)            // For reproducibility
    .build();

gb.fit(X_train, y_train);

// Make predictions
int[] predictions = gb.predict(X_train);
System.out.println("Predictions: " + Arrays.toString(predictions));

// Get probability predictions
double[][] probabilities = gb.predictProba(X_train);
for (int i = 0; i < probabilities.length; i++) {
    System.out.println("Sample " + i + " probabilities: " + Arrays.toString(probabilities[i]));
}

// Model information
System.out.println("Number of trees: " + gb.getNumTrees());
System.out.println("Classes: " + Arrays.toString(gb.getClasses()));

// Evaluate
double accuracy = Metrics.accuracy(y_train, predictions);
System.out.println("Training accuracy: " + accuracy);
```

### Linear Regression

```java
import com.mindforge.regression.LinearRegression;
import com.mindforge.validation.Metrics;

// Training data
double[][] X_train = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
double[] y_train = {2.0, 4.0, 6.0, 8.0, 10.0};

// Create and train the model
LinearRegression lr = new LinearRegression();
lr.train(X_train, y_train);

// Make predictions
double[] testPoint = {6.0};
double prediction = lr.predict(testPoint);
System.out.println("Prediction: " + prediction);

// Evaluate the model
double[] predictions = lr.predict(X_train);
double rmse = Metrics.rmse(y_train, predictions);
System.out.println("RMSE: " + rmse);
```

### Advanced Regression (Ridge, Lasso, ElasticNet, SVR)

```java
import com.mindforge.regression.*;
import com.mindforge.validation.Metrics;

// Training data
double[][] X = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}};
double[] y = {2.1, 4.0, 5.9, 8.1, 10.0, 12.1, 13.9, 16.0};

// === Ridge Regression (L2 regularization) ===
RidgeRegression ridge = new RidgeRegression(1.0); // alpha = 1.0
ridge.train(X, y);
double[] ridgePred = ridge.predict(X);
System.out.println("Ridge R²: " + Metrics.r2Score(y, ridgePred));

// === Lasso Regression (L1 regularization) ===
LassoRegression lasso = new LassoRegression(0.1); // alpha = 0.1
lasso.train(X, y);
double[] lassoPred = lasso.predict(X);
System.out.println("Lasso R²: " + Metrics.r2Score(y, lassoPred));

// === ElasticNet (L1 + L2 combined) ===
ElasticNetRegression elasticnet = new ElasticNetRegression(0.1, 0.5); // alpha, l1_ratio
elasticnet.train(X, y);
double[] enPred = elasticnet.predict(X);
System.out.println("ElasticNet R²: " + Metrics.r2Score(y, enPred));

// === Polynomial Regression ===
PolynomialRegression poly = new PolynomialRegression(2); // degree = 2
poly.train(X, y);
double polyPred = poly.predict(new double[]{9});
System.out.println("Polynomial prediction for x=9: " + polyPred);

// === Support Vector Regression (SVR) ===
// Linear kernel
SVR svrLinear = new SVR.Builder()
    .kernel(SVR.KernelType.LINEAR)
    .C(1.0)
    .epsilon(0.1)
    .build();
svrLinear.train(X, y);

// RBF kernel (for non-linear patterns)
SVR svrRBF = new SVR.Builder()
    .kernel(SVR.KernelType.RBF)
    .C(10.0)
    .epsilon(0.1)
    .gamma(0.5)
    .build();
svrRBF.train(X, y);
System.out.println("SVR Support Vectors: " + svrRBF.getNumSupportVectors());

// Polynomial kernel
SVR svrPoly = new SVR.Builder()
    .kernel(SVR.KernelType.POLYNOMIAL)
    .C(1.0)
    .degree(3)
    .build();
svrPoly.train(X, y);
```

### Linear Discriminant Analysis (LDA)

```java
import com.mindforge.decomposition.LinearDiscriminantAnalysis;

// Training data with 3 classes
double[][] X = {
    {4.0, 2.0}, {4.5, 2.5}, {4.2, 2.1},  // Class 0
    {1.0, 4.0}, {1.5, 4.5}, {1.2, 4.2},  // Class 1
    {5.0, 5.0}, {5.5, 5.5}, {5.2, 5.2}   // Class 2
};
int[] y = {0, 0, 0, 1, 1, 1, 2, 2, 2};

// Create LDA for dimensionality reduction
LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis(2); // 2 components
lda.fit(X, y);

// Transform data to lower dimensions
double[][] X_transformed = lda.transform(X);
System.out.println("Original dimensions: " + X[0].length);
System.out.println("Transformed dimensions: " + X_transformed[0].length);

// Use as classifier
int[] predictions = lda.predict(X);
System.out.println("Predictions: " + Arrays.toString(predictions));

// Get explained variance ratio
double[] varianceRatio = lda.getExplainedVarianceRatio();
System.out.println("Explained variance: " + Arrays.toString(varianceRatio));
```

### Recurrent Neural Networks (RNN/LSTM)

```java
import com.mindforge.neural.rnn.*;

// === Simple RNN for sequence prediction ===
int inputSize = 10;    // Input features per timestep
int hiddenSize = 32;   // Hidden state size
int outputSize = 5;    // Output size

SimpleRNN rnn = new SimpleRNN(inputSize, hiddenSize, outputSize);

// Training sequence data (batch_size, seq_length, input_size)
double[][][] sequences = new double[100][20][inputSize];
double[][] targets = new double[100][outputSize];
// ... populate with data ...

rnn.fit(sequences, targets, 100, 0.01); // epochs, learning_rate

// Predict
double[][] output = rnn.predict(sequences[0]);

// === LSTM for long sequences ===
LSTM lstm = new LSTM.Builder()
    .inputSize(inputSize)
    .hiddenSize(64)
    .outputSize(outputSize)
    .numLayers(2)           // Stacked LSTM layers
    .dropout(0.2)           // Dropout between layers
    .build();

lstm.fit(sequences, targets, 100, 0.001);

// LSTM handles long-term dependencies better
double[][] lstmOutput = lstm.predict(sequences[0]);
System.out.println("LSTM output shape: " + lstmOutput.length + " x " + lstmOutput[0].length);

// Get hidden states for analysis
double[][] hiddenStates = lstm.getLastHiddenState();
double[][] cellStates = lstm.getLastCellState();
```

### Clustering with K-Means

```java
import com.mindforge.clustering.KMeans;

// Data
double[][] data = {
    {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, 
    {8.0, 8.0}, {1.0, 0.6}, {9.0, 11.0}
};

// Create and run K-Means
KMeans kmeans = new KMeans(2);
kmeans.fit(data, 2);

// Get cluster assignments
int[] clusters = new int[data.length];
for (int i = 0; i < data.length; i++) {
    clusters[i] = kmeans.predict(data[i]);
    System.out.println("Point " + i + " -> Cluster " + clusters[i]);
}

// Get centroids
double[][] centroids = kmeans.getCentroids();
```

### Data Preprocessing

```java
import com.mindforge.preprocessing.*;

// Normalize features to [0, 1]
double[][] data = {{1.0, 100.0}, {2.0, 200.0}, {3.0, 300.0}};
MinMaxScaler scaler = new MinMaxScaler();
scaler.fit(data);
double[][] normalized = scaler.transform(data);

// Standardize features (mean=0, std=1)
StandardScaler standardScaler = new StandardScaler();
standardScaler.fit(data);
double[][] standardized = standardScaler.transform(data);

// Handle missing values
double[][] dataWithNaN = {{1.0, Double.NaN}, {2.0, 200.0}, {Double.NaN, 300.0}};
SimpleImputer imputer = new SimpleImputer(SimpleImputer.Strategy.MEAN);
imputer.fit(dataWithNaN);
double[][] imputed = imputer.transform(dataWithNaN);

// Encode categorical labels
String[] labels = {"cat", "dog", "cat", "bird", "dog"};
LabelEncoder encoder = new LabelEncoder();
int[] encoded = encoder.encode(labels);
String[] decoded = encoder.decode(encoded);

// Split data into train/test sets
double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}};
int[] y = {0, 0, 1, 1};
DataSplit.Split split = DataSplit.trainTestSplit(X, y, 0.25, 42);
// Access: split.XTrain, split.XTest, split.yTrain, split.yTest
```

### Feature Selection

```java
import com.mindforge.feature.VarianceThreshold;
import com.mindforge.feature.SelectKBest;
import com.mindforge.feature.RFE;

// === VarianceThreshold - Remove low-variance features ===
double[][] X = {
    {0.0, 2.0, 0.0, 3.0},
    {0.0, 1.0, 4.0, 3.0},
    {0.0, 1.0, 1.0, 3.0}
};

VarianceThreshold vt = new VarianceThreshold(0.0); // Remove constant features
double[][] X_vt = vt.fitTransform(X);
System.out.println("Features remaining: " + X_vt[0].length);
System.out.println("Selected indices: " + Arrays.toString(vt.getSelectedFeatureIndices()));

// === SelectKBest - Select top k features by statistical tests ===
double[][] X_train = {
    {1.0, 2.0, 0.1}, {1.2, 2.1, 0.2},
    {5.0, 2.0, 0.15}, {5.2, 2.1, 0.18}
};
int[] y_train = {0, 0, 1, 1};

// Using F-classif (ANOVA F-value)
SelectKBest selector = new SelectKBest(SelectKBest.ScoreFunction.F_CLASSIF, 2);
double[][] X_best = selector.fitTransform(X_train, y_train);
System.out.println("Feature scores: " + Arrays.toString(selector.getScores()));

// Using Chi-squared (for non-negative features)
SelectKBest chi2Selector = new SelectKBest(SelectKBest.ScoreFunction.CHI2, 2);
chi2Selector.fit(X_train, y_train);

// Using Mutual Information
SelectKBest miSelector = new SelectKBest(SelectKBest.ScoreFunction.MUTUAL_INFO, 2);
miSelector.fit(X_train, y_train);

// === RFE - Recursive Feature Elimination ===
RFE rfe = new RFE(2);           // Select 2 features
// RFE rfe = new RFE(2, 1);     // Select 2, remove 1 at a time
rfe.fit(X_train, y_train);

double[][] X_rfe = rfe.transform(X_train);
System.out.println("Feature rankings: " + Arrays.toString(rfe.getRanking()));
System.out.println("Feature importances: " + Arrays.toString(rfe.getFeatureImportances()));
```

### PCA (Principal Component Analysis)

```java
import com.mindforge.decomposition.PCA;

// Create sample data
double[][] X = {
    {1.0, 2.0, 3.0, 4.0},
    {2.0, 3.0, 4.0, 5.0},
    {3.0, 4.0, 5.0, 6.0},
    {4.0, 5.0, 6.0, 7.0}
};

// Reduce to 2 components
PCA pca = new PCA(2);
double[][] X_reduced = pca.fitTransform(X);
System.out.println("Original dimensions: " + X[0].length);
System.out.println("Reduced dimensions: " + X_reduced[0].length);

// Get explained variance
double[] varianceRatio = pca.getExplainedVarianceRatio();
System.out.println("Explained variance ratio: " + Arrays.toString(varianceRatio));

double[] cumulative = pca.getCumulativeExplainedVarianceRatio();
System.out.println("Cumulative variance: " + Arrays.toString(cumulative));

// Reconstruct original data (with some loss)
double[][] X_reconstructed = pca.inverseTransform(X_reduced);

// Get principal components
double[][] components = pca.getComponents();
System.out.println("Number of components: " + pca.getNumberOfComponents());
```

### Model Persistence

```java
import com.mindforge.persistence.ModelPersistence;
import com.mindforge.classification.GaussianNaiveBayes;

// Train a model
double[][] X_train = {{-2.0, -2.0}, {-1.8, -2.2}, {2.0, 2.0}, {1.8, 2.2}};
int[] y_train = {0, 0, 1, 1};

GaussianNaiveBayes model = new GaussianNaiveBayes();
model.train(X_train, y_train);

// Save model to file
ModelPersistence.save(model, "my_model.bin");

// Load model from file
GaussianNaiveBayes loadedModel = ModelPersistence.load("my_model.bin");

// Or with type checking
GaussianNaiveBayes typedModel = ModelPersistence.load("my_model.bin", GaussianNaiveBayes.class);

// Make predictions with loaded model
int[] predictions = loadedModel.predict(X_train);
System.out.println("Predictions: " + Arrays.toString(predictions));

// Get model metadata without fully loading
ModelPersistence.ModelMetadata metadata = ModelPersistence.getMetadata("my_model.bin");
System.out.println("Model class: " + metadata.getSimpleClassName());
System.out.println("File size: " + metadata.getFileSize() + " bytes");

// Check if file is valid
boolean isValid = ModelPersistence.isValidModelFile("my_model.bin");

// Serialize to byte array (for network transfer or database storage)
byte[] bytes = ModelPersistence.toBytes(model);
GaussianNaiveBayes fromBytes = ModelPersistence.fromBytes(bytes);
```

### Neural Networks (MLP)

```java
import com.mindforge.neural.*;

// Create a neural network for XOR problem
NeuralNetwork network = new NeuralNetwork();
network.addLayer(new DenseLayer(2, 8, "relu"));     // Input: 2 features, 8 neurons
network.addLayer(new DropoutLayer(0.2));             // 20% dropout for regularization
network.addLayer(new DenseLayer(8, 4, "relu"));     // Hidden layer
network.addLayer(new DenseLayer(4, 1, "sigmoid")); // Output: 1 neuron for binary classification

// Training data (XOR)
double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
double[][] y = {{0}, {1}, {1}, {0}};

// Train the network
network.fit(X, y, 1000, 4); // 1000 epochs, batch size 4

// Make predictions
double[] prediction = network.predict(new double[]{0, 1});
System.out.println("Prediction for [0,1]: " + prediction[0]);

// === Activation Functions ===
double[] input = {-1.0, 0.0, 1.0};
double[] sigmoidOut = ActivationFunction.sigmoid(input);
double[] reluOut = ActivationFunction.relu(input);
double[] tanhOut = ActivationFunction.tanh(input);
double[] softmaxOut = ActivationFunction.softmax(input);
```

### Dataset Loaders

```java
import com.mindforge.data.*;

// Load built-in datasets
Dataset iris = DatasetLoader.loadIris();           // 150 samples, 4 features, 3 classes
Dataset wine = DatasetLoader.loadWine();           // 178 samples, 13 features, 3 classes
Dataset cancer = DatasetLoader.loadBreastCancer(); // 569 samples, 30 features, 2 classes
Dataset boston = DatasetLoader.loadBostonHousing(); // 506 samples, 13 features (regression)

// Dataset information
System.out.println("Samples: " + iris.getNumSamples());
System.out.println("Features: " + iris.getNumFeatures());
System.out.println("Feature names: " + Arrays.toString(iris.getFeatureNames()));

// Train-test split
Dataset[] splits = iris.trainTestSplit(0.3, true, 42L); // 30% test, shuffle, seed 42
Dataset train = splits[0];
Dataset test = splits[1];

// Access data
double[][] X_train = train.getFeatures();
double[] y_train = train.getTargets();

// Get specific samples
double[] sample = iris.getSample(0);
double target = iris.getTarget(0);

// Shuffle dataset
Dataset shuffled = iris.shuffle(42L);
```

### Confusion Matrix and ROC Curve

```java
import com.mindforge.validation.*;

// === Confusion Matrix ===
int[] yTrue = {0, 0, 1, 1, 2, 2, 0, 1, 2};
int[] yPred = {0, 0, 1, 2, 2, 1, 0, 1, 2};

ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred, 3);

// Overall metrics
System.out.println("Accuracy: " + cm.getAccuracy());
System.out.println("Macro Precision: " + cm.getMacroPrecision());
System.out.println("Macro Recall: " + cm.getMacroRecall());
System.out.println("Macro F1: " + cm.getMacroF1());

// Per-class metrics
double[] precision = cm.getPrecisionPerClass();
double[] recall = cm.getRecallPerClass();
double[] f1 = cm.getF1ScorePerClass();

// Get the confusion matrix
int[][] matrix = cm.getMatrix();

// Print classification report
System.out.println(cm.getReport());

// === ROC Curve and AUC ===
double[] yTrueBinary = {0, 0, 0, 1, 1, 1};
double[] yScores = {0.1, 0.2, 0.4, 0.6, 0.8, 0.9}; // Prediction probabilities

double auc = ROCCurve.calculateAUC(yTrueBinary, yScores);
System.out.println("AUC: " + auc);

ROCCurve roc = new ROCCurve(yTrueBinary, yScores);
double[] tpr = roc.getTPR();        // True Positive Rates
double[] fpr = roc.getFPR();        // False Positive Rates
double[] thresholds = roc.getThresholds();
double optimalThreshold = roc.getOptimalThreshold();
```

### Visualization

```java
import com.mindforge.visualization.ChartGenerator;

// Line chart
double[] x = {1, 2, 3, 4, 5};
double[] y = {2, 4, 6, 8, 10};
ChartGenerator.saveLineChart("line_chart.png", x, y, "Training Loss");

// Scatter plot
double[] xData = {1, 2, 3, 4, 5, 6, 7, 8};
double[] yData = {2, 4, 3, 5, 7, 6, 8, 9};
ChartGenerator.saveScatterChart("scatter.png", xData, yData, "Data Distribution");

// Bar chart
String[] categories = {"Class A", "Class B", "Class C"};
double[] values = {30, 45, 25};
ChartGenerator.saveBarChart("bar_chart.png", categories, values, "Class Distribution");

// Heatmap (e.g., confusion matrix)
double[][] heatmapData = {{10, 2, 1}, {3, 15, 2}, {1, 3, 12}};
ChartGenerator.saveHeatmap("heatmap.png", heatmapData, "Confusion Matrix");
```

### Logging and Configuration

```java
import com.mindforge.util.*;

// === Logging ===
MindForgeLogger.setLevel(MindForgeLogger.Level.DEBUG);
MindForgeLogger.setLogFile("mindforge.log");

MindForgeLogger.debug("Debug message");
MindForgeLogger.info("Training started with %d samples", 1000);
MindForgeLogger.warn("Learning rate might be too high");
MindForgeLogger.error("Failed to load model");

// === Configuration ===
// Load from properties file
Configuration config = new Configuration("config.properties");
String modelPath = config.getString("model.path", "/default/path");
double learningRate = config.getDouble("learning.rate", 0.01);
int epochs = config.getInt("epochs", 100);
boolean verbose = config.getBoolean("verbose", false);

// Load from YAML
Configuration yamlConfig = new Configuration("config.yml");
String dbHost = yamlConfig.getString("database.host", "localhost");

// Programmatic configuration
Configuration appConfig = new Configuration();
appConfig.set("model.name", "MyModel");
appConfig.set("batch.size", 32);
appConfig.save("app_config.properties");
```

### Array Utilities

```java
import com.mindforge.util.ArrayUtils;

// Matrix operations
double[][] matrix = {{1, 2, 3}, {4, 5, 6}};
double[][] transposed = ArrayUtils.transpose(matrix);

double[] a = {1, 2, 3};
double[] b = {4, 5, 6};
double dot = ArrayUtils.dot(a, b);  // 32.0

// Statistics
double mean = ArrayUtils.mean(a);
double sum = ArrayUtils.sum(a);
double std = ArrayUtils.std(a);
double min = ArrayUtils.min(a);
double max = ArrayUtils.max(a);
int argmax = ArrayUtils.argmax(a);

// Normalization
double[] normalized = ArrayUtils.normalize(a);      // Scale to [0, 1]
double[] standardized = ArrayUtils.standardize(a);  // Z-score normalization

// Array creation
double[] zeros = ArrayUtils.zeros(10);
double[] ones = ArrayUtils.ones(10);
double[] range = ArrayUtils.range(0, 10, 1);
double[] linspace = ArrayUtils.linspace(0, 1, 11);

// One-hot encoding
int[] labels = {0, 1, 2, 1, 0};
double[][] oneHot = ArrayUtils.oneHotEncode(labels, 3);

// Element-wise operations
double[] added = ArrayUtils.add(a, b);
double[] scaled = ArrayUtils.scale(a, 2.0);
```

### REST API Server

```java
import com.mindforge.api.*;
import com.mindforge.classification.KNearestNeighbors;

// Create and train a model
KNearestNeighbors knn = new KNearestNeighbors(3);
double[][] X_train = {{1, 2}, {2, 3}, {8, 8}, {9, 10}};
int[] y_train = {0, 0, 1, 1};
knn.train(X_train, y_train);

// Create model server
ModelServer server = new ModelServer(8080);

// Register model as a prediction endpoint
server.registerModel("/predict/knn", features -> {
    int prediction = knn.predict(features);
    return new double[]{prediction};
});

// Start server
server.start();
System.out.println("Model server running on http://localhost:8080");

// === Client usage ===
ModelClient client = new ModelClient("http://localhost:8080");
double[] features = {5.0, 5.0};
double[] prediction = client.predict("/predict/knn", features);
System.out.println("Prediction: " + prediction[0]);

// Stop server when done
server.stop();
```

## 🧪 Running Tests

```bash
mvn test
```

All tests should pass:
```
Tests run: 1406, Failures: 0, Errors: 0, Skipped: 2
BUILD SUCCESS
```

## 📊 Code Coverage

MindForge uses **JaCoCo** for code coverage analysis.

### Current Coverage (v1.2.0-alpha)

| Metric | Coverage |
|--------|----------|
| **Lines** | 96% |
| **Branches** | 87% |
| **Instructions** | 96% |

### Running Coverage Reports

```bash
# Run tests with coverage
mvn clean test

# Generate HTML report
mvn jacoco:report

# Verify coverage thresholds
mvn jacoco:check
```

The HTML report is generated at: `target/site/jacoco/index.html`

### Coverage Thresholds

The project enforces minimum coverage thresholds:
- **Line Coverage**: 70% minimum
- **Branch Coverage**: 60% minimum

## 🏗️ Building

Compile the project:
```bash
mvn compile
```

Package the project:
```bash
mvn package
```

## 📊 API Documentation

### Classification

#### KNearestNeighbors
```java
KNearestNeighbors(int k)                              // Constructor with k neighbors
KNearestNeighbors(int k, DistanceMetric metric)      // Constructor with custom distance metric
void train(double[][] X, int[] y)                     // Train the model
int predict(double[] x)                               // Predict single instance
int[] predict(double[][] X)                           // Predict multiple instances
```

#### DecisionTreeClassifier
```java
DecisionTreeClassifier()                              // Default constructor
DecisionTreeClassifier.Builder()                      // Builder for custom configuration
  .maxDepth(int depth)                                // Set maximum tree depth
  .minSamplesSplit(int samples)                       // Set minimum samples to split
  .minSamplesLeaf(int samples)                        // Set minimum samples per leaf
  .criterion(Criterion criterion)                     // Set splitting criterion (GINI or ENTROPY)
  .build()                                            // Build the classifier
void train(double[][] X, int[] y)                     // Train the model
int predict(double[] x)                               // Predict single instance
int[] predict(double[][] X)                           // Predict multiple instances
double[] predictProba(double[] x)                     // Get class probabilities for single instance
double[][] predictProba(double[][] X)                 // Get class probabilities for multiple instances
int getTreeDepth()                                    // Get actual tree depth
int getNumLeaves()                                    // Get number of leaf nodes
boolean isFitted()                                    // Check if model is trained
```

#### RandomForestClassifier
```java
RandomForestClassifier.Builder()                      // Builder for custom configuration
  .nEstimators(int n)                                 // Set number of trees (default: 100)
  .maxFeatures(String mode)                           // Set max features: "sqrt" or "log2"
  .maxFeatures(int n)                                 // Set specific number of features
  .maxDepth(int depth)                                // Set maximum tree depth
  .minSamplesSplit(int samples)                       // Set minimum samples to split
  .minSamplesLeaf(int samples)                        // Set minimum samples per leaf
  .criterion(Criterion criterion)                     // Set splitting criterion (GINI or ENTROPY)
  .bootstrap(boolean use)                             // Enable/disable bootstrap sampling (default: true)
  .randomState(int seed)                              // Set random seed for reproducibility
  .build()                                            // Build the classifier
void fit(double[][] X, int[] y)                       // Train the model
int[] predict(double[][] X)                           // Predict multiple instances
double[][] predictProba(double[][] X)                 // Get class probabilities for multiple instances
double getOOBScore()                                  // Get out-of-bag score
double[] getFeatureImportance()                       // Get feature importance scores
int getNEstimators()                                  // Get number of trees
int[] getClasses()                                    // Get unique class labels
```

### Regression

#### LinearRegression
```java
LinearRegression()                                    // Constructor
void train(double[][] X, double[] y)                  // Train the model
double predict(double[] x)                            // Predict single instance
double[] predict(double[][] X)                        // Predict multiple instances
double[] getWeights()                                 // Get learned weights
double getBias()                                      // Get learned bias
boolean isFitted()                                    // Check if model is trained
```

### Clustering

#### KMeans
```java
KMeans(int k)                                         // Constructor with k clusters
KMeans(int k, InitStrategy strategy)                  // Constructor with initialization strategy
void fit(double[][] X, int k)                         // Fit the model
int predict(double[] x)                               // Predict cluster for single point
double[][] getCentroids()                             // Get cluster centroids
double getInertia()                                   // Get within-cluster sum of squares
```

### Preprocessing

#### MinMaxScaler
```java
MinMaxScaler()                                        // Scale to [0, 1]
MinMaxScaler(double min, double max)                  // Scale to custom range
void fit(double[][] X)                                // Learn min/max from data
double[][] transform(double[][] X)                    // Apply scaling
double[][] fitTransform(double[][] X)                 // Fit and transform
double[][] inverseTransform(double[][] X)             // Reverse scaling
```

#### StandardScaler
```java
StandardScaler()                                      // Constructor
StandardScaler(boolean withMean, boolean withStd)     // Constructor with options
void fit(double[][] X)                                // Learn mean and std
double[][] transform(double[][] X)                    // Apply standardization
double[][] fitTransform(double[][] X)                 // Fit and transform
double[][] inverseTransform(double[][] X)             // Reverse standardization
```

#### SimpleImputer
```java
SimpleImputer(Strategy strategy)                      // MEAN, MEDIAN, MOST_FREQUENT, CONSTANT
void fit(double[][] X)                                // Learn imputation values
double[][] transform(double[][] X)                    // Apply imputation
double[][] fitTransform(double[][] X)                 // Fit and transform
void setFillValue(double value)                       // Set constant fill value
```

#### LabelEncoder
```java
LabelEncoder()                                        // Constructor
int[] encode(String[] labels)                         // Encode string labels to integers
String[] decode(int[] encodedLabels)                  // Decode integers back to strings
int[] fitTransform(String[] labels)                   // Fit and transform
String[] inverseTransform(int[] encodedLabels)        // Inverse transform
```

#### DataSplit
```java
static Split trainTestSplit(double[][] X, int[] y, double testSize, Integer randomState)
static Split trainTestSplit(double[][] X, double[] y, double testSize, Integer randomState)
static Split stratifiedSplit(double[][] X, int[] y, double testSize, Integer randomState)
// Split class contains: XTrain, XTest, yTrain, yTest (int[] or double[])
```

### Metrics

#### Classification Metrics
```java
double accuracy(int[] actual, int[] predicted)
double precision(int[] actual, int[] predicted, int positiveClass)
double recall(int[] actual, int[] predicted, int positiveClass)
double f1Score(int[] actual, int[] predicted, int positiveClass)
int[][] confusionMatrix(int[] actual, int[] predicted)
```

#### Regression Metrics
```java
double mse(double[] actual, double[] predicted)       // Mean Squared Error
double rmse(double[] actual, double[] predicted)      // Root Mean Squared Error
double mae(double[] actual, double[] predicted)       // Mean Absolute Error
double r2Score(double[] actual, double[] predicted)   // R² Score
```

### Distance Functions
```java
double euclidean(double[] a, double[] b)              // Euclidean distance
double manhattan(double[] a, double[] b)              // Manhattan distance
double chebyshev(double[] a, double[] b)              // Chebyshev distance
double minkowski(double[] a, double[] b, double p)    // Minkowski distance
```

## 🛣️ Roadmap

### Completed
- [x] Decision Trees
- [x] Logistic Regression
- [x] Naive Bayes (Gaussian, Multinomial, Bernoulli)
- [x] Data preprocessing utilities (MinMaxScaler, StandardScaler, SimpleImputer, LabelEncoder)
- [x] Train/Test split functionality (with stratified split support)
- [x] Random Forest
- [x] Cross-validation (K-Fold, Stratified K-Fold, LOOCV, Shuffle Split)
- [x] Support Vector Machines (Linear SVM)
- [x] Gradient Boosting
- [x] Feature Selection (VarianceThreshold, SelectKBest, RFE)
- [x] PCA (Principal Component Analysis)
- [x] Model Persistence (Save/Load)
- [x] Neural Networks (MLP with backpropagation)
- [x] Advanced Metrics (Confusion Matrix, ROC Curve, AUC)
- [x] Dataset Loaders (Iris, Wine, Breast Cancer, Boston Housing)
- [x] Logging System
- [x] Configuration Management (YAML, Properties)
- [x] Array Utilities
- [x] Visualization (Chart generation)
- [x] REST API Server for model serving
- [x] **Ridge, Lasso, ElasticNet Regression** (v1.2.0)
- [x] **Polynomial Regression** (v1.2.0)
- [x] **Support Vector Regression (SVR)** with RBF, Polynomial, Sigmoid kernels (v1.2.0)
- [x] **Linear Discriminant Analysis (LDA)** (v1.2.0)
- [x] **RNN/LSTM** Recurrent Neural Networks (v1.2.0)
- [x] **GPU/CPU Acceleration** (v1.2.0)

### In Progress
- [ ] Deep Learning support (CNN)
- [ ] Advanced ensemble methods (AdaBoost, XGBoost)
- [ ] Transformer architecture

## 📄 Project Information

- **Group ID**: com.mindforge
- **Artifact ID**: mindforge
- **Version**: 1.2.0-alpha
- **Java Version**: 11

## 📚 Main Dependencies

- **Apache Commons Math 3.6.1** - Mathematical and statistical functions
- **ND4J 1.0.0-M2.1** - Numerical computing
- **JUnit 5.10.1** - Testing framework
- **SLF4J 2.0.9** - Logging facade
- **JaCoCo 0.8.11** - Code coverage analysis

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code of Conduct
- Development Setup
- Code Style Guidelines
- Testing Requirements
- Pull Request Process

### Quick Start for Contributors

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Write tests and ensure all tests pass (`mvn test`)
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

### Coding Standards

- Follow Java naming conventions
- Add unit tests for new features (minimum 80% coverage)
- Document public APIs with Javadoc
- Run `mvn verify` before submitting PRs

## 📝 License

TBD - License information will be added soon.

## 🙏 Acknowledgments

- Inspired by [Smile](https://haifengl.github.io/) (Statistical Machine Intelligence and Learning Engine)
- Built with ❤️ by the MindForge team

## 📧 Contact

For questions, suggestions, or feedback, please open an issue on GitHub.

---

**Author**: MindForge Team  
**Repository**: [https://github.com/yasmramos/MindForge](https://github.com/yasmramos/MindForge)  
**Examples**: [View all examples](examples/README.md)  
**Contributing**: [Contribution guidelines](CONTRIBUTING.md)
