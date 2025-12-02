# MindForge Examples

This directory contains practical examples demonstrating how to use the MindForge Machine Learning library.

## Quick Start

```bash
# From the MindForge root directory, build and install the library first
mvn clean install -DskipTests

# Then run examples
cd examples
mvn compile exec:java -Dexec.mainClass="com.mindforge.examples.QuickStart"
```

## Available Examples

### 1. QuickStart.java
A beginner-friendly introduction to MindForge covering the basic ML workflow:
- Data preparation (Iris-like dataset)
- Train/test splitting with DataSplit
- Feature scaling with StandardScaler
- Training multiple classifiers (KNN, Decision Tree, Naive Bayes)
- Making predictions and evaluating accuracy

```bash
mvn exec:java -Dexec.mainClass="com.mindforge.examples.QuickStart"
```

### 2. ClusteringExample.java
K-Means clustering demonstration:
- Synthetic data generation with multiple clusters
- Comparing different k values (elbow method)
- Computing inertia for cluster evaluation
- Practical customer segmentation example

```bash
mvn exec:java -Dexec.mainClass="com.mindforge.examples.ClusteringExample"
```

### 3. RegressionExample.java
Regression algorithms demonstration:
- Linear Regression for continuous prediction
- Ridge Regression with L2 regularization
- Evaluation metrics (MSE, RMSE, MAE, R²)
- Practical house price prediction example

```bash
mvn exec:java -Dexec.mainClass="com.mindforge.examples.RegressionExample"
```

### 4. PreprocessingExample.java
Data preprocessing techniques:
- StandardScaler (z-score normalization)
- MinMaxScaler (range scaling to [0,1] or custom range)
- LabelEncoder (categorical to numeric conversion)
- Practical employee data preprocessing example

```bash
mvn exec:java -Dexec.mainClass="com.mindforge.examples.PreprocessingExample"
```

### 5. PipelineExample.java
ML Pipelines demonstration:
- Chaining transformers and classifiers
- Comparing different model pipelines
- Pipeline benefits and best practices
- Train/test workflow management

```bash
mvn exec:java -Dexec.mainClass="com.mindforge.examples.PipelineExample"
```

### 6. ValidationExample.java
Model validation techniques:
- K-Fold Cross Validation
- Model comparison using CV
- Classification metrics (Accuracy, Precision, Recall, F1)
- Confusion matrix breakdown

```bash
mvn exec:java -Dexec.mainClass="com.mindforge.examples.ValidationExample"
```

## Project Structure

```
examples/
├── pom.xml                           # Maven configuration
├── README.md                         # This file
└── src/main/java/com/mindforge/examples/
    ├── QuickStart.java               # Basic ML workflow tutorial
    ├── ClusteringExample.java        # K-Means clustering example
    ├── RegressionExample.java        # Linear & Ridge regression
    ├── PreprocessingExample.java     # Data preprocessing techniques
    ├── PipelineExample.java          # ML pipelines
    └── ValidationExample.java        # Cross-validation & metrics
```

## Requirements

- Java 11 or higher
- Maven 3.6+
- MindForge library (installed via `mvn install` from parent project)

## Building and Running

```bash
# From MindForge root directory
# 1. Build and install the main library
mvn clean install -DskipTests

# 2. Run examples
cd examples
mvn compile

# 3. Run all examples
mvn exec:java -Dexec.mainClass="com.mindforge.examples.QuickStart"
mvn exec:java -Dexec.mainClass="com.mindforge.examples.ClusteringExample"
mvn exec:java -Dexec.mainClass="com.mindforge.examples.RegressionExample"
mvn exec:java -Dexec.mainClass="com.mindforge.examples.PreprocessingExample"
mvn exec:java -Dexec.mainClass="com.mindforge.examples.PipelineExample"
mvn exec:java -Dexec.mainClass="com.mindforge.examples.ValidationExample"
```

## Using MindForge in Your Own Project

Download the JAR from the [releases page](https://github.com/yasmramos/MindForge/releases) and add it to your classpath, or add it to your local Maven repository:

```bash
mvn install:install-file -Dfile=mindforge-1.1.0-alpha.jar \
    -DgroupId=com.mindforge -DartifactId=mindforge \
    -Dversion=1.1.0-alpha -Dpackaging=jar
```

Then add to your `pom.xml`:
```xml
<dependency>
    <groupId>com.mindforge</groupId>
    <artifactId>mindforge</artifactId>
    <version>1.1.0-alpha</version>
</dependency>
```

## Common Patterns

### Basic Classification
```java
// 1. Prepare data
double[][] X = /* your features */;
int[] y = /* your labels */;

// 2. Split data (test_size=0.2, shuffle=true, seed=42)
DataSplit.TrainTestSplit split = DataSplit.trainTestSplit(X, y, 0.2, true, 42);

// 3. Scale features
StandardScaler scaler = new StandardScaler();
scaler.fit(split.XTrain);
double[][] XTrain = scaler.transform(split.XTrain);
double[][] XTest = scaler.transform(split.XTest);

// 4. Train model
KNearestNeighbors knn = new KNearestNeighbors(5);
knn.train(XTrain, split.yTrain);

// 5. Predict
int[] predictions = new int[XTest.length];
for (int i = 0; i < XTest.length; i++) {
    predictions[i] = knn.predict(XTest[i]);
}
double accuracy = Metrics.accuracy(split.yTest, predictions);
```

### Clustering
```java
// K-Means with 3 clusters
KMeans kmeans = new KMeans(3, 100, new Random(42));
int[] labels = kmeans.cluster(X);
double[][] centroids = kmeans.getCentroids();
```

### Regression
```java
LinearRegression lr = new LinearRegression();
lr.train(XTrain, yTrain);
double prediction = lr.predict(xNew);
double r2 = Metrics.r2Score(yTest, predictions);
```

### Cross-Validation
```java
CrossValidationResult result = CrossValidation.kFold(
    (xTrain, yTrain) -> {
        KNearestNeighbors knn = new KNearestNeighbors(5);
        knn.train(xTrain, yTrain);
        return knn;
    },
    (model, xTest) -> {
        int[] preds = new int[xTest.length];
        for (int i = 0; i < xTest.length; i++) {
            preds[i] = model.predict(xTest[i]);
        }
        return preds;
    },
    X, y, 5, 42  // 5-fold CV with seed 42
);
System.out.println("Mean Accuracy: " + result.getMean());
```

## MindForge Features

**Classification:**
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- SVM (SVC)
- AdaBoost
- Ensemble methods (Voting, Bagging, Stacking)

**Regression:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Polynomial Regression

**Clustering:**
- K-Means
- DBSCAN
- Hierarchical Clustering
- Mean Shift

**Preprocessing:**
- StandardScaler
- MinMaxScaler
- LabelEncoder
- OneHotEncoder
- SimpleImputer
- PolynomialFeatures

**Dimensionality Reduction:**
- PCA

**Feature Selection:**
- SelectKBest
- VarianceThreshold
- RFE (Recursive Feature Elimination)

**Pipelines:**
- Pipeline (chaining transformers and estimators)
- ColumnTransformer
- GridSearchCV

**Validation:**
- Train/Test Split
- K-Fold Cross-Validation
- Stratified K-Fold
- Leave-One-Out
- Metrics (Accuracy, Precision, Recall, F1, MSE, RMSE, MAE, R²)

## License

MIT License - see the main project for details.
