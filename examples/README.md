# MindForge Examples

This directory contains practical examples demonstrating how to use the MindForge Machine Learning library.

## Quick Start

```bash
# From the MindForge root directory, build the library first
mvn clean package -DskipTests

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
- Synthetic data generation
- Comparing different k values
- Inertia (within-cluster sum of squares)
- Practical customer segmentation example

```bash
mvn exec:java -Dexec.mainClass="com.mindforge.examples.ClusteringExample"
```

## Project Structure

```
examples/
├── pom.xml                           # Maven configuration
├── README.md                         # This file
└── src/main/java/com/mindforge/examples/
    ├── QuickStart.java               # Basic ML workflow tutorial
    └── ClusteringExample.java        # K-Means clustering example
```

## Requirements

- Java 11 or higher
- Maven 3.6+
- MindForge library (built from parent project)

## Building and Running

```bash
# From MindForge root directory
# 1. Build the main library
mvn clean package -DskipTests

# 2. Run examples
cd examples
mvn compile

# 3. Run specific example
mvn exec:java -Dexec.mainClass="com.mindforge.examples.QuickStart"
mvn exec:java -Dexec.mainClass="com.mindforge.examples.ClusteringExample"
```

## Using MindForge in Your Own Project

Download the JAR from the [releases page](https://github.com/yasmramos/MindForge/releases) and add it to your classpath.

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

// 5. Predict and evaluate
int[] predictions = knn.predict(XTest);
double accuracy = Metrics.accuracy(split.yTest, predictions);
```

### Clustering
```java
// K-Means with 3 clusters
KMeans kmeans = new KMeans(3, 100, new Random(42));
int[] labels = kmeans.fitPredict(X);
double[][] centroids = kmeans.getCentroids();
double inertia = kmeans.getInertia();
```

## MindForge Features

**Classification:**
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- SVM (SVC)
- Logistic Regression
- AdaBoost
- Ensemble methods (Voting, Bagging, Stacking)

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
- PCA

**Validation:**
- Train/Test Split
- Cross-Validation
- Metrics (Accuracy, Precision, Recall, F1, MSE, RMSE, R2)

## License

MIT License - see the main project for details.
