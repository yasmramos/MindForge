# MindForge Quick Start Guide

Get started with MindForge in 5 minutes!

## Installation

### Maven

Add to your `pom.xml`:

```xml
<dependency>
    <groupId>io.github.yasmramos</groupId>
    <artifactId>mindforge</artifactId>
    <version>1.2.2</version>
</dependency>
```

### Gradle

```groovy
implementation 'io.github.yasmramos:mindforge:1.2.2'
```

## Basic Examples

### 1. Classification with Random Forest

```java
import com.mindforge.classification.RandomForestClassifier;
import com.mindforge.validation.metrics.Metrics;

public class ClassificationExample {
    public static void main(String[] args) {
        // Training data (Iris dataset example)
        double[][] X = {
            {5.1, 3.5, 1.4, 0.2},
            {4.9, 3.0, 1.4, 0.2},
            {7.0, 3.2, 4.7, 1.4},
            {6.4, 3.2, 4.5, 1.5},
            {6.3, 3.3, 6.0, 2.5},
            {5.8, 2.7, 5.1, 1.9}
        };
        int[] y = {0, 0, 1, 1, 2, 2};
        
        // Create and train model
        RandomForestClassifier rf = new RandomForestClassifier(100, 10);
        rf.fit(X, y);
        
        // Make predictions
        double[] sample = {5.0, 3.4, 1.5, 0.2};
        int prediction = rf.predict(new double[][]{sample})[0];
        System.out.println("Predicted class: " + prediction);
        
        // Evaluate accuracy
        int[] predictions = rf.predict(X);
        double accuracy = Metrics.accuracy(y, predictions);
        System.out.println("Accuracy: " + accuracy);
    }
}
```

### 2. Regression with Linear Regression

```java
import com.mindforge.regression.LinearRegression;
import com.mindforge.validation.metrics.Metrics;

public class RegressionExample {
    public static void main(String[] args) {
        // Training data
        double[][] X = {
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
        };
        double[] y = {2.1, 4.0, 6.2, 8.1, 9.8};
        
        // Create and train model
        LinearRegression lr = new LinearRegression();
        lr.fit(X, y);
        
        // Make predictions
        double[] sample = {6.0};
        double prediction = lr.predict(new double[][]{sample})[0];
        System.out.println("Predicted value: " + prediction);
        
        // Evaluate R² score
        double[] predictions = lr.predict(X);
        double r2 = Metrics.r2Score(y, predictions);
        System.out.println("R² score: " + r2);
    }
}
```

### 3. Clustering with K-Means

```java
import com.mindforge.clustering.KMeans;

public class ClusteringExample {
    public static void main(String[] args) {
        // Data points
        double[][] X = {
            {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0},
            {8.0, 8.0}, {1.0, 0.6}, {9.0, 11.0}
        };
        
        // Create and fit model
        KMeans km = new KMeans(3, 100);
        km.fit(X);
        
        // Get cluster assignments
        int[] clusters = km.getLabels();
        for (int i = 0; i < clusters.length; i++) {
            System.out.println("Point " + i + " -> Cluster " + clusters[i]);
        }
        
        // Predict new point
        double[] sample = {1.2, 1.5};
        int cluster = km.predict(new double[][]{sample})[0];
        System.out.println("New point belongs to cluster: " + cluster);
    }
}
```

### 4. Neural Network (MLP)

```java
import com.mindforge.neural.networks.MLPClassifier;

public class NeuralNetworkExample {
    public static void main(String[] args) {
        // XOR problem
        double[][] X = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        int[] y = {0, 1, 1, 0};
        
        // Create MLP: 2 inputs -> 4 hidden -> 2 outputs
        MLPClassifier mlp = new MLPClassifier(2, 4, 2);
        mlp.setLearningRate(0.1);
        mlp.setEpochs(1000);
        
        // Train
        mlp.fit(X, y);
        
        // Predict
        double[] sample = {1, 0};
        int prediction = mlp.predict(new double[][]{sample})[0];
        System.out.println("XOR(1, 0) = " + prediction);
    }
}
```

### 5. Model Interpretability with SHAP

```java
import com.mindforge.classification.RandomForestClassifier;
import com.mindforge.interpret.shap.TreeSHAP;

public class SHAPExample {
    public static void main(String[] args) {
        // Train model
        double[][] X = ...; // Your training data
        int[] y = ...;      // Your labels
        
        RandomForestClassifier rf = new RandomForestClassifier(100, 10);
        rf.fit(X, y);
        
        // Create TreeSHAP explainer
        TreeSHAP shap = new TreeSHAP(rf, X);
        
        // Explain a prediction
        double[] sample = X[0];
        double[] shapValues = shap.explainInstance(sample);
        
        System.out.println("Feature importance (SHAP values):");
        for (int i = 0; i < shapValues.length; i++) {
            System.out.println("Feature " + i + ": " + shapValues[i]);
        }
    }
}
```

### 6. Preprocessing Pipeline

```java
import com.mindforge.preprocessing.StandardScaler;
import com.mindforge.preprocessing.DataSplit;
import com.mindforge.classification.RandomForestClassifier;

public class PipelineExample {
    public static void main(String[] args) {
        double[][] X = ...; // Your data
        int[] y = ...;      // Your labels
        
        // Split data
        DataSplit split = new DataSplit(0.8, 42);
        double[][] X_train = split.getTrainFeatures(X);
        double[][] X_test = split.getTestFeatures(X);
        int[] y_train = split.getTrainLabels(y);
        int[] y_test = split.getTestLabels(y);
        
        // Scale features
        StandardScaler scaler = new StandardScaler();
        X_train = scaler.fitTransform(X_train);
        X_test = scaler.transform(X_test);
        
        // Train model
        RandomForestClassifier rf = new RandomForestClassifier(100, 10);
        rf.fit(X_train, y_train);
        
        // Evaluate
        int[] predictions = rf.predict(X_test);
        double accuracy = Metrics.accuracy(y_test, predictions);
        System.out.println("Test accuracy: " + accuracy);
    }
}
```

## Advanced Features

### Hyperparameter Tuning with GridSearchCV

```java
import com.mindforge.model_selection.GridSearchCV;
import com.mindforge.classification.RandomForestClassifier;

Map<String, Object[]> paramGrid = new HashMap<>();
paramGrid.put("n_estimators", new Object[]{50, 100, 200});
paramGrid.put("max_depth", new Object[]{10, 20, null});

GridSearchCV gridSearch = new GridSearchCV(
    new RandomForestClassifier(),
    paramGrid,
    5  // 5-fold CV
);

gridSearch.fit(X, y);

System.out.println("Best params: " + gridSearch.getBestParams());
System.out.println("Best score: " + gridSearch.getBestScore());
```

### Model Persistence

```java
import com.mindforge.persistence.ModelPersistence;
import com.mindforge.classification.RandomForestClassifier;

// Save model
RandomForestClassifier rf = ...;
ModelPersistence.save(rf, "model.ser");

// Load model
RandomForestClassifier loaded = 
    (RandomForestClassifier) ModelPersistence.load("model.ser");
```

### REST API Server

```java
import com.mindforge.api.ModelServer;
import com.mindforge.classification.RandomForestClassifier;

// Train model
RandomForestClassifier rf = ...;

// Start server
ModelServer server = new ModelServer(rf, 8080);
server.start();

// Access at: http://localhost:8080/predict
```

## Next Steps

- 📚 Read full documentation: [README.md](README.md)
- 📊 View benchmarks: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)
- 🔧 Learn about deployment: [DEPLOYMENT.md](DEPLOYMENT.md)
- 💻 Explore examples: `examples/` directory

## Support

- GitHub Issues: https://github.com/yasmramos/MindForge/issues
- Email: yasmramos95@gmail.com

Happy coding with MindForge! 🚀
