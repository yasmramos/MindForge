# MindForge

A Machine Learning and Artificial Intelligence library for Java, inspired by libraries like Smile, designed to be easy to use and efficient.

[![Java Version](https://img.shields.io/badge/Java-17%2B-blue)](https://www.oracle.com/java/)
[![Maven](https://img.shields.io/badge/Maven-3.6%2B-red)](https://maven.apache.org/)
[![License](https://img.shields.io/badge/License-TBD-yellow)](LICENSE)

## 🚀 Features

- **Classification Algorithms**: K-Nearest Neighbors (KNN), and more coming soon
- **Regression Algorithms**: Linear Regression, and more coming soon
- **Clustering Algorithms**: K-Means, and more coming soon
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, MSE, RMSE, MAE, R²
- **Distance Functions**: Euclidean, Manhattan, Chebyshev, Minkowski
- **Simple and Consistent Interface**: Intuitive APIs for all algorithms

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
│   ├── math/              # Mathematical functions
│   │   └── Distance.java
│   ├── validation/        # Evaluation metrics
│   │   └── Metrics.java
│   ├── neural/            # Neural networks (coming soon)
│   ├── data/              # Data processing (coming soon)
│   └── util/              # Utilities (coming soon)
└── pom.xml
```

## 🔧 Requirements

- **Java 17** or higher
- **Maven 3.6** or higher

## 📥 Installation

Clone the repository and build the project:

```bash
git clone https://github.com/yasmramos/MindForge.git
cd MindForge
mvn clean install
```

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

## 🧪 Running Tests

```bash
mvn test
```

All tests should pass:
```
Tests run: 27, Failures: 0, Errors: 0, Skipped: 0
BUILD SUCCESS
```

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

### Short Term
- [ ] Decision Trees
- [ ] Logistic Regression
- [ ] Naive Bayes
- [ ] Data preprocessing utilities
- [ ] Train/Test split functionality

### Medium Term
- [ ] Random Forest
- [ ] Support Vector Machines (SVM)
- [ ] Gradient Boosting
- [ ] Cross-validation
- [ ] Feature selection

### Long Term
- [ ] Neural Networks (MLP)
- [ ] Deep Learning support
- [ ] PCA (Principal Component Analysis)
- [ ] Advanced ensemble methods
- [ ] GPU acceleration

## 📄 Project Information

- **Group ID**: com.mindforge
- **Artifact ID**: mindforge
- **Version**: 1.0-SNAPSHOT
- **Java Version**: 17

## 📚 Main Dependencies

- **Apache Commons Math 3.6.1** - Mathematical and statistical functions
- **ND4J 1.0.0-M2.1** - Numerical computing
- **JUnit 5.10.1** - Testing framework
- **SLF4J 2.0.9** - Logging facade

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Standards

- Follow Java naming conventions
- Add unit tests for new features
- Maintain code coverage above 80%
- Document public APIs with Javadoc

## 📝 License

TBD - License information will be added soon.

## 🙏 Acknowledgments

- Inspired by [Smile](https://haifengl.github.io/) (Statistical Machine Intelligence and Learning Engine)
- Built with ❤️ by the MindForge team

## 📧 Contact

For questions, suggestions, or feedback, please open an issue on GitHub.

---

**Author**: Matrix Agent  
**Repository**: [https://github.com/yasmramos/MindForge](https://github.com/yasmramos/MindForge)
