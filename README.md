# MindForge

A Machine Learning and Artificial Intelligence library for Java, inspired by libraries like Smile, designed to be easy to use and efficient.

[![Java Version](https://img.shields.io/badge/Java-17%2B-blue)](https://www.oracle.com/java/)
[![Maven](https://img.shields.io/badge/Maven-3.6%2B-red)](https://maven.apache.org/)
[![License](https://img.shields.io/badge/License-TBD-yellow)](LICENSE)

## 🚀 Features

- **Classification Algorithms**: K-Nearest Neighbors (KNN), Decision Trees, and more coming soon
- **Regression Algorithms**: Linear Regression, and more coming soon
- **Clustering Algorithms**: K-Means, and more coming soon
- **Data Preprocessing**: MinMaxScaler, StandardScaler, SimpleImputer, LabelEncoder, DataSplit
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
│   ├── preprocessing/     # Data preprocessing
│   │   ├── MinMaxScaler.java
│   │   ├── StandardScaler.java
│   │   ├── SimpleImputer.java
│   │   ├── LabelEncoder.java
│   │   └── DataSplit.java
│   ├── math/              # Mathematical functions
│   │   └── Distance.java
│   ├── validation/        # Evaluation metrics
│   │   └── Metrics.java
│   ├── neural/            # Neural networks (coming soon)
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

## 🧪 Running Tests

```bash
mvn test
```

All tests should pass:
```
Tests run: 96, Failures: 0, Errors: 0, Skipped: 0
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

### Short Term
- [x] Decision Trees
- [ ] Logistic Regression
- [ ] Naive Bayes
- [x] Data preprocessing utilities (MinMaxScaler, StandardScaler, SimpleImputer, LabelEncoder)
- [x] Train/Test split functionality (with stratified split support)

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
- **Version**: 1.0.2-alpha
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

**Author**: MindForge Team  
**Repository**: [https://github.com/yasmramos/MindForge](https://github.com/yasmramos/MindForge)
