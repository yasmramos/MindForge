# MindForge

A Machine Learning and Artificial Intelligence library for Java, inspired by libraries like Smile, designed to be easy to use and efficient.

[![Java Version](https://img.shields.io/badge/Java-17%2B-blue)](https://www.oracle.com/java/)
[![Maven](https://img.shields.io/badge/Maven-3.6%2B-red)](https://maven.apache.org/)
[![License](https://img.shields.io/badge/License-TBD-yellow)](LICENSE)

## 🚀 Features

- **Classification Algorithms**: K-Nearest Neighbors (KNN), Decision Trees, Random Forest, Logistic Regression, Naive Bayes (Gaussian, Multinomial, Bernoulli), Support Vector Machines (SVM), Gradient Boosting
- **Regression Algorithms**: Linear Regression, and more coming soon
- **Clustering Algorithms**: K-Means, and more coming soon
- **Data Preprocessing**: MinMaxScaler, StandardScaler, SimpleImputer, LabelEncoder, DataSplit
- **Model Evaluation**: Cross-Validation (K-Fold, Stratified K-Fold, LOOCV, Shuffle Split), Train-Test Split
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
Tests run: 285, Failures: 0, Errors: 0, Skipped: 0
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

### Short Term
- [x] Decision Trees
- [x] Logistic Regression
- [ ] Naive Bayes
- [x] Data preprocessing utilities (MinMaxScaler, StandardScaler, SimpleImputer, LabelEncoder)
- [x] Train/Test split functionality (with stratified split support)

### Medium Term
- [x] Random Forest
- [x] Logistic Regression
- [x] Cross-validation (K-Fold, Stratified K-Fold, LOOCV, Shuffle Split)
- [x] Naive Bayes (Gaussian, Multinomial, Bernoulli)
- [x] Support Vector Machines (Linear SVM)
- [x] Gradient Boosting
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
- **Version**: 1.0.7-alpha
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
