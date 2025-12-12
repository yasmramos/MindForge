# Data Preprocessing Package

The `io.github.yasmramos.mindforge.preprocessing` package provides essential tools to prepare data before training machine learning models.

## 📦 Implemented Classes

### 1. **MinMaxScaler**
Scales features to a specific range (default: [0, 1]).

**Usage:**
```java
double[][] data = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}};
MinMaxScaler scaler = new MinMaxScaler(); // Range [0, 1]
// Or with custom range:
// MinMaxScaler scaler = new MinMaxScaler(-1.0, 1.0);

double[][] scaled = scaler.fitTransform(data);
// For new data:
double[][] testScaled = scaler.transform(testData);
// Inverse transformation:
double[][] original = scaler.inverseTransform(scaled);
```

**Methods:**
- `fit(double[][] X)` - Computes min and max of each feature
- `transform(double[][] X)` - Scales the data
- `fitTransform(double[][] X)` - Fit and transform in one step
- `inverseTransform(double[][] X)` - Reverses the scaling
- `getFeatureMin()` - Gets minimum values
- `getFeatureMax()` - Gets maximum values

---

### 2. **StandardScaler**
Standardizes features by removing the mean and scaling to unit variance (Z-score normalization).

**Usage:**
```java
double[][] data = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}};
StandardScaler scaler = new StandardScaler();
// Or with options:
// StandardScaler scaler = new StandardScaler(true, true);  // withMean, withStd

double[][] scaled = scaler.fitTransform(data);
double[][] original = scaler.inverseTransform(scaled);
```

**Methods:**
- `fit(double[][] X)` - Computes mean and standard deviation
- `transform(double[][] X)` - Standardizes the data
- `fitTransform(double[][] X)` - Fit and transform in one step
- `inverseTransform(double[][] X)` - Reverses the standardization
- `getMean()` - Gets the means
- `getStd()` - Gets the standard deviations

---

### 3. **DataSplit**
Splits datasets into train and test sets.

**Usage - Classification:**
```java
double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}};
int[] y = {0, 0, 1, 1};

// Simple split
DataSplit.TrainTestSplit split = DataSplit.trainTestSplit(
    X, y, 
    0.25,      // 25% for test
    true,      // shuffle
    42         // random seed (null for random)
);

// Access the data
double[][] XTrain = split.XTrain;
double[][] XTest = split.XTest;
int[] yTrain = split.yTrain;
int[] yTest = split.yTest;
```

**Usage - Regression:**
```java
double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}};
double[] y = {1.5, 2.5, 3.5, 4.5};

DataSplit.TrainTestSplitRegression split = DataSplit.trainTestSplit(
    X, y, 0.25, true, 42
);

double[] yTrain = split.yTrain;  // Type double for regression
double[] yTest = split.yTest;
```

**Usage - Stratified Split:**
```java
// Maintains class proportion in train and test
DataSplit.TrainTestSplit split = DataSplit.stratifiedTrainTestSplit(
    X, y,
    0.25,   // test size
    42      // random seed
);
```

**Methods:**
- `trainTestSplit(X, y, testSize, shuffle, seed)` - Simple split (classification)
- `trainTestSplit(X, y, testSize, shuffle, seed)` - Simple split (regression)
- `stratifiedTrainTestSplit(X, y, testSize, seed)` - Stratified split

---

### 4. **SimpleImputer**
Fills missing values (NaN) in datasets.

**Available strategies:**
- `MEAN` - Replaces with the mean
- `MEDIAN` - Replaces with the median
- `MOST_FREQUENT` - Replaces with the most frequent value
- `CONSTANT` - Replaces with a constant value

**Usage:**
```java
double[][] data = {
    {1.0, 2.0},
    {Double.NaN, 3.0},
    {7.0, Double.NaN},
    {4.0, 6.0}
};

// Mean imputation
SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.MEAN);
double[][] filled = imputer.fitTransform(data);

// Constant value imputation
SimpleImputer imputer2 = new SimpleImputer(
    SimpleImputer.ImputeStrategy.CONSTANT, 
    0.0  // constant value
);
double[][] filled2 = imputer2.fitTransform(data);
```

**Methods:**
- `fit(double[][] X)` - Computes imputation statistics
- `transform(double[][] X)` - Fills missing values
- `fitTransform(double[][] X)` - Fit and transform in one step
- `getStatistics()` - Gets the computed imputation values

---

### 5. **LabelEncoder**
Encodes categorical labels to numeric values.

**Usage:**
```java
String[] labels = {"cat", "dog", "cat", "bird", "dog"};

LabelEncoder encoder = new LabelEncoder();
int[] encoded = encoder.fitTransform(labels);
// Result: [0, 1, 0, 2, 1]

// Decode back
String[] decoded = encoder.inverseTransform(encoded);
// Result: ["cat", "dog", "cat", "bird", "dog"]

// Encode individual values
int catCode = encoder.encode("cat");        // 0
String label = encoder.decode(1);           // "dog"

// Get information
String[] classes = encoder.getClasses();    // ["cat", "dog", "bird"]
int numClasses = encoder.getNumClasses();   // 3
```

**Methods:**
- `fit(String[] labels)` - Learns the label mapping
- `transform(String[] labels)` - Encodes labels to integers
- `fitTransform(String[] labels)` - Fit and transform in one step
- `inverseTransform(int[] encoded)` - Decodes to original labels
- `encode(String label)` - Encodes a single label
- `decode(int encoded)` - Decodes a single value
- `getClasses()` - Gets all unique classes
- `getNumClasses()` - Gets the number of classes

---

## 🔄 Typical Workflow

### For Classification:

```java
// 1. Load data
double[][] X = loadData();
int[] y = loadLabels();

// 2. Impute missing values (if any)
SimpleImputer imputer = new SimpleImputer(ImputeStrategy.MEAN);
X = imputer.fitTransform(X);

// 3. Split into train/test
DataSplit.TrainTestSplit split = DataSplit.stratifiedTrainTestSplit(X, y, 0.2, 42);

// 4. Scale features (on train)
StandardScaler scaler = new StandardScaler();
double[][] XTrainScaled = scaler.fitTransform(split.XTrain);

// 5. Apply the same scaling to test (without re-fitting)
double[][] XTestScaled = scaler.transform(split.XTest);

// 6. Train model
KNearestNeighbors model = new KNearestNeighbors(5);
model.train(XTrainScaled, split.yTrain);

// 7. Evaluate
int[] predictions = model.predict(XTestScaled);
double accuracy = Metrics.accuracy(split.yTest, predictions);
```

### For Regression:

```java
// 1. Load data
double[][] X = loadData();
double[] y = loadTargets();

// 2. Split
DataSplit.TrainTestSplitRegression split = DataSplit.trainTestSplit(X, y, 0.2, true, 42);

// 3. Scale
MinMaxScaler scaler = new MinMaxScaler();
double[][] XTrainScaled = scaler.fitTransform(split.XTrain);
double[][] XTestScaled = scaler.transform(split.XTest);

// 4. Train
LinearRegression model = new LinearRegression();
model.train(XTrainScaled, split.yTrain);

// 5. Evaluate
double[] predictions = model.predict(XTestScaled);
double rmse = Metrics.rmse(split.yTest, predictions);
```

### Label Encoding:

```java
// If you have categorical labels as String
String[] categoricalLabels = {"red", "green", "blue", "red", "green"};

LabelEncoder encoder = new LabelEncoder();
int[] numericLabels = encoder.fitTransform(categoricalLabels);

// Use numericLabels to train models
// ...

// Convert predictions back to original labels
String[] predictedLabels = encoder.inverseTransform(predictions);
```

---

## ⚠️ Best Practices

1. **Always fit on train, transform on test:**
   ```java
   scaler.fit(XTrain);           // ✓ Correct
   XTrain = scaler.transform(XTrain);
   XTest = scaler.transform(XTest);  // ✓ Uses same parameters
   
   // ✗ INCORRECT: Don't fit on test
   scaler.fit(XTest);  // ✗ Causes data leakage
   ```

2. **Handle missing values before scaling:**
   ```java
   X = imputer.fitTransform(X);    // First impute
   X = scaler.fitTransform(X);     // Then scale
   ```

3. **Use stratified split for imbalanced data:**
   ```java
   // If you have 90% class A and 10% class B
   split = DataSplit.stratifiedTrainTestSplit(X, y, 0.2, 42);
   // Maintains the 90/10 proportion in train and test
   ```

4. **Save scalers and encoders for production:**
   ```java
   // In production, you'll need the same transformers
   StandardScaler scaler = trainScaler();
   LabelEncoder encoder = trainEncoder();
   // Save these objects to use in future predictions
   ```

---

## 📊 Tests

All classes include comprehensive tests:
- `MinMaxScalerTest` - 8 tests
- `StandardScalerTest` - 8 tests
- `DataSplitTest` - 9 tests
- `SimpleImputerTest` - 9 tests
- `LabelEncoderTest` - 13 tests

**Total: 47 preprocessing tests**

Run tests:
```bash
mvn test -Dtest="io.github.yasmramos.mindforge.preprocessing.*"
```
