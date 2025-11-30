# MindForge v1.0.0-alpha 🚀

This is the first alpha release of MindForge, a Machine Learning and AI library for Java.

## 🎯 Features

### Classification Algorithms
- **K-Nearest Neighbors (KNN)** - Configurable k parameter with multiple distance metrics support

### Regression Algorithms
- **Linear Regression** - Gradient descent optimization with configurable learning rate

### Clustering Algorithms
- **K-Means** - K-means++ initialization for improved convergence

### Evaluation Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Regression Metrics**: MSE, RMSE, MAE, R² Score

### Distance Functions
- Euclidean, Manhattan, Chebyshev, Minkowski distances

## 📦 Installation

### Maven
Add this to your `pom.xml`:

```xml
<dependency>
    <groupId>com.mindforge</groupId>
    <artifactId>mindforge</artifactId>
    <version>1.0.0-alpha</version>
</dependency>
```

You also need to configure the GitHub Packages repository:

```xml
<repositories>
    <repository>
        <id>github</id>
        <url>https://maven.pkg.github.com/yasmramos/MindForge</url>
    </repository>
</repositories>
```

## 🧪 Testing
All 27 unit tests passing ✅

## ⚠️ Alpha Release Notice
This is an alpha release intended for testing and feedback. The API may change in future versions.

## 📝 What's Next
- Decision Trees
- Logistic Regression  
- Naive Bayes
- Data Preprocessing utilities
- Cross-validation support

---

**Full Changelog**: Initial release
