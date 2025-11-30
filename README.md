# MindForge

**MindForge** es una biblioteca de Machine Learning e Inteligencia Artificial para Java, inspirada en bibliotecas como Smile, diseñada para ser fácil de usar y eficiente.

## 🚀 Características

- **Algoritmos de Clasificación**: K-Nearest Neighbors (KNN), y más por venir
- **Algoritmos de Regresión**: Regresión Lineal, y más por venir
- **Algoritmos de Clustering**: K-Means, y más por venir
- **Métricas de Evaluación**: Accuracy, Precision, Recall, F1-Score, MSE, RMSE, MAE, R²
- **Funciones de Distancia**: Euclidiana, Manhattan, Coseno, Minkowski
- **Interfaz Simple y Consistente**: APIs intuitivas para todos los algoritmos

## 📦 Estructura del Proyecto

```
MindForge/
├── src/main/java/com/mindforge/
│   ├── classification/     # Algoritmos de clasificación
│   │   ├── Classifier.java
│   │   └── KNearestNeighbors.java
│   ├── regression/         # Algoritmos de regresión
│   │   ├── Regressor.java
│   │   └── LinearRegression.java
│   ├── clustering/         # Algoritmos de clustering
│   │   ├── Clusterer.java
│   │   └── KMeans.java
│   ├── math/              # Funciones matemáticas
│   │   └── Distance.java
│   ├── validation/        # Métricas de evaluación
│   │   └── Metrics.java
│   ├── neural/            # Redes neuronales (próximamente)
│   ├── data/              # Procesamiento de datos (próximamente)
│   └── util/              # Utilidades (próximamente)
└── pom.xml
```

## 🔧 Requisitos

- **Java 17** o superior
- **Maven 3.6** o superior

## 📥 Instalación

Clona el repositorio y compila el proyecto:

```bash
git clone https://github.com/yasmramos/MindForge.git
cd MindForge
mvn clean install
```

## 💡 Ejemplos de Uso

### Clasificación con K-Nearest Neighbors

```java
import com.mindforge.classification.KNearestNeighbors;
import com.mindforge.validation.Metrics;

// Datos de entrenamiento
double[][] X_train = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0}, {6.0, 5.0}, {7.0, 8.0}, {8.0, 7.0}};
int[] y_train = {0, 0, 0, 1, 1, 1};

// Crear y entrenar el modelo
KNearestNeighbors knn = new KNearestNeighbors(3);
knn.train(X_train, y_train);

// Hacer predicciones
double[] testPoint = {5.0, 5.0};
int prediction = knn.predict(testPoint);
System.out.println("Predicción: " + prediction);

// Evaluar el modelo
int[] predictions = knn.predict(X_train);
double accuracy = Metrics.accuracy(y_train, predictions);
System.out.println("Accuracy: " + accuracy);
```

### Regresión Lineal

```java
import com.mindforge.regression.LinearRegression;
import com.mindforge.validation.Metrics;

// Datos de entrenamiento
double[][] X_train = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
double[] y_train = {2.0, 4.0, 6.0, 8.0, 10.0};

// Crear y entrenar el modelo
LinearRegression lr = new LinearRegression();
lr.train(X_train, y_train);

// Hacer predicciones
double[] testPoint = {6.0};
double prediction = lr.predict(testPoint);
System.out.println("Predicción: " + prediction);

// Evaluar el modelo
double[] predictions = lr.predict(X_train);
double rmse = Metrics.rmse(y_train, predictions);
System.out.println("RMSE: " + rmse);
```

### Clustering con K-Means

```java
import com.mindforge.clustering.KMeans;

// Datos
double[][] data = {
    {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, 
    {8.0, 8.0}, {1.0, 0.6}, {9.0, 11.0}
};

// Crear y ejecutar K-Means
KMeans kmeans = new KMeans(2);
int[] clusters = kmeans.cluster(data);

// Ver asignaciones de clusters
for (int i = 0; i < clusters.length; i++) {
    System.out.println("Punto " + i + " -> Cluster " + clusters[i]);
}

// Obtener centroides
double[][] centroids = kmeans.getCentroids();
```

## 🧪 Ejecutar Tests

```bash
mvn test
```

## 🏗️ Compilar

```bash
mvn compile
```

## 📦 Empaquetar

```bash
mvn package
```

## 🛣️ Roadmap

- [ ] Árboles de Decisión
- [ ] Random Forest
- [ ] Support Vector Machines (SVM)
- [ ] Redes Neuronales
- [ ] Naive Bayes
- [ ] Gradient Boosting
- [ ] PCA (Análisis de Componentes Principales)
- [ ] Procesamiento de datos y normalización
- [ ] Validación cruzada
- [ ] Selección de características

## 📄 Información del Proyecto

- **Group ID**: com.mindforge
- **Artifact ID**: mindforge
- **Version**: 1.0-SNAPSHOT
- **Java Version**: 17

## 📚 Dependencias Principales

- Apache Commons Math 3.6.1
- ND4J 1.0.0-M2.1 (para cálculo numérico)
- JUnit 5.10.1 (para testing)
- SLF4J 2.0.9 (para logging)

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request.

## 📝 Licencia

TBD

---

**Autor**: Matrix Agent  
**Inspirado en**: Smile (Statistical Machine Intelligence and Learning Engine)
