# Data Preprocessing Package

El paquete `com.mindforge.preprocessing` proporciona herramientas esenciales para preparar datos antes del entrenamiento de modelos de machine learning.

## 📦 Clases Implementadas

### 1. **MinMaxScaler**
Escala las características a un rango específico (predeterminado: [0, 1]).

**Uso:**
```java
double[][] data = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}};
MinMaxScaler scaler = new MinMaxScaler(); // Rango [0, 1]
// O con rango personalizado:
// MinMaxScaler scaler = new MinMaxScaler(-1.0, 1.0);

double[][] scaled = scaler.fitTransform(data);
// Para datos nuevos:
double[][] testScaled = scaler.transform(testData);
// Reversión:
double[][] original = scaler.inverseTransform(scaled);
```

**Métodos:**
- `fit(double[][] X)` - Calcula min y max de cada característica
- `transform(double[][] X)` - Escala los datos
- `fitTransform(double[][] X)` - Fit y transform en un paso
- `inverseTransform(double[][] X)` - Revierte la escala
- `getFeatureMin()` - Obtiene valores mínimos
- `getFeatureMax()` - Obtiene valores máximos

---

### 2. **StandardScaler**
Estandariza características removiendo la media y escalando a varianza unitaria (Z-score normalization).

**Uso:**
```java
double[][] data = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}};
StandardScaler scaler = new StandardScaler();
// O con opciones:
// StandardScaler scaler = new StandardScaler(true, true);  // withMean, withStd

double[][] scaled = scaler.fitTransform(data);
double[][] original = scaler.inverseTransform(scaled);
```

**Métodos:**
- `fit(double[][] X)` - Calcula media y desviación estándar
- `transform(double[][] X)` - Estandariza los datos
- `fitTransform(double[][] X)` - Fit y transform en un paso
- `inverseTransform(double[][] X)` - Revierte la estandarización
- `getMean()` - Obtiene las medias
- `getStd()` - Obtiene las desviaciones estándar

---

### 3. **DataSplit**
Divide conjuntos de datos en train y test.

**Uso - Clasificación:**
```java
double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}};
int[] y = {0, 0, 1, 1};

// Split simple
DataSplit.TrainTestSplit split = DataSplit.trainTestSplit(
    X, y, 
    0.25,      // 25% para test
    true,      // shuffle
    42         // random seed (null para aleatorio)
);

// Acceder a los datos
double[][] XTrain = split.XTrain;
double[][] XTest = split.XTest;
int[] yTrain = split.yTrain;
int[] yTest = split.yTest;
```

**Uso - Regresión:**
```java
double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}};
double[] y = {1.5, 2.5, 3.5, 4.5};

DataSplit.TrainTestSplitRegression split = DataSplit.trainTestSplit(
    X, y, 0.25, true, 42
);

double[] yTrain = split.yTrain;  // Tipo double para regresión
double[] yTest = split.yTest;
```

**Uso - Split Estratificado:**
```java
// Mantiene la proporción de clases en train y test
DataSplit.TrainTestSplit split = DataSplit.stratifiedTrainTestSplit(
    X, y,
    0.25,   // test size
    42      // random seed
);
```

**Métodos:**
- `trainTestSplit(X, y, testSize, shuffle, seed)` - Split simple (clasificación)
- `trainTestSplit(X, y, testSize, shuffle, seed)` - Split simple (regresión)
- `stratifiedTrainTestSplit(X, y, testSize, seed)` - Split estratificado

---

### 4. **SimpleImputer**
Completa valores faltantes (NaN) en datasets.

**Estrategias disponibles:**
- `MEAN` - Reemplaza con la media
- `MEDIAN` - Reemplaza con la mediana
- `MOST_FREQUENT` - Reemplaza con el valor más frecuente
- `CONSTANT` - Reemplaza con un valor constante

**Uso:**
```java
double[][] data = {
    {1.0, 2.0},
    {Double.NaN, 3.0},
    {7.0, Double.NaN},
    {4.0, 6.0}
};

// Imputación por media
SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.MEAN);
double[][] filled = imputer.fitTransform(data);

// Imputación con valor constante
SimpleImputer imputer2 = new SimpleImputer(
    SimpleImputer.ImputeStrategy.CONSTANT, 
    0.0  // valor constante
);
double[][] filled2 = imputer2.fitTransform(data);
```

**Métodos:**
- `fit(double[][] X)` - Calcula estadísticas de imputación
- `transform(double[][] X)` - Rellena valores faltantes
- `fitTransform(double[][] X)` - Fit y transform en un paso
- `getStatistics()` - Obtiene los valores de imputación calculados

---

### 5. **LabelEncoder**
Codifica etiquetas categóricas a valores numéricos.

**Uso:**
```java
String[] labels = {"cat", "dog", "cat", "bird", "dog"};

LabelEncoder encoder = new LabelEncoder();
int[] encoded = encoder.fitTransform(labels);
// Resultado: [0, 1, 0, 2, 1]

// Decodificar de vuelta
String[] decoded = encoder.inverseTransform(encoded);
// Resultado: ["cat", "dog", "cat", "bird", "dog"]

// Codificar valores individuales
int catCode = encoder.encode("cat");        // 0
String label = encoder.decode(1);           // "dog"

// Obtener información
String[] classes = encoder.getClasses();    // ["cat", "dog", "bird"]
int numClasses = encoder.getNumClasses();   // 3
```

**Métodos:**
- `fit(String[] labels)` - Aprende el mapeo de etiquetas
- `transform(String[] labels)` - Codifica etiquetas a enteros
- `fitTransform(String[] labels)` - Fit y transform en un paso
- `inverseTransform(int[] encoded)` - Decodifica a etiquetas originales
- `encode(String label)` - Codifica una etiqueta individual
- `decode(int encoded)` - Decodifica un valor individual
- `getClasses()` - Obtiene todas las clases únicas
- `getNumClasses()` - Obtiene el número de clases

---

## 🔄 Flujo de Trabajo Típico

### Para Clasificación:

```java
// 1. Cargar datos
double[][] X = loadData();
int[] y = loadLabels();

// 2. Imputar valores faltantes (si existen)
SimpleImputer imputer = new SimpleImputer(ImputeStrategy.MEAN);
X = imputer.fitTransform(X);

// 3. Dividir en train/test
DataSplit.TrainTestSplit split = DataSplit.stratifiedTrainTestSplit(X, y, 0.2, 42);

// 4. Escalar características (en train)
StandardScaler scaler = new StandardScaler();
double[][] XTrainScaled = scaler.fitTransform(split.XTrain);

// 5. Aplicar el mismo escalado a test (sin re-fit)
double[][] XTestScaled = scaler.transform(split.XTest);

// 6. Entrenar modelo
KNearestNeighbors model = new KNearestNeighbors(5);
model.train(XTrainScaled, split.yTrain);

// 7. Evaluar
int[] predictions = model.predict(XTestScaled);
double accuracy = Metrics.accuracy(split.yTest, predictions);
```

### Para Regresión:

```java
// 1. Cargar datos
double[][] X = loadData();
double[] y = loadTargets();

// 2. Split
DataSplit.TrainTestSplitRegression split = DataSplit.trainTestSplit(X, y, 0.2, true, 42);

// 3. Escalar
MinMaxScaler scaler = new MinMaxScaler();
double[][] XTrainScaled = scaler.fitTransform(split.XTrain);
double[][] XTestScaled = scaler.transform(split.XTest);

// 4. Entrenar
LinearRegression model = new LinearRegression();
model.train(XTrainScaled, split.yTrain);

// 5. Evaluar
double[] predictions = model.predict(XTestScaled);
double rmse = Metrics.rmse(split.yTest, predictions);
```

### Codificación de Labels:

```java
// Si tienes etiquetas categóricas como String
String[] categoricalLabels = {"red", "green", "blue", "red", "green"};

LabelEncoder encoder = new LabelEncoder();
int[] numericLabels = encoder.fitTransform(categoricalLabels);

// Usar numericLabels para entrenar modelos
// ...

// Convertir predicciones de vuelta a labels originales
String[] predictedLabels = encoder.inverseTransform(predictions);
```

---

## ⚠️ Buenas Prácticas

1. **Siempre fit en train, transform en test:**
   ```java
   scaler.fit(XTrain);           // ✓ Correcto
   XTrain = scaler.transform(XTrain);
   XTest = scaler.transform(XTest);  // ✓ Usa mismos parámetros
   
   // ✗ INCORRECTO: No hacer fit en test
   scaler.fit(XTest);  // ✗ Causa data leakage
   ```

2. **Manejar valores faltantes antes de escalar:**
   ```java
   X = imputer.fitTransform(X);    // Primero imputar
   X = scaler.fitTransform(X);     // Luego escalar
   ```

3. **Usar stratified split para datos desbalanceados:**
   ```java
   // Si tienes 90% clase A y 10% clase B
   split = DataSplit.stratifiedTrainTestSplit(X, y, 0.2, 42);
   // Mantiene la proporción 90/10 en train y test
   ```

4. **Guardar scalers y encoders para producción:**
   ```java
   // En producción, necesitarás los mismos transformadores
   StandardScaler scaler = trainScaler();
   LabelEncoder encoder = trainEncoder();
   // Guardar estos objetos para usar en predicciones futuras
   ```

---

## 📊 Tests

Todas las clases incluyen tests comprehensivos:
- `MinMaxScalerTest` - 8 tests
- `StandardScalerTest` - 8 tests
- `DataSplitTest` - 9 tests
- `SimpleImputerTest` - 9 tests
- `LabelEncoderTest` - 13 tests

**Total: 47 tests de preprocessing**

Ejecutar tests:
```bash
mvn test -Dtest="com.mindforge.preprocessing.*"
```
