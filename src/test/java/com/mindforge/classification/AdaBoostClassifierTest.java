package com.mindforge.classification;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;

/**
 * Comprehensive tests for AdaBoostClassifier.
 */
class AdaBoostClassifierTest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Default constructor")
        void testDefaultConstructor() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertEquals(50, ada.getNEstimators());
            assertEquals(1.0, ada.getLearningRate(), 1e-10);
        }
        
        @Test
        @DisplayName("Constructor with nEstimators")
        void testNEstimatorsConstructor() {
            AdaBoostClassifier ada = new AdaBoostClassifier(100);
            assertEquals(100, ada.getNEstimators());
        }
        
        @Test
        @DisplayName("Full constructor")
        void testFullConstructor() {
            AdaBoostClassifier ada = new AdaBoostClassifier(25, 0.5, 42);
            assertEquals(25, ada.getNEstimators());
            assertEquals(0.5, ada.getLearningRate(), 1e-10);
        }
        
        @Test
        @DisplayName("Invalid nEstimators throws exception")
        void testInvalidNEstimators() {
            assertThrows(IllegalArgumentException.class, () -> new AdaBoostClassifier(0));
            assertThrows(IllegalArgumentException.class, () -> new AdaBoostClassifier(-1));
        }
        
        @Test
        @DisplayName("Invalid learningRate throws exception")
        void testInvalidLearningRate() {
            assertThrows(IllegalArgumentException.class, () -> new AdaBoostClassifier(10, 0, 42));
            assertThrows(IllegalArgumentException.class, () -> new AdaBoostClassifier(10, -0.5, 42));
        }
    }
    
    @Nested
    @DisplayName("Binary Classification Tests")
    class BinaryClassificationTests {
        
        @Test
        @DisplayName("Simple linearly separable data")
        void testLinearSeparable() {
            AdaBoostClassifier ada = new AdaBoostClassifier(10, 1.0, 42);
            
            double[][] X = {
                {0, 0}, {0, 1}, {1, 0}, {1, 1},
                {5, 5}, {5, 6}, {6, 5}, {6, 6}
            };
            int[] y = {0, 0, 0, 0, 1, 1, 1, 1};
            
            ada.fit(X, y);
            int[] predictions = ada.predict(X);
            
            // Should achieve good accuracy on training data
            int correct = 0;
            for (int i = 0; i < y.length; i++) {
                if (predictions[i] == y[i]) correct++;
            }
            assertTrue(correct >= 6, "Should classify at least 75% correctly");
        }
        
        @Test
        @DisplayName("XOR-like pattern")
        void testXORPattern() {
            AdaBoostClassifier ada = new AdaBoostClassifier(50, 1.0, 42);
            
            double[][] X = {
                {0, 0}, {1, 1}, {2, 2},
                {0, 1}, {1, 0}, {1, 2}, {2, 1}
            };
            int[] y = {0, 0, 0, 1, 1, 1, 1};
            
            ada.fit(X, y);
            assertTrue(ada.isFitted());
            assertTrue(ada.getActualNEstimators() > 0);
        }
        
        @Test
        @DisplayName("Negative class labels")
        void testNegativeLabels() {
            AdaBoostClassifier ada = new AdaBoostClassifier(10, 1.0, 42);
            
            double[][] X = {{0}, {1}, {2}, {3}, {4}, {5}};
            int[] y = {-1, -1, -1, 1, 1, 1};
            
            ada.fit(X, y);
            int[] predictions = ada.predict(X);
            
            // Check predictions are in the right set
            for (int pred : predictions) {
                assertTrue(pred == -1 || pred == 1);
            }
        }
        
        @Test
        @DisplayName("Different class labels")
        void testDifferentLabels() {
            AdaBoostClassifier ada = new AdaBoostClassifier(10, 1.0, 42);
            
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {5, 5, 10, 10};
            
            ada.fit(X, y);
            int[] classes = ada.getClasses();
            
            assertArrayEquals(new int[]{5, 10}, classes);
        }
    }
    
    @Nested
    @DisplayName("Probability Tests")
    class ProbabilityTests {
        
        @Test
        @DisplayName("Predict probability returns valid probabilities")
        void testPredictProba() {
            AdaBoostClassifier ada = new AdaBoostClassifier(20, 1.0, 42);
            
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            ada.fit(X, y);
            double[][] proba = ada.predictProba(X);
            
            assertEquals(4, proba.length);
            assertEquals(2, proba[0].length);
            
            // Probabilities should sum to 1
            for (double[] p : proba) {
                assertEquals(1.0, p[0] + p[1], 1e-6);
                assertTrue(p[0] >= 0 && p[0] <= 1);
                assertTrue(p[1] >= 0 && p[1] <= 1);
            }
        }
        
        @Test
        @DisplayName("Probability reflects confidence")
        void testProbaConfidence() {
            AdaBoostClassifier ada = new AdaBoostClassifier(50, 1.0, 42);
            
            double[][] X = {{0}, {1}, {100}, {101}};
            int[] y = {0, 0, 1, 1};
            
            ada.fit(X, y);
            double[][] proba = ada.predictProba(X);
            
            // Points clearly in class 0 region should have higher prob for class 0
            assertTrue(proba[0][0] > proba[0][1] || proba[1][0] > proba[1][1]);
            // Points clearly in class 1 region should have higher prob for class 1
            assertTrue(proba[2][1] > proba[2][0] || proba[3][1] > proba[3][0]);
        }
    }
    
    @Nested
    @DisplayName("Feature Importance Tests")
    class FeatureImportanceTests {
        
        @Test
        @DisplayName("Feature importances sum to 1")
        void testFeatureImportancesSum() {
            AdaBoostClassifier ada = new AdaBoostClassifier(20, 1.0, 42);
            
            double[][] X = {{0, 0}, {0, 1}, {10, 0}, {10, 1}};
            int[] y = {0, 0, 1, 1};
            
            ada.fit(X, y);
            double[] importances = ada.getFeatureImportances();
            
            assertEquals(2, importances.length);
            double sum = 0;
            for (double imp : importances) {
                sum += imp;
                assertTrue(imp >= 0);
            }
            assertEquals(1.0, sum, 1e-6);
        }
        
        @Test
        @DisplayName("Relevant feature has higher importance")
        void testRelevantFeature() {
            AdaBoostClassifier ada = new AdaBoostClassifier(50, 1.0, 42);
            
            // Only first feature is relevant
            double[][] X = new double[100][2];
            int[] y = new int[100];
            for (int i = 0; i < 100; i++) {
                X[i][0] = i < 50 ? 0 : 10; // Relevant
                X[i][1] = Math.random() * 10; // Noise
                y[i] = i < 50 ? 0 : 1;
            }
            
            ada.fit(X, y);
            double[] importances = ada.getFeatureImportances();
            
            // First feature should have higher importance
            assertTrue(importances[0] > importances[1],
                "Relevant feature should have higher importance");
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Single estimator")
        void testSingleEstimator() {
            AdaBoostClassifier ada = new AdaBoostClassifier(1);
            
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            ada.fit(X, y);
            assertEquals(1, ada.getActualNEstimators());
        }
        
        @Test
        @DisplayName("All same labels - one class only")
        void testSameLabels() {
            AdaBoostClassifier ada = new AdaBoostClassifier(10);
            
            double[][] X = {{0}, {1}, {2}};
            int[] y = {1, 1, 1};
            
            // Should throw because we need 2 classes
            assertThrows(IllegalArgumentException.class, () -> ada.fit(X, y));
        }
        
        @Test
        @DisplayName("Three or more classes throws exception")
        void testMulticlass() {
            AdaBoostClassifier ada = new AdaBoostClassifier(10);
            
            double[][] X = {{0}, {1}, {2}};
            int[] y = {0, 1, 2};
            
            assertThrows(IllegalArgumentException.class, () -> ada.fit(X, y));
        }
        
        @Test
        @DisplayName("Null X throws exception")
        void testNullX() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertThrows(IllegalArgumentException.class, () -> ada.fit(null, new int[]{0, 1}));
        }
        
        @Test
        @DisplayName("Null y throws exception")
        void testNullY() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertThrows(IllegalArgumentException.class, () -> ada.fit(new double[][]{{0}, {1}}, null));
        }
        
        @Test
        @DisplayName("Empty X throws exception")
        void testEmptyX() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertThrows(IllegalArgumentException.class, () -> ada.fit(new double[0][], new int[0]));
        }
        
        @Test
        @DisplayName("Mismatched X and y length throws exception")
        void testMismatchedLength() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> ada.fit(new double[][]{{0}, {1}}, new int[]{0}));
        }
    }
    
    @Nested
    @DisplayName("State Tests")
    class StateTests {
        
        @Test
        @DisplayName("isFitted returns correct state")
        void testIsFitted() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertFalse(ada.isFitted());
            
            ada.fit(new double[][]{{0}, {1}}, new int[]{0, 1});
            assertTrue(ada.isFitted());
        }
        
        @Test
        @DisplayName("Predict before fit throws exception")
        void testPredictBeforeFit() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertThrows(IllegalStateException.class, 
                () -> ada.predict(new double[][]{{0}}));
        }
        
        @Test
        @DisplayName("PredictProba before fit throws exception")
        void testPredictProbaBeforeFit() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertThrows(IllegalStateException.class, 
                () -> ada.predictProba(new double[][]{{0}}));
        }
        
        @Test
        @DisplayName("getActualNEstimators before fit throws exception")
        void testGetActualNEstimatorsBeforeFit() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertThrows(IllegalStateException.class, ada::getActualNEstimators);
        }
        
        @Test
        @DisplayName("getEstimatorWeights before fit throws exception")
        void testGetEstimatorWeightsBeforeFit() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertThrows(IllegalStateException.class, ada::getEstimatorWeights);
        }
        
        @Test
        @DisplayName("getFeatureImportances before fit throws exception")
        void testGetFeatureImportancesBeforeFit() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertThrows(IllegalStateException.class, ada::getFeatureImportances);
        }
        
        @Test
        @DisplayName("getClasses before fit throws exception")
        void testGetClassesBeforeFit() {
            AdaBoostClassifier ada = new AdaBoostClassifier();
            assertThrows(IllegalStateException.class, ada::getClasses);
        }
        
        @Test
        @DisplayName("Dimension mismatch in predict throws exception")
        void testDimensionMismatch() {
            AdaBoostClassifier ada = new AdaBoostClassifier(10);
            ada.fit(new double[][]{{0, 0}, {1, 1}}, new int[]{0, 1});
            
            assertThrows(IllegalArgumentException.class, 
                () -> ada.predict(new double[][]{{0}}));
        }
    }
    
    @Nested
    @DisplayName("Estimator Weights Tests")
    class EstimatorWeightsTests {
        
        @Test
        @DisplayName("Estimator weights are positive")
        void testPositiveWeights() {
            AdaBoostClassifier ada = new AdaBoostClassifier(20, 1.0, 42);
            
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            ada.fit(X, y);
            double[] weights = ada.getEstimatorWeights();
            
            assertTrue(weights.length > 0);
            for (double w : weights) {
                assertTrue(w > 0, "Weights should be positive");
            }
        }
    }
    
    @Nested
    @DisplayName("Learning Rate Tests")
    class LearningRateTests {
        
        @Test
        @DisplayName("Lower learning rate produces smaller weights")
        void testLearningRateEffect() {
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            AdaBoostClassifier ada1 = new AdaBoostClassifier(10, 1.0, 42);
            AdaBoostClassifier ada2 = new AdaBoostClassifier(10, 0.1, 42);
            
            ada1.fit(X, y);
            ada2.fit(X, y);
            
            double[] weights1 = ada1.getEstimatorWeights();
            double[] weights2 = ada2.getEstimatorWeights();
            
            // With same data and seed, weights with lower LR should be smaller
            double sum1 = 0, sum2 = 0;
            for (double w : weights1) sum1 += w;
            for (double w : weights2) sum2 += w;
            
            assertTrue(sum2 < sum1, "Lower learning rate should produce smaller total weight");
        }
    }
    
    @Nested
    @DisplayName("Serialization Tests")
    class SerializationTests {
        
        @Test
        @DisplayName("Serialization and deserialization works")
        void testSerialization() throws IOException, ClassNotFoundException {
            AdaBoostClassifier ada = new AdaBoostClassifier(20, 0.8, 42);
            
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            ada.fit(X, y);
            
            // Serialize
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(ada);
            oos.close();
            
            // Deserialize
            ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bis);
            AdaBoostClassifier restored = (AdaBoostClassifier) ois.readObject();
            ois.close();
            
            // Verify
            assertEquals(ada.getNEstimators(), restored.getNEstimators());
            assertEquals(ada.getLearningRate(), restored.getLearningRate(), 1e-10);
            assertTrue(restored.isFitted());
            
            // Test predictions are the same
            int[] pred1 = ada.predict(X);
            int[] pred2 = restored.predict(X);
            assertArrayEquals(pred1, pred2);
        }
    }
    
    @Nested
    @DisplayName("Random State Tests")
    class RandomStateTests {
        
        @Test
        @DisplayName("Same random state produces same results")
        void testReproducibility() {
            double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {5, 5}, {5, 6}, {6, 5}, {6, 6}};
            int[] y = {0, 0, 0, 0, 1, 1, 1, 1};
            
            AdaBoostClassifier ada1 = new AdaBoostClassifier(10, 1.0, 123);
            AdaBoostClassifier ada2 = new AdaBoostClassifier(10, 1.0, 123);
            
            ada1.fit(X, y);
            ada2.fit(X, y);
            
            int[] pred1 = ada1.predict(X);
            int[] pred2 = ada2.predict(X);
            
            assertArrayEquals(pred1, pred2);
        }
        
        @Test
        @DisplayName("Different random state may produce different results")
        void testDifferentSeeds() {
            double[][] X = new double[50][5];
            int[] y = new int[50];
            
            for (int i = 0; i < 50; i++) {
                for (int j = 0; j < 5; j++) {
                    X[i][j] = Math.random() * 10;
                }
                y[i] = i < 25 ? 0 : 1;
            }
            
            AdaBoostClassifier ada1 = new AdaBoostClassifier(20, 1.0, 1);
            AdaBoostClassifier ada2 = new AdaBoostClassifier(20, 1.0, 2);
            
            ada1.fit(X, y);
            ada2.fit(X, y);
            
            // They should both work, results may differ
            assertTrue(ada1.isFitted());
            assertTrue(ada2.isFitted());
        }
    }
    
    @Nested
    @DisplayName("Large Dataset Tests")
    class LargeDatasetTests {
        
        @Test
        @DisplayName("Handles larger dataset")
        void testLargeDataset() {
            int n = 500;
            double[][] X = new double[n][3];
            int[] y = new int[n];
            
            for (int i = 0; i < n; i++) {
                X[i][0] = i < n/2 ? Math.random() * 5 : Math.random() * 5 + 10;
                X[i][1] = Math.random() * 10;
                X[i][2] = Math.random() * 10;
                y[i] = i < n/2 ? 0 : 1;
            }
            
            AdaBoostClassifier ada = new AdaBoostClassifier(50, 1.0, 42);
            ada.fit(X, y);
            
            int[] predictions = ada.predict(X);
            
            // Calculate accuracy
            int correct = 0;
            for (int i = 0; i < n; i++) {
                if (predictions[i] == y[i]) correct++;
            }
            double accuracy = (double) correct / n;
            
            assertTrue(accuracy > 0.7, "Should achieve reasonable accuracy on structured data");
        }
    }
}
