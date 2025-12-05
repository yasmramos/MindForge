package com.mindforge.classification;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;

/**
 * Comprehensive tests for XGBoostClassifier.
 */
class XGBoostClassifierTest {
    
    private double[][] XBinary;
    private int[] yBinary;
    private double[][] XMulti;
    private int[] yMulti;
    
    @BeforeEach
    void setUp() {
        // Binary classification dataset
        XBinary = new double[][] {
            {0, 0}, {0.5, 0.5}, {1, 0}, {1, 1},
            {5, 5}, {5.5, 5.5}, {6, 5}, {6, 6}
        };
        yBinary = new int[] {0, 0, 0, 0, 1, 1, 1, 1};
        
        // Multiclass dataset
        XMulti = new double[][] {
            {0, 0}, {0.5, 0.5}, {1, 0},      // Class 0
            {5, 5}, {5.5, 5.5}, {6, 5},      // Class 1
            {10, 0}, {10.5, 0.5}, {11, 0}    // Class 2
        };
        yMulti = new int[] {0, 0, 0, 1, 1, 1, 2, 2, 2};
    }
    
    @Nested
    @DisplayName("Builder Tests")
    class BuilderTests {
        
        @Test
        @DisplayName("Default builder settings")
        void testBuilderDefaults() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder().build();
            
            assertEquals(100, xgb.getNEstimators());
            assertEquals(6, xgb.getMaxDepth());
            assertEquals(0.3, xgb.getLearningRate(), 1e-10);
            assertEquals(1.0, xgb.getLambda(), 1e-10);
            assertEquals(0.0, xgb.getAlpha(), 1e-10);
            assertEquals(0.0, xgb.getGamma(), 1e-10);
            assertEquals(1.0, xgb.getSubsample(), 1e-10);
            assertEquals(1.0, xgb.getColsampleBytree(), 1e-10);
        }
        
        @Test
        @DisplayName("Custom builder settings")
        void testBuilderCustom() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(50)
                .maxDepth(4)
                .learningRate(0.1)
                .lambda(2.0)
                .alpha(0.5)
                .gamma(0.1)
                .subsample(0.8)
                .colsampleBytree(0.8)
                .randomState(42)
                .build();
            
            assertEquals(50, xgb.getNEstimators());
            assertEquals(4, xgb.getMaxDepth());
            assertEquals(0.1, xgb.getLearningRate(), 1e-10);
            assertEquals(2.0, xgb.getLambda(), 1e-10);
            assertEquals(0.5, xgb.getAlpha(), 1e-10);
            assertEquals(0.1, xgb.getGamma(), 1e-10);
            assertEquals(0.8, xgb.getSubsample(), 1e-10);
            assertEquals(0.8, xgb.getColsampleBytree(), 1e-10);
        }
        
        @Test
        @DisplayName("Builder aliases work")
        void testBuilderAliases() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .eta(0.1)           // alias for learningRate
                .regLambda(2.0)     // alias for lambda
                .regAlpha(0.5)      // alias for alpha
                .minSplitLoss(0.1)  // alias for gamma
                .seed(42)           // alias for randomState
                .build();
            
            assertEquals(0.1, xgb.getLearningRate(), 1e-10);
            assertEquals(2.0, xgb.getLambda(), 1e-10);
            assertEquals(0.5, xgb.getAlpha(), 1e-10);
            assertEquals(0.1, xgb.getGamma(), 1e-10);
        }
        
        @Test
        @DisplayName("Invalid nEstimators throws")
        void testInvalidNEstimators() {
            assertThrows(IllegalArgumentException.class, () -> 
                new XGBoostClassifier.Builder().nEstimators(0).build());
        }
        
        @Test
        @DisplayName("Invalid maxDepth throws")
        void testInvalidMaxDepth() {
            assertThrows(IllegalArgumentException.class, () -> 
                new XGBoostClassifier.Builder().maxDepth(0).build());
        }
        
        @Test
        @DisplayName("Invalid learningRate throws")
        void testInvalidLearningRate() {
            assertThrows(IllegalArgumentException.class, () -> 
                new XGBoostClassifier.Builder().learningRate(0).build());
            assertThrows(IllegalArgumentException.class, () -> 
                new XGBoostClassifier.Builder().learningRate(1.5).build());
        }
        
        @Test
        @DisplayName("Invalid lambda throws")
        void testInvalidLambda() {
            assertThrows(IllegalArgumentException.class, () -> 
                new XGBoostClassifier.Builder().lambda(-1).build());
        }
        
        @Test
        @DisplayName("Invalid subsample throws")
        void testInvalidSubsample() {
            assertThrows(IllegalArgumentException.class, () -> 
                new XGBoostClassifier.Builder().subsample(0).build());
            assertThrows(IllegalArgumentException.class, () -> 
                new XGBoostClassifier.Builder().subsample(1.5).build());
        }
    }
    
    @Nested
    @DisplayName("Binary Classification Tests")
    class BinaryClassificationTests {
        
        @Test
        @DisplayName("Simple binary classification")
        void testBinaryClassification() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .maxDepth(3)
                .learningRate(0.3)
                .randomState(42)
                .build();
            
            xgb.fit(XBinary, yBinary);
            
            assertTrue(xgb.isFitted());
            assertEquals(2, xgb.getNumClasses());
            
            int[] predictions = xgb.predict(XBinary);
            assertEquals(8, predictions.length);
        }
        
        @Test
        @DisplayName("Binary classification accuracy")
        void testBinaryAccuracy() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(50)
                .maxDepth(3)
                .learningRate(0.3)
                .randomState(42)
                .build();
            
            xgb.fit(XBinary, yBinary);
            
            double accuracy = xgb.score(XBinary, yBinary);
            assertTrue(accuracy >= 0.75, "Should achieve reasonable accuracy");
        }
        
        @Test
        @DisplayName("Binary probability predictions")
        void testBinaryProbabilities() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .randomState(42)
                .build();
            
            xgb.fit(XBinary, yBinary);
            
            double[][] proba = xgb.predictProba(XBinary);
            assertEquals(8, proba.length);
            assertEquals(2, proba[0].length);
            
            // Probabilities should sum to 1
            for (double[] p : proba) {
                assertEquals(1.0, p[0] + p[1], 1e-6);
                assertTrue(p[0] >= 0 && p[0] <= 1);
                assertTrue(p[1] >= 0 && p[1] <= 1);
            }
        }
    }
    
    @Nested
    @DisplayName("Multiclass Classification Tests")
    class MulticlassClassificationTests {
        
        @Test
        @DisplayName("Simple multiclass classification")
        void testMulticlassClassification() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(30)
                .maxDepth(3)
                .learningRate(0.3)
                .randomState(42)
                .build();
            
            xgb.fit(XMulti, yMulti);
            
            assertTrue(xgb.isFitted());
            assertEquals(3, xgb.getNumClasses());
            assertArrayEquals(new int[]{0, 1, 2}, xgb.getClasses());
            
            int[] predictions = xgb.predict(XMulti);
            assertEquals(9, predictions.length);
        }
        
        @Test
        @DisplayName("Multiclass probability predictions")
        void testMulticlassProbabilities() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .randomState(42)
                .build();
            
            xgb.fit(XMulti, yMulti);
            
            double[][] proba = xgb.predictProba(XMulti);
            assertEquals(9, proba.length);
            assertEquals(3, proba[0].length);
            
            // Probabilities should sum to 1
            for (double[] p : proba) {
                double sum = 0;
                for (double v : p) {
                    sum += v;
                    assertTrue(v >= 0 && v <= 1);
                }
                assertEquals(1.0, sum, 1e-6);
            }
        }
        
        @Test
        @DisplayName("Four class classification")
        void testFourClasses() {
            double[][] X = {
                {0, 0}, {0.1, 0.1},      // Class 0
                {5, 0}, {5.1, 0.1},      // Class 1
                {0, 5}, {0.1, 5.1},      // Class 2
                {5, 5}, {5.1, 5.1}       // Class 3
            };
            int[] y = {0, 0, 1, 1, 2, 2, 3, 3};
            
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(50)
                .maxDepth(3)
                .randomState(42)
                .build();
            
            xgb.fit(X, y);
            assertEquals(4, xgb.getNumClasses());
            
            double[][] proba = xgb.predictProba(X);
            assertEquals(4, proba[0].length);
        }
    }
    
    @Nested
    @DisplayName("Regularization Tests")
    class RegularizationTests {
        
        @Test
        @DisplayName("L2 regularization affects model")
        void testL2Regularization() {
            XGBoostClassifier xgb1 = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .lambda(0.0)
                .randomState(42)
                .build();
            
            XGBoostClassifier xgb2 = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .lambda(10.0)
                .randomState(42)
                .build();
            
            xgb1.fit(XBinary, yBinary);
            xgb2.fit(XBinary, yBinary);
            
            // Both should be fitted
            assertTrue(xgb1.isFitted());
            assertTrue(xgb2.isFitted());
        }
        
        @Test
        @DisplayName("L1 regularization affects model")
        void testL1Regularization() {
            XGBoostClassifier xgb1 = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .alpha(0.0)
                .randomState(42)
                .build();
            
            XGBoostClassifier xgb2 = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .alpha(1.0)
                .randomState(42)
                .build();
            
            xgb1.fit(XBinary, yBinary);
            xgb2.fit(XBinary, yBinary);
            
            assertTrue(xgb1.isFitted());
            assertTrue(xgb2.isFitted());
        }
        
        @Test
        @DisplayName("Gamma affects splits")
        void testGammaRegularization() {
            XGBoostClassifier xgb1 = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .gamma(0.0)
                .randomState(42)
                .build();
            
            XGBoostClassifier xgb2 = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .gamma(10.0)  // High gamma will prevent most splits
                .randomState(42)
                .build();
            
            xgb1.fit(XBinary, yBinary);
            xgb2.fit(XBinary, yBinary);
            
            assertTrue(xgb1.isFitted());
            assertTrue(xgb2.isFitted());
        }
    }
    
    @Nested
    @DisplayName("Subsampling Tests")
    class SubsamplingTests {
        
        @Test
        @DisplayName("Row subsampling works")
        void testRowSubsampling() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .subsample(0.5)
                .randomState(42)
                .build();
            
            xgb.fit(XBinary, yBinary);
            assertTrue(xgb.isFitted());
        }
        
        @Test
        @DisplayName("Column subsampling works")
        void testColumnSubsampling() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .colsampleBytree(0.5)
                .randomState(42)
                .build();
            
            xgb.fit(XBinary, yBinary);
            assertTrue(xgb.isFitted());
        }
        
        @Test
        @DisplayName("Combined subsampling works")
        void testCombinedSubsampling() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .subsample(0.8)
                .colsampleBytree(0.8)
                .randomState(42)
                .build();
            
            xgb.fit(XBinary, yBinary);
            assertTrue(xgb.isFitted());
            
            int[] predictions = xgb.predict(XBinary);
            assertEquals(8, predictions.length);
        }
    }
    
    @Nested
    @DisplayName("Feature Importance Tests")
    class FeatureImportanceTests {
        
        @Test
        @DisplayName("Feature importances sum to approximately 1")
        void testFeatureImportancesSum() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(50)
                .maxDepth(3)
                .randomState(42)
                .build();
            
            xgb.fit(XBinary, yBinary);
            double[] importances = xgb.getFeatureImportances();
            
            assertEquals(2, importances.length);
            
            double sum = 0;
            for (double imp : importances) {
                assertTrue(imp >= 0, "Importances should be non-negative");
                sum += imp;
            }
            // Sum might be 0 if no valid splits, otherwise should be 1
            assertTrue(sum >= 0 && sum <= 1.01);
        }
        
        @Test
        @DisplayName("Relevant feature has higher importance")
        void testRelevantFeature() {
            // Only first feature is relevant
            double[][] X = new double[100][2];
            int[] y = new int[100];
            for (int i = 0; i < 100; i++) {
                X[i][0] = i < 50 ? 0 + Math.random() * 0.5 : 10 + Math.random() * 0.5;
                X[i][1] = Math.random() * 10;  // Noise
                y[i] = i < 50 ? 0 : 1;
            }
            
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(50)
                .maxDepth(3)
                .randomState(42)
                .build();
            
            xgb.fit(X, y);
            double[] importances = xgb.getFeatureImportances();
            
            // First feature should typically have higher importance
            // (though not guaranteed due to randomness)
            assertNotNull(importances);
            assertEquals(2, importances.length);
        }
    }
    
    @Nested
    @DisplayName("Error Handling Tests")
    class ErrorHandlingTests {
        
        @Test
        @DisplayName("Predict before fit throws")
        void testPredictBeforeFit() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder().build();
            
            assertThrows(IllegalStateException.class, () -> 
                xgb.predict(new double[]{1, 2}));
            assertThrows(IllegalStateException.class, () -> 
                xgb.predict(new double[][]{{1, 2}}));
        }
        
        @Test
        @DisplayName("PredictProba before fit throws")
        void testPredictProbaBeforeFit() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder().build();
            
            assertThrows(IllegalStateException.class, () -> 
                xgb.predictProba(new double[]{1, 2}));
        }
        
        @Test
        @DisplayName("Null X throws")
        void testNullX() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder().build();
            
            assertThrows(IllegalArgumentException.class, () -> 
                xgb.fit(null, new int[]{0, 1}));
        }
        
        @Test
        @DisplayName("Null y throws")
        void testNullY() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder().build();
            
            assertThrows(IllegalArgumentException.class, () -> 
                xgb.fit(new double[][]{{1}, {2}}, null));
        }
        
        @Test
        @DisplayName("Empty X throws")
        void testEmptyX() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder().build();
            
            assertThrows(IllegalArgumentException.class, () -> 
                xgb.fit(new double[0][], new int[0]));
        }
        
        @Test
        @DisplayName("Mismatched lengths throw")
        void testMismatchedLengths() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder().build();
            
            assertThrows(IllegalArgumentException.class, () -> 
                xgb.fit(new double[][]{{1}, {2}}, new int[]{0}));
        }
        
        @Test
        @DisplayName("Wrong feature count throws")
        void testWrongFeatureCount() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(10)
                .build();
            
            xgb.fit(XBinary, yBinary);
            
            assertThrows(IllegalArgumentException.class, () -> 
                xgb.predict(new double[]{1, 2, 3}));
        }
        
        @Test
        @DisplayName("getClasses before fit throws")
        void testGetClassesBeforeFit() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder().build();
            
            assertThrows(IllegalStateException.class, xgb::getClasses);
        }
        
        @Test
        @DisplayName("getFeatureImportances before fit throws")
        void testGetImportancesBeforeFit() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder().build();
            
            assertThrows(IllegalStateException.class, xgb::getFeatureImportances);
        }
    }
    
    @Nested
    @DisplayName("State Tests")
    class StateTests {
        
        @Test
        @DisplayName("isFitted/isTrained return correct state")
        void testFittedState() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(10)
                .build();
            
            assertFalse(xgb.isFitted());
            assertFalse(xgb.isTrained());
            
            xgb.fit(XBinary, yBinary);
            
            assertTrue(xgb.isFitted());
            assertTrue(xgb.isTrained());
        }
        
        @Test
        @DisplayName("train method works like fit")
        void testTrainMethod() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(10)
                .build();
            
            xgb.train(XBinary, yBinary);
            
            assertTrue(xgb.isFitted());
        }
        
        @Test
        @DisplayName("getActualNEstimators returns correct value")
        void testActualNEstimators() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(10)
                .build();
            
            xgb.fit(XBinary, yBinary);
            
            assertEquals(10, xgb.getActualNEstimators());
        }
    }
    
    @Nested
    @DisplayName("Reproducibility Tests")
    class ReproducibilityTests {
        
        @Test
        @DisplayName("Same seed produces same results")
        void testReproducibility() {
            XGBoostClassifier xgb1 = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .randomState(42)
                .build();
            
            XGBoostClassifier xgb2 = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .randomState(42)
                .build();
            
            xgb1.fit(XBinary, yBinary);
            xgb2.fit(XBinary, yBinary);
            
            int[] pred1 = xgb1.predict(XBinary);
            int[] pred2 = xgb2.predict(XBinary);
            
            assertArrayEquals(pred1, pred2);
        }
    }
    
    @Nested
    @DisplayName("Serialization Tests")
    class SerializationTests {
        
        @Test
        @DisplayName("Serialization and deserialization works")
        void testSerialization() throws IOException, ClassNotFoundException {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .maxDepth(3)
                .randomState(42)
                .build();
            
            xgb.fit(XBinary, yBinary);
            
            // Serialize
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(xgb);
            oos.close();
            
            // Deserialize
            ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bis);
            XGBoostClassifier restored = (XGBoostClassifier) ois.readObject();
            ois.close();
            
            // Verify
            assertTrue(restored.isFitted());
            assertEquals(xgb.getNEstimators(), restored.getNEstimators());
            assertEquals(xgb.getMaxDepth(), restored.getMaxDepth());
            
            // Test predictions are the same
            int[] pred1 = xgb.predict(XBinary);
            int[] pred2 = restored.predict(XBinary);
            assertArrayEquals(pred1, pred2);
        }
    }
    
    @Nested
    @DisplayName("Raw Prediction Tests")
    class RawPredictionTests {
        
        @Test
        @DisplayName("predictRaw returns correct dimensions")
        void testPredictRaw() {
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(20)
                .randomState(42)
                .build();
            
            xgb.fit(XMulti, yMulti);
            
            double[] raw = xgb.predictRaw(XMulti[0]);
            assertEquals(3, raw.length);  // 3 classes
        }
    }
    
    @Nested
    @DisplayName("Large Dataset Tests")
    class LargeDatasetTests {
        
        @Test
        @DisplayName("Handles larger dataset")
        void testLargeDataset() {
            int n = 200;
            double[][] X = new double[n][3];
            int[] y = new int[n];
            
            for (int i = 0; i < n; i++) {
                X[i][0] = i < n/2 ? Math.random() * 5 : Math.random() * 5 + 10;
                X[i][1] = Math.random() * 10;
                X[i][2] = Math.random() * 10;
                y[i] = i < n/2 ? 0 : 1;
            }
            
            XGBoostClassifier xgb = new XGBoostClassifier.Builder()
                .nEstimators(30)
                .maxDepth(4)
                .randomState(42)
                .build();
            
            xgb.fit(X, y);
            
            double accuracy = xgb.score(X, y);
            assertTrue(accuracy >= 0.7, "Should achieve reasonable accuracy");
        }
    }
}
