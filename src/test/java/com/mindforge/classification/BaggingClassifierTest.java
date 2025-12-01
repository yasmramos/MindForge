package com.mindforge.classification;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;

/**
 * Comprehensive tests for BaggingClassifier.
 */
class BaggingClassifierTest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Default constructor")
        void testDefaultConstructor() {
            BaggingClassifier bc = new BaggingClassifier();
            assertEquals(10, bc.getNEstimators());
            assertFalse(bc.isTrained());
        }
        
        @Test
        @DisplayName("Constructor with supplier and estimators")
        void testSupplierConstructor() {
            BaggingClassifier bc = new BaggingClassifier(
                () -> new KNearestNeighbors(3),
                5,
                42
            );
            assertEquals(5, bc.getNEstimators());
        }
        
        @Test
        @DisplayName("Full constructor")
        void testFullConstructor() {
            BaggingClassifier bc = new BaggingClassifier(
                () -> new DecisionTreeClassifier(),
                20,
                0.8,
                0.7,
                true,
                42
            );
            assertEquals(20, bc.getNEstimators());
            assertEquals(0.8, bc.getMaxSamples(), 1e-10);
            assertEquals(0.7, bc.getMaxFeatures(), 1e-10);
            assertTrue(bc.isBootstrap());
        }
        
        @Test
        @DisplayName("Null supplier throws exception")
        void testNullSupplier() {
            assertThrows(IllegalArgumentException.class, 
                () -> new BaggingClassifier(null, 10, 42));
        }
        
        @Test
        @DisplayName("Invalid nEstimators throws exception")
        void testInvalidNEstimators() {
            assertThrows(IllegalArgumentException.class, 
                () -> new BaggingClassifier(() -> new KNearestNeighbors(1), 0, 42));
        }
        
        @Test
        @DisplayName("Invalid maxSamples throws exception")
        void testInvalidMaxSamples() {
            assertThrows(IllegalArgumentException.class, 
                () -> new BaggingClassifier(
                    () -> new KNearestNeighbors(1), 10, 0.0, 1.0, true, 42));
            assertThrows(IllegalArgumentException.class, 
                () -> new BaggingClassifier(
                    () -> new KNearestNeighbors(1), 10, 1.5, 1.0, true, 42));
        }
        
        @Test
        @DisplayName("Invalid maxFeatures throws exception")
        void testInvalidMaxFeatures() {
            assertThrows(IllegalArgumentException.class, 
                () -> new BaggingClassifier(
                    () -> new KNearestNeighbors(1), 10, 1.0, 0.0, true, 42));
            assertThrows(IllegalArgumentException.class, 
                () -> new BaggingClassifier(
                    () -> new KNearestNeighbors(1), 10, 1.0, 1.5, true, 42));
        }
    }
    
    @Nested
    @DisplayName("Training Tests")
    class TrainingTests {
        
        @Test
        @DisplayName("Null X throws exception")
        void testNullX() {
            BaggingClassifier bc = new BaggingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> bc.train(null, new int[]{0, 1}));
        }
        
        @Test
        @DisplayName("Empty X throws exception")
        void testEmptyX() {
            BaggingClassifier bc = new BaggingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> bc.train(new double[0][], new int[0]));
        }
        
        @Test
        @DisplayName("Null y throws exception")
        void testNullY() {
            BaggingClassifier bc = new BaggingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> bc.train(new double[][]{{0}, {1}}, null));
        }
        
        @Test
        @DisplayName("Mismatched lengths throws exception")
        void testMismatchedLengths() {
            BaggingClassifier bc = new BaggingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> bc.train(new double[][]{{0}, {1}}, new int[]{0}));
        }
        
        @Test
        @DisplayName("Successful training")
        void testSuccessfulTraining() {
            BaggingClassifier bc = new BaggingClassifier(
                () -> new KNearestNeighbors(1),
                5,
                42
            );
            
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            bc.train(X, y);
            
            assertTrue(bc.isTrained());
            assertEquals(5, bc.getActualNEstimators());
            assertEquals(2, bc.getNumClasses());
        }
        
        @Test
        @DisplayName("Training with feature subsampling")
        void testFeatureSubsampling() {
            BaggingClassifier bc = new BaggingClassifier(
                () -> new DecisionTreeClassifier(),
                5,
                1.0,
                0.5,  // Use 50% of features
                true,
                42
            );
            
            double[][] X = {{0, 0, 0, 0}, {1, 1, 1, 1}, {10, 10, 10, 10}, {11, 11, 11, 11}};
            int[] y = {0, 0, 1, 1};
            
            bc.train(X, y);
            
            assertTrue(bc.isTrained());
        }
        
        @Test
        @DisplayName("Training without bootstrap")
        void testWithoutBootstrap() {
            BaggingClassifier bc = new BaggingClassifier(
                () -> new KNearestNeighbors(1),
                5,
                0.8,
                1.0,
                false,  // No bootstrap
                42
            );
            
            double[][] X = {{0}, {1}, {2}, {10}, {11}, {12}};
            int[] y = {0, 0, 0, 1, 1, 1};
            
            bc.train(X, y);
            
            assertTrue(bc.isTrained());
        }
    }
    
    @Nested
    @DisplayName("Prediction Tests")
    class PredictionTests {
        
        @Test
        @DisplayName("Predict before training throws exception")
        void testPredictBeforeTraining() {
            BaggingClassifier bc = new BaggingClassifier();
            assertThrows(IllegalStateException.class, 
                () -> bc.predict(new double[]{0}));
        }
        
        @Test
        @DisplayName("Predict with null throws exception")
        void testPredictNull() {
            BaggingClassifier bc = createTrainedBaggingClassifier();
            double[] x = null;
            assertThrows(IllegalArgumentException.class, () -> bc.predict(x));
        }
        
        @Test
        @DisplayName("Predict with wrong dimensions throws exception")
        void testPredictWrongDimensions() {
            BaggingClassifier bc = new BaggingClassifier(
                () -> new KNearestNeighbors(1), 5, 42);
            bc.train(new double[][]{{0, 0}, {10, 10}}, new int[]{0, 1});
            
            assertThrows(IllegalArgumentException.class, 
                () -> bc.predict(new double[]{0}));
        }
        
        @Test
        @DisplayName("Single prediction")
        void testSinglePrediction() {
            BaggingClassifier bc = createTrainedBaggingClassifier();
            
            int pred = bc.predict(new double[]{0});
            assertTrue(pred == 0 || pred == 1);
        }
        
        @Test
        @DisplayName("Batch prediction")
        void testBatchPrediction() {
            BaggingClassifier bc = createTrainedBaggingClassifier();
            
            double[][] X = {{0}, {5}, {10}};
            int[] predictions = bc.predict(X);
            
            assertEquals(3, predictions.length);
        }
        
        @Test
        @DisplayName("Batch prediction with null throws exception")
        void testBatchPredictionNull() {
            BaggingClassifier bc = createTrainedBaggingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> bc.predict((double[][]) null));
        }
        
        @Test
        @DisplayName("Batch prediction with empty throws exception")
        void testBatchPredictionEmpty() {
            BaggingClassifier bc = createTrainedBaggingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> bc.predict(new double[0][]));
        }
        
        @Test
        @DisplayName("Predictions are reasonable")
        void testReasonablePredictions() {
            double[][] X = new double[100][2];
            int[] y = new int[100];
            
            for (int i = 0; i < 50; i++) {
                X[i] = new double[]{i * 0.1, i * 0.1};
                y[i] = 0;
            }
            for (int i = 50; i < 100; i++) {
                X[i] = new double[]{10 + i * 0.1, 10 + i * 0.1};
                y[i] = 1;
            }
            
            BaggingClassifier bc = new BaggingClassifier(
                () -> new DecisionTreeClassifier.Builder().maxDepth(3).build(),
                10,
                42
            );
            bc.train(X, y);
            
            int[] predictions = bc.predict(X);
            
            int correct = 0;
            for (int i = 0; i < y.length; i++) {
                if (predictions[i] == y[i]) correct++;
            }
            double accuracy = (double) correct / y.length;
            
            assertTrue(accuracy > 0.7, "Should achieve reasonable accuracy: " + accuracy);
        }
    }
    
    @Nested
    @DisplayName("Probability Tests")
    class ProbabilityTests {
        
        @Test
        @DisplayName("Predict probability before training throws exception")
        void testPredictProbaBeforeTraining() {
            BaggingClassifier bc = new BaggingClassifier();
            assertThrows(IllegalStateException.class, 
                () -> bc.predictProba(new double[]{0}));
        }
        
        @Test
        @DisplayName("Predict probability with null throws exception")
        void testPredictProbaNull() {
            BaggingClassifier bc = createTrainedBaggingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> bc.predictProba(null));
        }
        
        @Test
        @DisplayName("Probabilities sum to 1")
        void testProbabilitiesSum() {
            BaggingClassifier bc = createTrainedBaggingClassifier();
            
            double[] proba = bc.predictProba(new double[]{5});
            
            double sum = 0;
            for (double p : proba) {
                sum += p;
                assertTrue(p >= 0 && p <= 1);
            }
            assertEquals(1.0, sum, 1e-10);
        }
    }
    
    @Nested
    @DisplayName("Getter Tests")
    class GetterTests {
        
        @Test
        @DisplayName("getClasses before training throws exception")
        void testGetClassesBeforeTraining() {
            BaggingClassifier bc = new BaggingClassifier();
            assertThrows(IllegalStateException.class, bc::getClasses);
        }
        
        @Test
        @DisplayName("getActualNEstimators before training throws exception")
        void testGetActualNEstimatorsBeforeTraining() {
            BaggingClassifier bc = new BaggingClassifier();
            assertThrows(IllegalStateException.class, bc::getActualNEstimators);
        }
        
        @Test
        @DisplayName("getClasses returns correct values")
        void testGetClasses() {
            BaggingClassifier bc = createTrainedBaggingClassifier();
            
            int[] classes = bc.getClasses();
            assertArrayEquals(new int[]{0, 1}, classes);
        }
        
        @Test
        @DisplayName("getNumClasses before training returns 0")
        void testGetNumClassesBeforeTraining() {
            BaggingClassifier bc = new BaggingClassifier();
            assertEquals(0, bc.getNumClasses());
        }
    }
    
    @Nested
    @DisplayName("Builder Tests")
    class BuilderTests {
        
        @Test
        @DisplayName("Builder creates classifier")
        void testBuilder() {
            BaggingClassifier bc = new BaggingClassifier.Builder()
                .baseEstimator(() -> new KNearestNeighbors(3))
                .nEstimators(15)
                .maxSamples(0.9)
                .maxFeatures(0.8)
                .bootstrap(true)
                .randomState(42)
                .build();
            
            assertEquals(15, bc.getNEstimators());
            assertEquals(0.9, bc.getMaxSamples(), 1e-10);
            assertEquals(0.8, bc.getMaxFeatures(), 1e-10);
            assertTrue(bc.isBootstrap());
        }
        
        @Test
        @DisplayName("Builder with defaults")
        void testBuilderDefaults() {
            BaggingClassifier bc = new BaggingClassifier.Builder().build();
            
            assertEquals(10, bc.getNEstimators());
            assertEquals(1.0, bc.getMaxSamples(), 1e-10);
            assertEquals(1.0, bc.getMaxFeatures(), 1e-10);
            assertTrue(bc.isBootstrap());
        }
    }
    
    @Nested
    @DisplayName("Reproducibility Tests")
    class ReproducibilityTests {
        
        @Test
        @DisplayName("Same seed produces same results")
        void testSameSeed() {
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            BaggingClassifier bc1 = new BaggingClassifier(
                () -> new KNearestNeighbors(1), 5, 42);
            BaggingClassifier bc2 = new BaggingClassifier(
                () -> new KNearestNeighbors(1), 5, 42);
            
            bc1.train(X, y);
            bc2.train(X, y);
            
            int[] pred1 = bc1.predict(X);
            int[] pred2 = bc2.predict(X);
            
            assertArrayEquals(pred1, pred2);
        }
    }
    
    @Nested
    @DisplayName("Multi-class Tests")
    class MultiClassTests {
        
        @Test
        @DisplayName("Three class classification")
        void testThreeClasses() {
            double[][] X = {
                {0}, {1}, {2},
                {10}, {11}, {12},
                {20}, {21}, {22}
            };
            int[] y = {0, 0, 0, 1, 1, 1, 2, 2, 2};
            
            BaggingClassifier bc = new BaggingClassifier(
                () -> new KNearestNeighbors(2), 10, 42);
            bc.train(X, y);
            
            assertEquals(3, bc.getNumClasses());
            
            int[] predictions = bc.predict(X);
            assertEquals(9, predictions.length);
        }
    }
    
    @Nested
    @DisplayName("Serialization Tests")
    class SerializationTests {
        
        @Test
        @DisplayName("Serialize and deserialize")
        @Disabled("Lambda expressions for classifier factories are not serializable")
        void testSerialization() throws IOException, ClassNotFoundException {
            BaggingClassifier bc = createTrainedBaggingClassifier();
            
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(bc);
            oos.close();
            
            ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bis);
            BaggingClassifier restored = (BaggingClassifier) ois.readObject();
            ois.close();
            
            assertTrue(restored.isTrained());
            assertEquals(bc.getNEstimators(), restored.getNEstimators());
        }
    }
    
    private BaggingClassifier createTrainedBaggingClassifier() {
        BaggingClassifier bc = new BaggingClassifier(
            () -> new KNearestNeighbors(1),
            5,
            42
        );
        
        double[][] X = {{0}, {1}, {10}, {11}};
        int[] y = {0, 0, 1, 1};
        bc.train(X, y);
        
        return bc;
    }
}
