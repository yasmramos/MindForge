package com.mindforge.classification;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;

/**
 * Comprehensive tests for Perceptron classifier.
 */
class PerceptronTest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Default constructor")
        void testDefaultConstructor() {
            Perceptron p = new Perceptron();
            assertEquals(0.01, p.getLearningRate(), 1e-10);
            assertEquals(1000, p.getMaxIterations());
            assertFalse(p.isTrained());
        }
        
        @Test
        @DisplayName("Constructor with learning rate")
        void testLearningRateConstructor() {
            Perceptron p = new Perceptron(0.1);
            assertEquals(0.1, p.getLearningRate(), 1e-10);
        }
        
        @Test
        @DisplayName("Full constructor")
        void testFullConstructor() {
            Perceptron p = new Perceptron(0.05, 500, 42, false);
            assertEquals(0.05, p.getLearningRate(), 1e-10);
            assertEquals(500, p.getMaxIterations());
        }
        
        @Test
        @DisplayName("Invalid learning rate throws exception")
        void testInvalidLearningRate() {
            assertThrows(IllegalArgumentException.class, () -> new Perceptron(0));
            assertThrows(IllegalArgumentException.class, () -> new Perceptron(-0.1));
        }
        
        @Test
        @DisplayName("Invalid max iterations throws exception")
        void testInvalidMaxIterations() {
            assertThrows(IllegalArgumentException.class, () -> new Perceptron(0.1, 0, 42));
            assertThrows(IllegalArgumentException.class, () -> new Perceptron(0.1, -1, 42));
        }
    }
    
    @Nested
    @DisplayName("Binary Classification Tests")
    class BinaryClassificationTests {
        
        @Test
        @DisplayName("Linearly separable data")
        void testLinearlySeparable() {
            Perceptron p = new Perceptron(0.1, 100, 42);
            
            double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {5, 5}, {5, 6}, {6, 5}, {6, 6}};
            int[] y = {0, 0, 0, 0, 1, 1, 1, 1};
            
            p.train(X, y);
            
            assertTrue(p.isTrained());
            assertEquals(2, p.getNumClasses());
            
            // Test predictions
            int[] predictions = p.predict(X);
            int correct = 0;
            for (int i = 0; i < y.length; i++) {
                if (predictions[i] == y[i]) correct++;
            }
            assertTrue(correct >= 6, "Should classify at least 75% correctly");
        }
        
        @Test
        @DisplayName("Negative class labels")
        void testNegativeLabels() {
            Perceptron p = new Perceptron(0.1, 100, 42);
            
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {-1, -1, 1, 1};
            
            p.train(X, y);
            
            int[] classes = p.getClasses();
            assertArrayEquals(new int[]{-1, 1}, classes);
        }
        
        @Test
        @DisplayName("Single feature binary classification")
        void testSingleFeature() {
            Perceptron p = new Perceptron(0.5, 200, 42);
            
            double[][] X = {{0}, {1}, {2}, {10}, {11}, {12}};
            int[] y = {0, 0, 0, 1, 1, 1};
            
            p.train(X, y);
            
            assertEquals(0, p.predict(new double[]{-1}));
            assertEquals(1, p.predict(new double[]{15}));
        }
    }
    
    @Nested
    @DisplayName("Multi-class Classification Tests")
    class MultiClassTests {
        
        @Test
        @DisplayName("Three class classification")
        void testThreeClasses() {
            Perceptron p = new Perceptron(0.1, 200, 42);
            
            double[][] X = {
                {0, 0}, {0, 1}, {1, 0},
                {5, 0}, {5, 1}, {6, 0},
                {0, 5}, {1, 5}, {0, 6}
            };
            int[] y = {0, 0, 0, 1, 1, 1, 2, 2, 2};
            
            p.train(X, y);
            
            assertEquals(3, p.getNumClasses());
            assertArrayEquals(new int[]{0, 1, 2}, p.getClasses());
        }
        
        @Test
        @DisplayName("Four class classification")
        void testFourClasses() {
            Perceptron p = new Perceptron(0.1, 300, 42);
            
            double[][] X = new double[40][2];
            int[] y = new int[40];
            
            // Create 4 clusters
            for (int i = 0; i < 10; i++) {
                X[i] = new double[]{0 + i * 0.1, 0};
                y[i] = 0;
                X[10 + i] = new double[]{10 + i * 0.1, 0};
                y[10 + i] = 1;
                X[20 + i] = new double[]{0, 10 + i * 0.1};
                y[20 + i] = 2;
                X[30 + i] = new double[]{10 + i * 0.1, 10};
                y[30 + i] = 3;
            }
            
            p.train(X, y);
            
            assertEquals(4, p.getNumClasses());
        }
    }
    
    @Nested
    @DisplayName("Decision Function Tests")
    class DecisionFunctionTests {
        
        @Test
        @DisplayName("Decision function returns correct shape")
        void testDecisionFunctionShape() {
            Perceptron p = new Perceptron(0.1, 100, 42);
            
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            p.train(X, y);
            
            double[][] decisions = p.decisionFunction(X);
            assertEquals(4, decisions.length);
            assertEquals(1, decisions[0].length); // Binary: 1 decision value
        }
        
        @Test
        @DisplayName("Decision function before training throws exception")
        void testDecisionFunctionBeforeTraining() {
            Perceptron p = new Perceptron();
            assertThrows(IllegalStateException.class, 
                () -> p.decisionFunction(new double[][]{{1}}));
        }
    }
    
    @Nested
    @DisplayName("Weights Tests")
    class WeightsTests {
        
        @Test
        @DisplayName("Get weights after training")
        void testGetWeights() {
            Perceptron p = new Perceptron(0.1, 100, 42);
            
            double[][] X = {{0}, {10}};
            int[] y = {0, 1};
            
            p.train(X, y);
            
            double[][] weights = p.getWeights();
            assertNotNull(weights);
            assertEquals(1, weights.length); // Binary classification
            assertEquals(2, weights[0].length); // 1 feature + 1 bias
        }
        
        @Test
        @DisplayName("Get weights returns copy")
        void testGetWeightsReturnsCopy() {
            Perceptron p = new Perceptron(0.1, 100, 42);
            p.train(new double[][]{{0}, {10}}, new int[]{0, 1});
            
            double[][] w1 = p.getWeights();
            double[][] w2 = p.getWeights();
            
            assertNotSame(w1, w2);
            assertNotSame(w1[0], w2[0]);
        }
        
        @Test
        @DisplayName("Get weights before training throws exception")
        void testGetWeightsBeforeTraining() {
            Perceptron p = new Perceptron();
            assertThrows(IllegalStateException.class, p::getWeights);
        }
    }
    
    @Nested
    @DisplayName("Validation Tests")
    class ValidationTests {
        
        @Test
        @DisplayName("Null X throws exception")
        void testNullX() {
            Perceptron p = new Perceptron();
            assertThrows(IllegalArgumentException.class, 
                () -> p.train(null, new int[]{0, 1}));
        }
        
        @Test
        @DisplayName("Empty X throws exception")
        void testEmptyX() {
            Perceptron p = new Perceptron();
            assertThrows(IllegalArgumentException.class, 
                () -> p.train(new double[0][], new int[0]));
        }
        
        @Test
        @DisplayName("Null y throws exception")
        void testNullY() {
            Perceptron p = new Perceptron();
            assertThrows(IllegalArgumentException.class, 
                () -> p.train(new double[][]{{0}}, null));
        }
        
        @Test
        @DisplayName("Mismatched lengths throws exception")
        void testMismatchedLengths() {
            Perceptron p = new Perceptron();
            assertThrows(IllegalArgumentException.class, 
                () -> p.train(new double[][]{{0}, {1}}, new int[]{0}));
        }
        
        @Test
        @DisplayName("Single class throws exception")
        void testSingleClass() {
            Perceptron p = new Perceptron();
            assertThrows(IllegalArgumentException.class, 
                () -> p.train(new double[][]{{0}, {1}}, new int[]{0, 0}));
        }
        
        @Test
        @DisplayName("Predict before training throws exception")
        void testPredictBeforeTraining() {
            Perceptron p = new Perceptron();
            assertThrows(IllegalStateException.class, 
                () -> p.predict(new double[]{0}));
        }
        
        @Test
        @DisplayName("Predict with null throws exception")
        void testPredictNull() {
            Perceptron p = new Perceptron();
            p.train(new double[][]{{0}, {10}}, new int[]{0, 1});
            double[] x = null;
            assertThrows(IllegalArgumentException.class, () -> p.predict(x));
        }
        
        @Test
        @DisplayName("Predict with wrong dimensions throws exception")
        void testPredictWrongDimensions() {
            Perceptron p = new Perceptron();
            p.train(new double[][]{{0, 0}, {10, 10}}, new int[]{0, 1});
            assertThrows(IllegalArgumentException.class, 
                () -> p.predict(new double[]{0}));
        }
        
        @Test
        @DisplayName("Batch predict with null throws exception")
        void testBatchPredictNull() {
            Perceptron p = new Perceptron();
            p.train(new double[][]{{0}, {10}}, new int[]{0, 1});
            assertThrows(IllegalArgumentException.class, 
                () -> p.predict((double[][]) null));
        }
        
        @Test
        @DisplayName("Batch predict with empty throws exception")
        void testBatchPredictEmpty() {
            Perceptron p = new Perceptron();
            p.train(new double[][]{{0}, {10}}, new int[]{0, 1});
            assertThrows(IllegalArgumentException.class, 
                () -> p.predict(new double[0][]));
        }
        
        @Test
        @DisplayName("getClasses before training throws exception")
        void testGetClassesBeforeTraining() {
            Perceptron p = new Perceptron();
            assertThrows(IllegalStateException.class, p::getClasses);
        }
        
        @Test
        @DisplayName("getNumClasses before training returns 0")
        void testGetNumClassesBeforeTraining() {
            Perceptron p = new Perceptron();
            assertEquals(0, p.getNumClasses());
        }
    }
    
    @Nested
    @DisplayName("Reproducibility Tests")
    class ReproducibilityTests {
        
        @Test
        @DisplayName("Same seed produces same results")
        void testSameSeed() {
            double[][] X = {{0, 0}, {1, 1}, {5, 5}, {6, 6}};
            int[] y = {0, 0, 1, 1};
            
            Perceptron p1 = new Perceptron(0.1, 100, 42, true);
            Perceptron p2 = new Perceptron(0.1, 100, 42, true);
            
            p1.train(X, y);
            p2.train(X, y);
            
            int[] pred1 = p1.predict(X);
            int[] pred2 = p2.predict(X);
            
            assertArrayEquals(pred1, pred2);
        }
    }
    
    @Nested
    @DisplayName("Shuffle Option Tests")
    class ShuffleTests {
        
        @Test
        @DisplayName("Training without shuffle")
        void testWithoutShuffle() {
            Perceptron p = new Perceptron(0.1, 100, 42, false);
            
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            p.train(X, y);
            
            assertTrue(p.isTrained());
        }
    }
    
    @Nested
    @DisplayName("Serialization Tests")
    class SerializationTests {
        
        @Test
        @DisplayName("Serialize and deserialize")
        void testSerialization() throws IOException, ClassNotFoundException {
            Perceptron p = new Perceptron(0.1, 100, 42);
            p.train(new double[][]{{0}, {10}}, new int[]{0, 1});
            
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(p);
            oos.close();
            
            ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bis);
            Perceptron restored = (Perceptron) ois.readObject();
            ois.close();
            
            assertEquals(p.getLearningRate(), restored.getLearningRate(), 1e-10);
            assertTrue(restored.isTrained());
            assertEquals(p.predict(new double[]{5}), restored.predict(new double[]{5}));
        }
    }
}
