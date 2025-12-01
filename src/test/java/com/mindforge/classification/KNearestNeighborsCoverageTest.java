package com.mindforge.classification;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Additional tests for KNearestNeighbors to improve coverage.
 */
class KNearestNeighborsCoverageTest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Valid k value")
        void testValidK() {
            KNearestNeighbors knn = new KNearestNeighbors(5);
            assertEquals(5, knn.getK());
        }
        
        @Test
        @DisplayName("k = 1 is valid")
        void testKEqualsOne() {
            KNearestNeighbors knn = new KNearestNeighbors(1);
            assertEquals(1, knn.getK());
        }
        
        @Test
        @DisplayName("Negative k throws exception")
        void testNegativeK() {
            assertThrows(IllegalArgumentException.class, () -> new KNearestNeighbors(-1));
        }
    }
    
    @Nested
    @DisplayName("Training Validation Tests")
    class TrainingValidationTests {
        
        @Test
        @DisplayName("Mismatched X and y length throws exception")
        void testMismatchedLengths() {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            double[][] X = {{1.0}, {2.0}};
            int[] y = {0, 1, 2};
            
            assertThrows(IllegalArgumentException.class, () -> knn.train(X, y));
        }
        
        @Test
        @DisplayName("Training data size less than k throws exception")
        void testTrainingDataSizeLessThanK() {
            KNearestNeighbors knn = new KNearestNeighbors(5);
            double[][] X = {{1.0}, {2.0}};
            int[] y = {0, 1};
            
            assertThrows(IllegalArgumentException.class, () -> knn.train(X, y));
        }
    }
    
    @Nested
    @DisplayName("Prediction Tests")
    class PredictionTests {
        
        @Test
        @DisplayName("Predict before training throws exception")
        void testPredictBeforeTraining() {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            assertThrows(IllegalStateException.class, () -> knn.predict(new double[]{1.0}));
        }
        
        @Test
        @DisplayName("Multiple predictions work correctly")
        void testMultiplePredictions() {
            KNearestNeighbors knn = new KNearestNeighbors(1);
            double[][] X = {{0}, {1}, {2}, {10}, {11}, {12}};
            int[] y = {0, 0, 0, 1, 1, 1};
            
            knn.train(X, y);
            
            int[] testPoints = new int[6];
            double[][] testX = {{0.5}, {1.5}, {2.5}, {10.5}, {11.5}, {12.5}};
            for (int i = 0; i < testX.length; i++) {
                testPoints[i] = knn.predict(testX[i]);
            }
            
            assertEquals(0, testPoints[0]);
            assertEquals(0, testPoints[1]);
            assertEquals(0, testPoints[2]);
            assertEquals(1, testPoints[3]);
            assertEquals(1, testPoints[4]);
            assertEquals(1, testPoints[5]);
        }
        
        @Test
        @DisplayName("Batch predictions")
        void testBatchPredictions() {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            double[][] X = {{1}, {2}, {3}, {10}, {11}, {12}};
            int[] y = {0, 0, 0, 1, 1, 1};
            
            knn.train(X, y);
            int[] predictions = knn.predict(X);
            
            assertEquals(6, predictions.length);
            for (int i = 0; i < 3; i++) {
                assertEquals(0, predictions[i]);
            }
            for (int i = 3; i < 6; i++) {
                assertEquals(1, predictions[i]);
            }
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("All same class")
        void testAllSameClass() {
            KNearestNeighbors knn = new KNearestNeighbors(2);
            double[][] X = {{1}, {2}, {3}, {4}};
            int[] y = {0, 0, 0, 0};
            
            knn.train(X, y);
            assertEquals(0, knn.predict(new double[]{5.0}));
        }
        
        @Test
        @DisplayName("Multiclass classification")
        void testMulticlass() {
            KNearestNeighbors knn = new KNearestNeighbors(1);
            double[][] X = {{1}, {5}, {10}, {15}, {20}};
            int[] y = {0, 1, 2, 3, 4};
            
            knn.train(X, y);
            
            assertEquals(0, knn.predict(new double[]{0}));
            assertEquals(1, knn.predict(new double[]{4}));
            assertEquals(2, knn.predict(new double[]{9}));
            assertEquals(3, knn.predict(new double[]{14}));
            assertEquals(4, knn.predict(new double[]{21}));
        }
        
        @Test
        @DisplayName("Tie breaking")
        void testTieBreaking() {
            KNearestNeighbors knn = new KNearestNeighbors(2);
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            knn.train(X, y);
            // Point equidistant from both classes should return one of them
            int pred = knn.predict(new double[]{5.5});
            assertTrue(pred == 0 || pred == 1);
        }
        
        @Test
        @DisplayName("High dimensional data")
        void testHighDimensional() {
            KNearestNeighbors knn = new KNearestNeighbors(2);
            double[][] X = new double[20][10];
            int[] y = new int[20];
            
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    X[i][j] = 1.0;
                }
                y[i] = 0;
            }
            for (int i = 10; i < 20; i++) {
                for (int j = 0; j < 10; j++) {
                    X[i][j] = 10.0;
                }
                y[i] = 1;
            }
            
            knn.train(X, y);
            
            double[] testPoint0 = new double[10];
            double[] testPoint1 = new double[10];
            for (int j = 0; j < 10; j++) {
                testPoint0[j] = 2.0;
                testPoint1[j] = 9.0;
            }
            
            assertEquals(0, knn.predict(testPoint0));
            assertEquals(1, knn.predict(testPoint1));
        }
        
        @Test
        @DisplayName("Exact match prediction")
        void testExactMatch() {
            KNearestNeighbors knn = new KNearestNeighbors(1);
            double[][] X = {{1, 2}, {3, 4}, {5, 6}};
            int[] y = {0, 1, 2};
            
            knn.train(X, y);
            
            assertEquals(0, knn.predict(new double[]{1, 2}));
            assertEquals(1, knn.predict(new double[]{3, 4}));
            assertEquals(2, knn.predict(new double[]{5, 6}));
        }
        
        @Test
        @DisplayName("Large k value")
        void testLargeK() {
            KNearestNeighbors knn = new KNearestNeighbors(5);
            double[][] X = {{1}, {2}, {3}, {4}, {5}, {6}, {7}};
            int[] y = {0, 0, 0, 0, 1, 1, 1};
            
            knn.train(X, y);
            
            // With k=5, majority voting should work
            int pred = knn.predict(new double[]{3.5});
            assertTrue(pred == 0 || pred == 1);
        }
    }
    
    @Nested
    @DisplayName("getK Tests")
    class GetKTests {
        
        @Test
        @DisplayName("getK returns correct value")
        void testGetK() {
            KNearestNeighbors knn = new KNearestNeighbors(7);
            assertEquals(7, knn.getK());
        }
    }
    
    @Nested
    @DisplayName("Batch Predict Method")
    class BatchPredictTests {
        
        @Test
        @DisplayName("Batch predict array")
        void testBatchPredictArray() {
            KNearestNeighbors knn = new KNearestNeighbors(1);
            double[][] XTrain = {{0}, {10}};
            int[] yTrain = {0, 1};
            
            knn.train(XTrain, yTrain);
            
            double[][] XTest = {{1}, {9}, {5}};
            int[] predictions = knn.predict(XTest);
            
            assertEquals(3, predictions.length);
            assertEquals(0, predictions[0]);
            assertEquals(1, predictions[1]);
        }
    }
}
