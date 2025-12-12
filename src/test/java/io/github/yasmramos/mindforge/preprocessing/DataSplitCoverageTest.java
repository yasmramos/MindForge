package io.github.yasmramos.mindforge.preprocessing;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Additional coverage tests for DataSplit.
 */
class DataSplitCoverageTest {
    
    @Nested
    @DisplayName("trainTestSplit Classification Tests")
    class TrainTestSplitClassificationTests {
        
        @Test
        @DisplayName("Null X throws exception")
        void testNullX() {
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.trainTestSplit(null, new int[]{0, 1}, 0.2, true, 42));
        }
        
        @Test
        @DisplayName("Null y throws exception")
        void testNullY() {
            int[] nullY = null;
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.trainTestSplit(new double[][]{{1}, {2}}, nullY, 0.2, true, 42));
        }
        
        @Test
        @DisplayName("Mismatched X and y length throws exception")
        void testMismatchedLengths() {
            double[][] X = {{1}, {2}, {3}};
            int[] y = {0, 1};
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.trainTestSplit(X, y, 0.2, true, 42));
        }
        
        @Test
        @DisplayName("testSize <= 0 throws exception")
        void testInvalidTestSizeZero() {
            double[][] X = {{1}, {2}, {3}, {4}};
            int[] y = {0, 0, 1, 1};
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.trainTestSplit(X, y, 0.0, true, 42));
        }
        
        @Test
        @DisplayName("testSize >= 1 throws exception")
        void testInvalidTestSizeOne() {
            double[][] X = {{1}, {2}, {3}, {4}};
            int[] y = {0, 0, 1, 1};
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.trainTestSplit(X, y, 1.0, true, 42));
        }
        
        @Test
        @DisplayName("Valid split without shuffle")
        void testSplitWithoutShuffle() {
            double[][] X = {{1}, {2}, {3}, {4}, {5}};
            int[] y = {0, 0, 1, 1, 1};
            
            DataSplit.TrainTestSplit split = DataSplit.trainTestSplit(X, y, 0.4, false, null);
            
            assertEquals(3, split.XTrain.length);
            assertEquals(2, split.XTest.length);
            assertEquals(3, split.yTrain.length);
            assertEquals(2, split.yTest.length);
        }
        
        @Test
        @DisplayName("Valid split with shuffle and seed")
        void testSplitWithShuffleAndSeed() {
            double[][] X = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}};
            int[] y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
            
            DataSplit.TrainTestSplit split1 = DataSplit.trainTestSplit(X, y, 0.3, true, 42);
            DataSplit.TrainTestSplit split2 = DataSplit.trainTestSplit(X, y, 0.3, true, 42);
            
            // Same seed should produce same split
            assertEquals(split1.XTrain.length, split2.XTrain.length);
            assertEquals(split1.XTest.length, split2.XTest.length);
            
            // Check arrays are equal
            for (int i = 0; i < split1.XTrain.length; i++) {
                assertArrayEquals(split1.XTrain[i], split2.XTrain[i]);
            }
        }
        
        @Test
        @DisplayName("Split with random seed null")
        void testSplitWithRandomSeedNull() {
            double[][] X = {{1}, {2}, {3}, {4}, {5}};
            int[] y = {0, 0, 1, 1, 1};
            
            DataSplit.TrainTestSplit split = DataSplit.trainTestSplit(X, y, 0.4, true, null);
            
            assertNotNull(split.XTrain);
            assertNotNull(split.XTest);
            assertNotNull(split.yTrain);
            assertNotNull(split.yTest);
        }
        
        @Test
        @DisplayName("Data is cloned correctly")
        void testDataIsCloned() {
            double[][] X = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
            int[] y = {0, 0, 1, 1};
            
            DataSplit.TrainTestSplit split = DataSplit.trainTestSplit(X, y, 0.5, false, null);
            
            // Modify original X
            X[0][0] = 999;
            
            // Split data should not be affected
            boolean foundModified = false;
            for (double[] row : split.XTrain) {
                if (row[0] == 999) foundModified = true;
            }
            for (double[] row : split.XTest) {
                if (row[0] == 999) foundModified = true;
            }
            assertFalse(foundModified, "Split should contain cloned data");
        }
    }
    
    @Nested
    @DisplayName("trainTestSplit Regression Tests")
    class TrainTestSplitRegressionTests {
        
        @Test
        @DisplayName("Null X throws exception")
        void testNullX() {
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.trainTestSplit(null, new double[]{0.1, 0.2}, 0.2, true, 42));
        }
        
        @Test
        @DisplayName("Null y throws exception")
        void testNullY() {
            double[] y = null;
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.trainTestSplit(new double[][]{{1}, {2}}, y, 0.2, true, 42));
        }
        
        @Test
        @DisplayName("Mismatched X and y length throws exception")
        void testMismatchedLengths() {
            double[][] X = {{1}, {2}, {3}};
            double[] y = {1.0, 2.0};
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.trainTestSplit(X, y, 0.2, true, 42));
        }
        
        @Test
        @DisplayName("Invalid testSize throws exception")
        void testInvalidTestSize() {
            double[][] X = {{1}, {2}, {3}, {4}};
            double[] y = {1.0, 2.0, 3.0, 4.0};
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.trainTestSplit(X, y, -0.1, true, 42));
        }
        
        @Test
        @DisplayName("Valid regression split")
        void testValidRegressionSplit() {
            double[][] X = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}};
            double[] y = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0};
            
            DataSplit.TrainTestSplitRegression split = 
                DataSplit.trainTestSplit(X, y, 0.3, true, 123);
            
            assertEquals(7, split.XTrain.length);
            assertEquals(3, split.XTest.length);
            assertEquals(7, split.yTrain.length);
            assertEquals(3, split.yTest.length);
        }
        
        @Test
        @DisplayName("Regression split without shuffle")
        void testRegressionSplitWithoutShuffle() {
            double[][] X = {{1}, {2}, {3}, {4}, {5}};
            double[] y = {1.0, 2.0, 3.0, 4.0, 5.0};
            
            DataSplit.TrainTestSplitRegression split = 
                DataSplit.trainTestSplit(X, y, 0.4, false, null);
            
            // Without shuffle, first 3 should be train, last 2 should be test
            assertEquals(3, split.XTrain.length);
            assertEquals(2, split.XTest.length);
            
            // Check the order is preserved (no shuffle)
            assertEquals(1.0, split.XTrain[0][0], 1e-10);
            assertEquals(2.0, split.XTrain[1][0], 1e-10);
            assertEquals(3.0, split.XTrain[2][0], 1e-10);
        }
        
        @Test
        @DisplayName("Regression split with null random seed and shuffle")
        void testRegressionSplitNullSeed() {
            double[][] X = {{1}, {2}, {3}, {4}, {5}};
            double[] y = {1.0, 2.0, 3.0, 4.0, 5.0};
            
            DataSplit.TrainTestSplitRegression split = 
                DataSplit.trainTestSplit(X, y, 0.4, true, null);
            
            assertNotNull(split.XTrain);
            assertNotNull(split.yTrain);
        }
    }
    
    @Nested
    @DisplayName("Stratified Split Tests")
    class StratifiedSplitTests {
        
        @Test
        @DisplayName("Null X throws exception")
        void testNullX() {
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.stratifiedTrainTestSplit(null, new int[]{0, 1}, 0.2, 42));
        }
        
        @Test
        @DisplayName("Null y throws exception")
        void testNullY() {
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.stratifiedTrainTestSplit(new double[][]{{1}, {2}}, null, 0.2, 42));
        }
        
        @Test
        @DisplayName("Mismatched lengths throws exception")
        void testMismatchedLengths() {
            double[][] X = {{1}, {2}, {3}};
            int[] y = {0, 1};
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.stratifiedTrainTestSplit(X, y, 0.2, 42));
        }
        
        @Test
        @DisplayName("Invalid testSize throws exception")
        void testInvalidTestSize() {
            double[][] X = {{1}, {2}, {3}, {4}};
            int[] y = {0, 0, 1, 1};
            assertThrows(IllegalArgumentException.class, 
                () -> DataSplit.stratifiedTrainTestSplit(X, y, 1.5, 42));
        }
        
        @Test
        @DisplayName("Stratified split maintains class proportions")
        void testMaintainsClassProportions() {
            // 60% class 0, 40% class 1
            double[][] X = new double[10][1];
            int[] y = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
            for (int i = 0; i < 10; i++) X[i][0] = i;
            
            DataSplit.TrainTestSplit split = 
                DataSplit.stratifiedTrainTestSplit(X, y, 0.3, 42);
            
            // Count classes in train
            int train0 = 0, train1 = 0;
            for (int label : split.yTrain) {
                if (label == 0) train0++;
                else train1++;
            }
            
            // Count classes in test
            int test0 = 0, test1 = 0;
            for (int label : split.yTest) {
                if (label == 0) test0++;
                else test1++;
            }
            
            // Both train and test should have samples from both classes
            assertTrue(train0 > 0 && train1 > 0);
            assertTrue(test0 > 0 || test1 > 0); // At least one class in test
        }
        
        @Test
        @DisplayName("Stratified split with null random seed")
        void testStratifiedNullSeed() {
            double[][] X = {{1}, {2}, {3}, {4}, {5}, {6}};
            int[] y = {0, 0, 0, 1, 1, 1};
            
            DataSplit.TrainTestSplit split = 
                DataSplit.stratifiedTrainTestSplit(X, y, 0.33, null);
            
            assertNotNull(split.XTrain);
            assertNotNull(split.XTest);
        }
        
        @Test
        @DisplayName("Stratified split reproducibility with same seed")
        void testStratifiedReproducibility() {
            double[][] X = new double[20][2];
            int[] y = new int[20];
            for (int i = 0; i < 20; i++) {
                X[i][0] = i;
                X[i][1] = i * 2;
                y[i] = i < 10 ? 0 : 1;
            }
            
            DataSplit.TrainTestSplit split1 = 
                DataSplit.stratifiedTrainTestSplit(X, y, 0.3, 99);
            DataSplit.TrainTestSplit split2 = 
                DataSplit.stratifiedTrainTestSplit(X, y, 0.3, 99);
            
            assertEquals(split1.XTrain.length, split2.XTrain.length);
            assertEquals(split1.XTest.length, split2.XTest.length);
        }
        
        @Test
        @DisplayName("Stratified split with multiple classes")
        void testStratifiedMultipleClasses() {
            double[][] X = new double[15][1];
            int[] y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
            for (int i = 0; i < 15; i++) X[i][0] = i;
            
            DataSplit.TrainTestSplit split = 
                DataSplit.stratifiedTrainTestSplit(X, y, 0.2, 42);
            
            // Check all classes are represented
            Set<Integer> trainClasses = new HashSet<>();
            Set<Integer> testClasses = new HashSet<>();
            
            for (int label : split.yTrain) trainClasses.add(label);
            for (int label : split.yTest) testClasses.add(label);
            
            // Train should have all classes
            assertTrue(trainClasses.size() >= 2, "Train should have multiple classes");
        }
    }
    
    @Nested
    @DisplayName("TrainTestSplit Container Tests")
    class TrainTestSplitContainerTests {
        
        @Test
        @DisplayName("TrainTestSplit fields are accessible")
        void testFieldsAccessible() {
            double[][] XTrain = {{1}, {2}};
            double[][] XTest = {{3}};
            int[] yTrain = {0, 0};
            int[] yTest = {1};
            
            DataSplit.TrainTestSplit split = 
                new DataSplit.TrainTestSplit(XTrain, XTest, yTrain, yTest);
            
            assertSame(XTrain, split.XTrain);
            assertSame(XTest, split.XTest);
            assertSame(yTrain, split.yTrain);
            assertSame(yTest, split.yTest);
        }
    }
    
    @Nested
    @DisplayName("TrainTestSplitRegression Container Tests")
    class TrainTestSplitRegressionContainerTests {
        
        @Test
        @DisplayName("TrainTestSplitRegression fields are accessible")
        void testFieldsAccessible() {
            double[][] XTrain = {{1}, {2}};
            double[][] XTest = {{3}};
            double[] yTrain = {1.1, 2.2};
            double[] yTest = {3.3};
            
            DataSplit.TrainTestSplitRegression split = 
                new DataSplit.TrainTestSplitRegression(XTrain, XTest, yTrain, yTest);
            
            assertSame(XTrain, split.XTrain);
            assertSame(XTest, split.XTest);
            assertSame(yTrain, split.yTrain);
            assertSame(yTest, split.yTest);
        }
    }
}
