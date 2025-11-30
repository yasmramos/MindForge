package com.mindforge.preprocessing;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DataSplitTest {

    @Test
    void testBasicTrainTestSplit() {
        double[][] X = {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 4.0},
            {4.0, 5.0}
        };
        int[] y = {0, 0, 1, 1};

        DataSplit.TrainTestSplit split = DataSplit.trainTestSplit(X, y, 0.25, false, null);

        assertEquals(3, split.XTrain.length);
        assertEquals(1, split.XTest.length);
        assertEquals(3, split.yTrain.length);
        assertEquals(1, split.yTest.length);
    }

    @Test
    void testTrainTestSplitWithShuffle() {
        double[][] X = {
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
        };
        int[] y = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        DataSplit.TrainTestSplit split = DataSplit.trainTestSplit(X, y, 0.2, true, 42);

        assertEquals(8, split.XTrain.length);
        assertEquals(2, split.XTest.length);
        assertEquals(8, split.yTrain.length);
        assertEquals(2, split.yTest.length);

        // With seed 42, the split should be reproducible
        DataSplit.TrainTestSplit split2 = DataSplit.trainTestSplit(X, y, 0.2, true, 42);
        
        assertArrayEquals(split.yTrain, split2.yTrain);
        assertArrayEquals(split.yTest, split2.yTest);
    }

    @Test
    void testRegressionTrainTestSplit() {
        double[][] X = {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 4.0},
            {4.0, 5.0}
        };
        double[] y = {1.5, 2.5, 3.5, 4.5};

        DataSplit.TrainTestSplitRegression split = DataSplit.trainTestSplit(X, y, 0.5, false, null);

        assertEquals(2, split.XTrain.length);
        assertEquals(2, split.XTest.length);
        assertEquals(2, split.yTrain.length);
        assertEquals(2, split.yTest.length);
    }

    @Test
    void testStratifiedSplit() {
        double[][] X = new double[100][2];
        int[] y = new int[100];
        
        // Create balanced dataset: 50 samples of class 0, 50 of class 1
        for (int i = 0; i < 100; i++) {
            X[i][0] = i;
            X[i][1] = i * 2;
            y[i] = i < 50 ? 0 : 1;
        }

        DataSplit.TrainTestSplit split = DataSplit.stratifiedTrainTestSplit(X, y, 0.2, 42);

        // Count classes in train and test sets
        int trainClass0 = 0, trainClass1 = 0;
        int testClass0 = 0, testClass1 = 0;

        for (int label : split.yTrain) {
            if (label == 0) trainClass0++;
            else trainClass1++;
        }

        for (int label : split.yTest) {
            if (label == 0) testClass0++;
            else testClass1++;
        }

        // Check stratification: proportions should be maintained
        assertEquals(40, trainClass0);  // 80% of 50
        assertEquals(40, trainClass1);  // 80% of 50
        assertEquals(10, testClass0);   // 20% of 50
        assertEquals(10, testClass1);   // 20% of 50
    }

    @Test
    void testStratifiedSplitImbalanced() {
        double[][] X = new double[100][2];
        int[] y = new int[100];
        
        // Create imbalanced dataset: 30 samples of class 0, 70 of class 1
        for (int i = 0; i < 100; i++) {
            X[i][0] = i;
            X[i][1] = i * 2;
            y[i] = i < 30 ? 0 : 1;
        }

        DataSplit.TrainTestSplit split = DataSplit.stratifiedTrainTestSplit(X, y, 0.25, 42);

        // Count classes
        int trainClass0 = 0, trainClass1 = 0;
        int testClass0 = 0, testClass1 = 0;

        for (int label : split.yTrain) {
            if (label == 0) trainClass0++;
            else trainClass1++;
        }

        for (int label : split.yTest) {
            if (label == 0) testClass0++;
            else testClass1++;
        }

        // Check stratification with imbalanced data
        assertEquals(22, trainClass0, 1);  // ~75% of 30
        assertEquals(53, trainClass1, 1);  // ~75% of 70
        assertEquals(8, testClass0, 1);    // ~25% of 30
        assertEquals(17, testClass1, 1);   // ~25% of 70
    }

    @Test
    void testInvalidTestSize() {
        double[][] X = {{1.0}, {2.0}};
        int[] y = {0, 1};

        assertThrows(IllegalArgumentException.class, 
            () -> DataSplit.trainTestSplit(X, y, 0.0, false, null));
        assertThrows(IllegalArgumentException.class, 
            () -> DataSplit.trainTestSplit(X, y, 1.0, false, null));
        assertThrows(IllegalArgumentException.class, 
            () -> DataSplit.trainTestSplit(X, y, -0.1, false, null));
        assertThrows(IllegalArgumentException.class, 
            () -> DataSplit.trainTestSplit(X, y, 1.5, false, null));
    }

    @Test
    void testMismatchedArrays() {
        double[][] X = {{1.0}, {2.0}, {3.0}};
        int[] y = {0, 1};

        assertThrows(IllegalArgumentException.class, 
            () -> DataSplit.trainTestSplit(X, y, 0.25, false, null));
    }

    @Test
    void testDataNotModified() {
        double[][] X = {
            {1.0, 2.0},
            {3.0, 4.0}
        };
        int[] y = {0, 1};

        double[][] XOriginal = {
            {1.0, 2.0},
            {3.0, 4.0}
        };
        int[] yOriginal = {0, 1};

        DataSplit.trainTestSplit(X, y, 0.5, true, 42);

        // Original data should not be modified
        assertArrayEquals(XOriginal[0], X[0]);
        assertArrayEquals(XOriginal[1], X[1]);
        assertArrayEquals(yOriginal, y);
    }
}
