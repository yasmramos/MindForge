package com.mindforge.preprocessing;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SimpleImputerTest {

    private static final double DELTA = 1e-6;

    @Test
    void testMeanImputation() {
        double[][] data = {
            {1.0, 2.0},
            {Double.NaN, 3.0},
            {7.0, Double.NaN},
            {4.0, 6.0}
        };

        SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.MEAN);
        double[][] filled = imputer.fitTransform(data);

        // First column mean: (1 + 7 + 4) / 3 = 4.0
        // Second column mean: (2 + 3 + 6) / 3 = 3.666...
        assertEquals(1.0, filled[0][0], DELTA);
        assertEquals(2.0, filled[0][1], DELTA);
        assertEquals(4.0, filled[1][0], DELTA);  // Imputed
        assertEquals(3.0, filled[1][1], DELTA);
        assertEquals(7.0, filled[2][0], DELTA);
        assertEquals(3.666666, filled[2][1], 0.001);  // Imputed
        assertEquals(4.0, filled[3][0], DELTA);
        assertEquals(6.0, filled[3][1], DELTA);
    }

    @Test
    void testMedianImputation() {
        double[][] data = {
            {1.0, 2.0},
            {Double.NaN, 3.0},
            {7.0, Double.NaN},
            {4.0, 6.0},
            {10.0, 10.0}
        };

        SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.MEDIAN);
        double[][] filled = imputer.fitTransform(data);

        // First column values: [1, 7, 4, 10] -> median = (4 + 7) / 2 = 5.5
        // Second column values: [2, 3, 6, 10] -> median = (3 + 6) / 2 = 4.5
        assertEquals(5.5, filled[1][0], DELTA);  // Imputed
        assertEquals(4.5, filled[2][1], DELTA);  // Imputed
    }

    @Test
    void testMostFrequentImputation() {
        double[][] data = {
            {1.0, 5.0},
            {2.0, 5.0},
            {Double.NaN, 5.0},
            {2.0, Double.NaN},
            {2.0, 10.0}
        };

        SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.MOST_FREQUENT);
        double[][] filled = imputer.fitTransform(data);

        // First column: 2.0 appears 3 times (most frequent)
        // Second column: 5.0 appears 3 times (most frequent)
        assertEquals(2.0, filled[2][0], DELTA);  // Imputed
        assertEquals(5.0, filled[3][1], DELTA);  // Imputed
    }

    @Test
    void testConstantImputation() {
        double[][] data = {
            {1.0, Double.NaN},
            {Double.NaN, 3.0},
            {7.0, 6.0}
        };

        SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.CONSTANT, 999.0);
        double[][] filled = imputer.fitTransform(data);

        assertEquals(999.0, filled[0][1], DELTA);  // Imputed
        assertEquals(999.0, filled[1][0], DELTA);  // Imputed
        assertEquals(1.0, filled[0][0], DELTA);
        assertEquals(3.0, filled[1][1], DELTA);
        assertEquals(7.0, filled[2][0], DELTA);
        assertEquals(6.0, filled[2][1], DELTA);
    }

    @Test
    void testAllMissingColumn() {
        double[][] data = {
            {Double.NaN, 1.0},
            {Double.NaN, 2.0},
            {Double.NaN, 3.0}
        };

        SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.MEAN);
        double[][] filled = imputer.fitTransform(data);

        // All missing values should be filled with 0.0 (default for all-missing column)
        assertEquals(0.0, filled[0][0], DELTA);
        assertEquals(0.0, filled[1][0], DELTA);
        assertEquals(0.0, filled[2][0], DELTA);
    }

    @Test
    void testNoMissingValues() {
        double[][] data = {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        };

        SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.MEAN);
        double[][] filled = imputer.fitTransform(data);

        // Data should remain unchanged
        for (int i = 0; i < data.length; i++) {
            assertArrayEquals(data[i], filled[i], DELTA);
        }
    }

    @Test
    void testFitAndTransformSeparately() {
        double[][] trainData = {
            {1.0, 2.0},
            {Double.NaN, 3.0},
            {7.0, 6.0}
        };

        double[][] testData = {
            {Double.NaN, 5.0},
            {4.0, Double.NaN}
        };

        SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.MEAN);
        imputer.fit(trainData);
        
        // Mean from training data: [4.0, 3.666...]
        double[] stats = imputer.getStatistics();
        assertEquals(4.0, stats[0], DELTA);
        assertEquals(3.666666, stats[1], 0.001);

        double[][] filledTest = imputer.transform(testData);
        
        // Test data should be filled with training statistics
        assertEquals(4.0, filledTest[0][0], DELTA);  // Filled with train mean
        assertEquals(3.666666, filledTest[1][1], 0.001);  // Filled with train mean
    }

    @Test
    void testNotFittedError() {
        SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.MEAN);
        double[][] data = {{1.0, 2.0}};

        assertThrows(IllegalStateException.class, () -> imputer.transform(data));
        assertThrows(IllegalStateException.class, () -> imputer.getStatistics());
    }

    @Test
    void testIsFitted() {
        SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.MEAN);
        assertFalse(imputer.isFitted());

        double[][] data = {{1.0, 2.0}};
        imputer.fit(data);
        assertTrue(imputer.isFitted());
    }
}
