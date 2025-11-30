package com.mindforge.preprocessing;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MinMaxScalerTest {

    private static final double DELTA = 1e-6;

    @Test
    void testDefaultScaling() {
        double[][] data = {
            {1.0, 2.0},
            {2.0, 4.0},
            {3.0, 6.0}
        };

        MinMaxScaler scaler = new MinMaxScaler();
        double[][] scaled = scaler.fitTransform(data);

        // First column: [1, 2, 3] -> [0, 0.5, 1]
        // Second column: [2, 4, 6] -> [0, 0.5, 1]
        assertEquals(0.0, scaled[0][0], DELTA);
        assertEquals(0.0, scaled[0][1], DELTA);
        assertEquals(0.5, scaled[1][0], DELTA);
        assertEquals(0.5, scaled[1][1], DELTA);
        assertEquals(1.0, scaled[2][0], DELTA);
        assertEquals(1.0, scaled[2][1], DELTA);
    }

    @Test
    void testCustomRangeScaling() {
        double[][] data = {
            {0.0},
            {5.0},
            {10.0}
        };

        MinMaxScaler scaler = new MinMaxScaler(-1.0, 1.0);
        double[][] scaled = scaler.fitTransform(data);

        assertEquals(-1.0, scaled[0][0], DELTA);
        assertEquals(0.0, scaled[1][0], DELTA);
        assertEquals(1.0, scaled[2][0], DELTA);
    }

    @Test
    void testInverseTransform() {
        double[][] data = {
            {1.0, 2.0},
            {2.0, 4.0},
            {3.0, 6.0}
        };

        MinMaxScaler scaler = new MinMaxScaler();
        double[][] scaled = scaler.fitTransform(data);
        double[][] inverse = scaler.inverseTransform(scaled);

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                assertEquals(data[i][j], inverse[i][j], DELTA);
            }
        }
    }

    @Test
    void testConstantFeature() {
        double[][] data = {
            {5.0, 1.0},
            {5.0, 2.0},
            {5.0, 3.0}
        };

        MinMaxScaler scaler = new MinMaxScaler();
        double[][] scaled = scaler.fitTransform(data);

        // Constant feature should be scaled to the middle of the range
        assertEquals(0.5, scaled[0][0], DELTA);
        assertEquals(0.5, scaled[1][0], DELTA);
        assertEquals(0.5, scaled[2][0], DELTA);

        // Second feature should scale normally
        assertEquals(0.0, scaled[0][1], DELTA);
        assertEquals(0.5, scaled[1][1], DELTA);
        assertEquals(1.0, scaled[2][1], DELTA);
    }

    @Test
    void testGetFeatureRange() {
        double[][] data = {
            {1.0, 10.0},
            {2.0, 20.0},
            {3.0, 30.0}
        };

        MinMaxScaler scaler = new MinMaxScaler();
        scaler.fit(data);

        double[] min = scaler.getFeatureMin();
        double[] max = scaler.getFeatureMax();

        assertArrayEquals(new double[]{1.0, 10.0}, min, DELTA);
        assertArrayEquals(new double[]{3.0, 30.0}, max, DELTA);
    }

    @Test
    void testNotFittedError() {
        MinMaxScaler scaler = new MinMaxScaler();
        double[][] data = {{1.0, 2.0}};

        assertThrows(IllegalStateException.class, () -> scaler.transform(data));
        assertThrows(IllegalStateException.class, () -> scaler.inverseTransform(data));
        assertThrows(IllegalStateException.class, () -> scaler.getFeatureMin());
        assertThrows(IllegalStateException.class, () -> scaler.getFeatureMax());
    }

    @Test
    void testIsFitted() {
        MinMaxScaler scaler = new MinMaxScaler();
        assertFalse(scaler.isFitted());

        double[][] data = {{1.0, 2.0}};
        scaler.fit(data);
        assertTrue(scaler.isFitted());
    }
}
