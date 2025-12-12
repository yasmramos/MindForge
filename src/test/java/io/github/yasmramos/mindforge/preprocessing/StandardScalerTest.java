package io.github.yasmramos.mindforge.preprocessing;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class StandardScalerTest {

    private static final double DELTA = 1e-6;

    @Test
    void testStandardScaling() {
        double[][] data = {
            {0.0, 0.0},
            {1.0, 1.0},
            {2.0, 2.0}
        };

        StandardScaler scaler = new StandardScaler();
        double[][] scaled = scaler.fitTransform(data);

        // Mean should be [1.0, 1.0], std should be [0.816..., 0.816...]
        // First row: (0 - 1) / 0.816... = -1.224...
        assertEquals(-1.224744, scaled[0][0], 0.001);
        assertEquals(-1.224744, scaled[0][1], 0.001);
        assertEquals(0.0, scaled[1][0], DELTA);
        assertEquals(0.0, scaled[1][1], DELTA);
        assertEquals(1.224744, scaled[2][0], 0.001);
        assertEquals(1.224744, scaled[2][1], 0.001);
    }

    @Test
    void testMeanAndStd() {
        double[][] data = {
            {1.0, 10.0},
            {2.0, 20.0},
            {3.0, 30.0}
        };

        StandardScaler scaler = new StandardScaler();
        scaler.fit(data);

        double[] mean = scaler.getMean();
        double[] std = scaler.getStd();

        assertArrayEquals(new double[]{2.0, 20.0}, mean, DELTA);
        assertEquals(0.816496, std[0], 0.001);
        assertEquals(8.164965, std[1], 0.001);
    }

    @Test
    void testInverseTransform() {
        double[][] data = {
            {1.0, 2.0},
            {2.0, 4.0},
            {3.0, 6.0}
        };

        StandardScaler scaler = new StandardScaler();
        double[][] scaled = scaler.fitTransform(data);
        double[][] inverse = scaler.inverseTransform(scaled);

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                assertEquals(data[i][j], inverse[i][j], DELTA);
            }
        }
    }

    @Test
    void testWithMeanOnly() {
        double[][] data = {
            {1.0, 2.0},
            {2.0, 4.0},
            {3.0, 6.0}
        };

        StandardScaler scaler = new StandardScaler(true, false);
        double[][] scaled = scaler.fitTransform(data);

        // Mean should be [2.0, 4.0]
        // Only centering, no scaling
        assertEquals(-1.0, scaled[0][0], DELTA);
        assertEquals(-2.0, scaled[0][1], DELTA);
        assertEquals(0.0, scaled[1][0], DELTA);
        assertEquals(0.0, scaled[1][1], DELTA);
        assertEquals(1.0, scaled[2][0], DELTA);
        assertEquals(2.0, scaled[2][1], DELTA);
    }

    @Test
    void testWithStdOnly() {
        double[][] data = {
            {0.0, 0.0},
            {1.0, 2.0},
            {2.0, 4.0}
        };

        StandardScaler scaler = new StandardScaler(false, true);
        double[][] scaled = scaler.fitTransform(data);

        // No centering, only scaling by std
        // Std should be [0.816..., 1.632...]
        assertEquals(0.0, scaled[0][0], DELTA);
        assertEquals(0.0, scaled[0][1], DELTA);
        assertEquals(1.224744, scaled[1][0], 0.001);
        assertEquals(1.224744, scaled[1][1], 0.001);
        assertEquals(2.449489, scaled[2][0], 0.001);
        assertEquals(2.449489, scaled[2][1], 0.001);
    }

    @Test
    void testConstantFeature() {
        double[][] data = {
            {5.0, 1.0},
            {5.0, 2.0},
            {5.0, 3.0}
        };

        StandardScaler scaler = new StandardScaler();
        double[][] scaled = scaler.fitTransform(data);

        // Constant feature (std = 0) should result in 0 after centering
        assertEquals(0.0, scaled[0][0], DELTA);
        assertEquals(0.0, scaled[1][0], DELTA);
        assertEquals(0.0, scaled[2][0], DELTA);

        // Second feature should scale normally
        assertEquals(-1.224744, scaled[0][1], 0.001);
        assertEquals(0.0, scaled[1][1], DELTA);
        assertEquals(1.224744, scaled[2][1], 0.001);
    }

    @Test
    void testNotFittedError() {
        StandardScaler scaler = new StandardScaler();
        double[][] data = {{1.0, 2.0}};

        assertThrows(IllegalStateException.class, () -> scaler.transform(data));
        assertThrows(IllegalStateException.class, () -> scaler.inverseTransform(data));
        assertThrows(IllegalStateException.class, () -> scaler.getMean());
        assertThrows(IllegalStateException.class, () -> scaler.getStd());
    }

    @Test
    void testIsFitted() {
        StandardScaler scaler = new StandardScaler();
        assertFalse(scaler.isFitted());

        double[][] data = {{1.0, 2.0}};
        scaler.fit(data);
        assertTrue(scaler.isFitted());
    }
}
