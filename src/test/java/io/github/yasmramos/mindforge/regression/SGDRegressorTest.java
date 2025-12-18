package io.github.yasmramos.mindforge.regression;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SGDRegressorTest {

    private double[][] X = {
        {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}
    };
    private double[] y = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

    @Test
    void testSGDRegressorDefaultConstructor() {
        SGDRegressor reg = new SGDRegressor();
        assertNotNull(reg);
    }

    @Test
    void testSGDRegressorFitPredict() {
        SGDRegressor reg = new SGDRegressor();
        reg.fit(X, y);
        
        double[] predictions = reg.predict(X);
        assertEquals(X.length, predictions.length);
    }

    @Test
    void testSGDRegressorBuilder() {
        SGDRegressor reg = new SGDRegressor.Builder()
            .learningRate(0.01)
            .maxIterations(500)
            .build();
        
        reg.fit(X, y);
        double[] predictions = reg.predict(X);
        
        assertEquals(X.length, predictions.length);
    }
}
