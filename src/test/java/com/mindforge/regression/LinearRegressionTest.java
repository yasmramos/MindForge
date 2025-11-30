package com.mindforge.regression;

import com.mindforge.validation.Metrics;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class LinearRegressionTest {
    
    @Test
    void testSimpleLinearRelationship() {
        // y = 2x
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
        double[] y = {2.0, 4.0, 6.0, 8.0, 10.0};
        
        LinearRegression lr = new LinearRegression();
        lr.train(X, y);
        
        // Test prediction
        double prediction = lr.predict(new double[]{6.0});
        assertEquals(12.0, prediction, 1.0, "Should predict approximately 12 for input 6");
        
        // Test low error on training data
        double[] predictions = lr.predict(X);
        double rmse = Metrics.rmse(y, predictions);
        assertTrue(rmse < 0.5, "RMSE should be low on simple linear data");
    }
    
    @Test
    void testR2Score() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
        double[] y = {2.1, 4.2, 5.9, 8.1, 10.0};
        
        LinearRegression lr = new LinearRegression();
        lr.train(X, y);
        
        double[] predictions = lr.predict(X);
        double r2 = Metrics.r2Score(y, predictions);
        assertTrue(r2 > 0.95, "R² score should be high for near-linear data");
    }
    
    @Test
    void testIsFitted() {
        LinearRegression lr = new LinearRegression();
        assertFalse(lr.isFitted(), "Model should not be fitted initially");
        
        double[][] X = {{1.0}, {2.0}};
        double[] y = {1.0, 2.0};
        lr.train(X, y);
        
        assertTrue(lr.isFitted(), "Model should be fitted after training");
    }
    
    @Test
    void testPredictBeforeTraining() {
        LinearRegression lr = new LinearRegression();
        assertThrows(IllegalStateException.class, () -> {
            lr.predict(new double[]{1.0});
        }, "Should throw exception when predicting before training");
    }
}
