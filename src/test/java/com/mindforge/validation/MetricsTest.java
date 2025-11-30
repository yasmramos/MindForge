package com.mindforge.validation;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MetricsTest {
    
    @Test
    void testAccuracy() {
        int[] yTrue = {0, 1, 0, 1, 0, 1};
        int[] yPred = {0, 1, 0, 1, 1, 0};
        
        double accuracy = Metrics.accuracy(yTrue, yPred);
        assertEquals(0.6667, accuracy, 0.01, "Accuracy should be 4/6");
    }
    
    @Test
    void testPerfectAccuracy() {
        int[] yTrue = {0, 1, 2, 1, 0};
        int[] yPred = {0, 1, 2, 1, 0};
        
        double accuracy = Metrics.accuracy(yTrue, yPred);
        assertEquals(1.0, accuracy, 0.001, "Perfect predictions should have accuracy 1.0");
    }
    
    @Test
    void testMSE() {
        double[] yTrue = {1.0, 2.0, 3.0, 4.0};
        double[] yPred = {1.0, 2.0, 3.0, 4.0};
        
        double mse = Metrics.mse(yTrue, yPred);
        assertEquals(0.0, mse, 0.001, "Perfect predictions should have MSE 0.0");
        
        double[] yPred2 = {2.0, 3.0, 4.0, 5.0};
        mse = Metrics.mse(yTrue, yPred2);
        assertEquals(1.0, mse, 0.001, "Constant error of 1 should give MSE 1.0");
    }
    
    @Test
    void testRMSE() {
        double[] yTrue = {1.0, 2.0, 3.0, 4.0};
        double[] yPred = {2.0, 3.0, 4.0, 5.0};
        
        double rmse = Metrics.rmse(yTrue, yPred);
        assertEquals(1.0, rmse, 0.001, "RMSE should be square root of MSE");
    }
    
    @Test
    void testMAE() {
        double[] yTrue = {1.0, 2.0, 3.0, 4.0};
        double[] yPred = {2.0, 3.0, 4.0, 5.0};
        
        double mae = Metrics.mae(yTrue, yPred);
        assertEquals(1.0, mae, 0.001, "Constant error of 1 should give MAE 1.0");
    }
    
    @Test
    void testR2Score() {
        double[] yTrue = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] yPred = {1.0, 2.0, 3.0, 4.0, 5.0};
        
        double r2 = Metrics.r2Score(yTrue, yPred);
        assertEquals(1.0, r2, 0.001, "Perfect predictions should have R² = 1.0");
    }
    
    @Test
    void testPrecision() {
        int[] yTrue = {1, 1, 0, 1, 0, 0};
        int[] yPred = {1, 0, 0, 1, 0, 1};
        
        double precision = Metrics.precision(yTrue, yPred, 1);
        assertEquals(0.6667, precision, 0.01, "Precision should be 2/3");
    }
    
    @Test
    void testRecall() {
        int[] yTrue = {1, 1, 0, 1, 0, 0};
        int[] yPred = {1, 0, 0, 1, 0, 1};
        
        double recall = Metrics.recall(yTrue, yPred, 1);
        assertEquals(0.6667, recall, 0.01, "Recall should be 2/3");
    }
    
    @Test
    void testF1Score() {
        int[] yTrue = {1, 1, 0, 1, 0, 0};
        int[] yPred = {1, 1, 0, 1, 0, 0};
        
        double f1 = Metrics.f1Score(yTrue, yPred, 1);
        assertEquals(1.0, f1, 0.001, "Perfect predictions should have F1 = 1.0");
    }
}
