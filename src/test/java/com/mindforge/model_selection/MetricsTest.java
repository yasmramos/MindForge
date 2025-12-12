package com.mindforge.model_selection;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MetricsTest {
    
    @Test
    void testAccuracy() {
        int[] yTrue = {0, 1, 1, 0, 1};
        int[] yPred = {0, 1, 0, 0, 1};
        
        assertEquals(0.8, Metrics.accuracy(yTrue, yPred), 0.001);
    }
    
    @Test
    void testPerfectAccuracy() {
        int[] y = {0, 1, 2, 0, 1};
        assertEquals(1.0, Metrics.accuracy(y, y), 0.001);
    }
    
    @Test
    void testConfusionMatrix() {
        int[] yTrue = {0, 0, 1, 1};
        int[] yPred = {0, 1, 0, 1};
        
        int[][] cm = Metrics.confusionMatrix(yTrue, yPred);
        
        assertEquals(1, cm[0][0]); // TN
        assertEquals(1, cm[0][1]); // FP
        assertEquals(1, cm[1][0]); // FN
        assertEquals(1, cm[1][1]); // TP
    }
    
    @Test
    void testF1Score() {
        int[] yTrue = {0, 0, 1, 1, 1};
        int[] yPred = {0, 0, 1, 1, 0};
        
        double f1 = Metrics.f1Score(yTrue, yPred);
        assertTrue(f1 > 0.7 && f1 < 1.0);
    }
    
    @Test
    void testRocAuc() {
        int[] yTrue = {0, 0, 1, 1};
        double[] yScores = {0.1, 0.4, 0.6, 0.9};
        
        double auc = Metrics.rocAucScore(yTrue, yScores);
        assertEquals(1.0, auc, 0.001); // Perfect separation
    }
    
    @Test
    void testMSE() {
        double[] yTrue = {1.0, 2.0, 3.0, 4.0};
        double[] yPred = {1.1, 2.1, 2.9, 4.1};
        
        double mse = Metrics.meanSquaredError(yTrue, yPred);
        assertTrue(mse < 0.02);
    }
    
    @Test
    void testR2Score() {
        double[] yTrue = {1.0, 2.0, 3.0, 4.0};
        double[] yPred = {1.0, 2.0, 3.0, 4.0};
        
        assertEquals(1.0, Metrics.r2Score(yTrue, yPred), 0.001);
    }
}
