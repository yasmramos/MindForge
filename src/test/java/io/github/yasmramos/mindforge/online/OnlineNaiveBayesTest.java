package io.github.yasmramos.mindforge.online;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for OnlineNaiveBayes.
 */
public class OnlineNaiveBayesTest {
    
    @Test
    public void testPartialFit() {
        OnlineNaiveBayes classifier = new OnlineNaiveBayes(4, 3);
        
        double[][] X1 = {
            {5.1, 3.5, 1.4, 0.2},
            {4.9, 3.0, 1.4, 0.2},
            {7.0, 3.2, 4.7, 1.4}
        };
        int[] y1 = {0, 0, 1};
        
        classifier.partialFit(X1, y1);
        
        double[][] X2 = {
            {6.3, 3.3, 6.0, 2.5},
            {5.8, 2.7, 5.1, 1.9}
        };
        int[] y2 = {2, 2};
        
        classifier.partialFit(X2, y2);
        
        int prediction = classifier.predict(new double[]{5.0, 3.4, 1.5, 0.2});
        assertTrue(prediction >= 0 && prediction < 3, "Prediction should be valid class");
    }
    
    @Test
    public void testPredictProba() {
        OnlineNaiveBayes classifier = new OnlineNaiveBayes(2, 2);
        
        double[][] X = {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 4.0},
            {8.0, 9.0},
            {9.0, 10.0},
            {10.0, 11.0}
        };
        int[] y = {0, 0, 0, 1, 1, 1};
        
        classifier.partialFit(X, y);
        
        double[] proba = classifier.predictProba(new double[]{2.0, 3.0});
        assertEquals(2, proba.length, "Should return probability for each class");
        assertTrue(proba[0] >= 0.0 && proba[0] <= 1.0, "Probability should be in [0, 1]");
        assertTrue(proba[1] >= 0.0 && proba[1] <= 1.0, "Probability should be in [0, 1]");
        assertEquals(1.0, proba[0] + proba[1], 0.001, "Probabilities should sum to 1");
    }
    
    @Test
    public void testIncrementalLearning() {
        OnlineNaiveBayes classifier = new OnlineNaiveBayes(2, 2);
        
        double[][] X1 = {{1.0, 2.0}, {2.0, 3.0}};
        int[] y1 = {0, 0};
        classifier.partialFit(X1, y1);
        
        int pred1 = classifier.predict(new double[]{1.5, 2.5});
        
        double[][] X2 = {{8.0, 9.0}, {9.0, 10.0}};
        int[] y2 = {1, 1};
        classifier.partialFit(X2, y2);
        
        int pred2 = classifier.predict(new double[]{8.5, 9.5});
        
        assertNotEquals(pred1, pred2, "Predictions should differ after learning new class");
    }
}
