package io.github.yasmramos.mindforge.multilabel;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for BinaryRelevanceClassifier.
 */
public class BinaryRelevanceClassifierTest {
    
    @Test
    public void testFitAndPredict() {
        int nLabels = 3;
        int nFeatures = 4;
        BinaryRelevanceClassifier classifier = new BinaryRelevanceClassifier(nLabels, nFeatures);
        
        double[][] X = {
            {1.0, 2.0, 3.0, 4.0},
            {2.0, 3.0, 4.0, 5.0},
            {5.0, 6.0, 7.0, 8.0},
            {6.0, 7.0, 8.0, 9.0}
        };
        
        int[][] y = {
            {1, 0, 1},
            {1, 0, 0},
            {0, 1, 1},
            {0, 1, 0}
        };
        
        classifier.fit(X, y);
        
        int[] prediction = classifier.predict(new double[]{1.5, 2.5, 3.5, 4.5});
        assertEquals(3, prediction.length, "Should predict for each label");
        assertTrue(prediction[0] == 0 || prediction[0] == 1, "Prediction should be binary");
        assertTrue(prediction[1] == 0 || prediction[1] == 1, "Prediction should be binary");
        assertTrue(prediction[2] == 0 || prediction[2] == 1, "Prediction should be binary");
    }
    
    @Test
    public void testPredictProba() {
        int nLabels = 2;
        int nFeatures = 3;
        BinaryRelevanceClassifier classifier = new BinaryRelevanceClassifier(nLabels, nFeatures);
        
        double[][] X = {
            {1.0, 1.0, 1.0},
            {2.0, 2.0, 2.0},
            {8.0, 8.0, 8.0},
            {9.0, 9.0, 9.0}
        };
        
        int[][] y = {
            {1, 0},
            {1, 0},
            {0, 1},
            {0, 1}
        };
        
        classifier.fit(X, y);
        
        double[] proba = classifier.predictProba(new double[]{1.5, 1.5, 1.5});
        assertEquals(2, proba.length, "Should return probability for each label");
        assertTrue(proba[0] >= 0.0 && proba[0] <= 1.0, "Probability should be in [0, 1]");
        assertTrue(proba[1] >= 0.0 && proba[1] <= 1.0, "Probability should be in [0, 1]");
    }
    
    @Test
    public void testBatchPrediction() {
        int nLabels = 2;
        int nFeatures = 2;
        BinaryRelevanceClassifier classifier = new BinaryRelevanceClassifier(nLabels, nFeatures);
        
        double[][] X = {
            {1.0, 2.0},
            {3.0, 4.0},
            {7.0, 8.0}
        };
        
        int[][] y = {
            {1, 0},
            {1, 1},
            {0, 1}
        };
        
        classifier.fit(X, y);
        
        int[][] predictions = classifier.predict(X);
        assertEquals(3, predictions.length, "Should predict for each sample");
        assertEquals(2, predictions[0].length, "Should predict for each label");
    }
    
    @Test
    public void testSetThreshold() {
        int nLabels = 2;
        int nFeatures = 2;
        BinaryRelevanceClassifier classifier = new BinaryRelevanceClassifier(nLabels, nFeatures);
        
        double[][] X = {
            {1.0, 2.0},
            {2.0, 3.0},
            {8.0, 9.0},
            {9.0, 10.0}
        };
        
        int[][] y = {
            {1, 0},
            {1, 0},
            {0, 1},
            {0, 1}
        };
        
        classifier.fit(X, y);
        classifier.setThreshold(0, 0.3);
        classifier.setThreshold(1, 0.7);
        
        int[] prediction = classifier.predict(new double[]{1.5, 2.5});
        assertNotNull(prediction);
    }
}
