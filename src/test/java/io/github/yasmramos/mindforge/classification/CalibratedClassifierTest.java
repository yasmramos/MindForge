package io.github.yasmramos.mindforge.classification;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CalibratedClassifierTest {

    private double[][] X = {
        {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4},
        {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}
    };
    private int[] y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    @Test
    void testCalibratedClassifierFitPredict() {
        SVC baseModel = new SVC();
        CalibratedClassifier clf = new CalibratedClassifier(baseModel);
        
        clf.fit(X, y);
        int[] predictions = clf.predict(X);
        
        assertEquals(X.length, predictions.length);
    }

    @Test
    void testCalibratedClassifierPredictProbability() {
        SVC baseModel = new SVC();
        CalibratedClassifier clf = new CalibratedClassifier(baseModel);
        
        clf.fit(X, y);
        double[][] proba = clf.predictProbability(X);
        
        assertNotNull(proba);
        assertEquals(X.length, proba.length);
        
        // Check probabilities sum to 1
        for (double[] p : proba) {
            double sum = 0;
            for (double prob : p) {
                sum += prob;
                assertTrue(prob >= 0 && prob <= 1);
            }
            assertEquals(1.0, sum, 0.01);
        }
    }
}
