package io.github.yasmramos.mindforge.anomaly;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class AnomalyDetectionTest {

    private double[][] createNormalData() {
        // Normal cluster around (0,0)
        return new double[][] {
            {0.1, 0.2}, {-0.1, 0.1}, {0.2, -0.1}, {0.0, 0.0},
            {0.15, 0.1}, {-0.2, 0.2}, {0.1, -0.2}, {-0.1, -0.1},
            {0.05, 0.15}, {-0.15, 0.05}, {0.2, 0.2}, {-0.2, -0.2},
            {0.1, 0.0}, {0.0, 0.1}, {-0.1, 0.0}, {0.0, -0.1},
            // Outliers
            {5.0, 5.0}, {-5.0, -5.0}, {4.5, 4.5}
        };
    }

    @Test
    void testIsolationForestFitPredict() {
        double[][] X = createNormalData();
        IsolationForest iforest = new IsolationForest.Builder()
            .nEstimators(50)
            .maxSamples(10)
            .contamination(0.15)
            .randomSeed(42)
            .build();
        
        iforest.fit(X);
        int[] predictions = iforest.predict(X);
        
        assertEquals(X.length, predictions.length);
        // Check predictions are either -1 or 1
        for (int p : predictions) {
            assertTrue(p == -1 || p == 1);
        }
    }

    @Test
    void testIsolationForestDecisionFunction() {
        double[][] X = createNormalData();
        IsolationForest iforest = new IsolationForest();
        iforest.fit(X);
        
        double[] scores = iforest.decisionFunction(X);
        assertEquals(X.length, scores.length);
    }

    @Test
    void testIsolationForestDefaultConstructor() {
        IsolationForest iforest = new IsolationForest();
        assertNotNull(iforest);
    }

    @Test
    void testLocalOutlierFactorFitPredict() {
        double[][] X = createNormalData();
        LocalOutlierFactor lof = new LocalOutlierFactor.Builder()
            .nNeighbors(5)
            .contamination(0.15)
            .build();
        
        int[] predictions = lof.fitPredict(X);
        
        assertEquals(X.length, predictions.length);
        for (int p : predictions) {
            assertTrue(p == -1 || p == 1);
        }
    }

    @Test
    void testLocalOutlierFactorDecisionFunction() {
        double[][] X = createNormalData();
        LocalOutlierFactor lof = new LocalOutlierFactor();
        lof.fit(X);
        
        double[] scores = lof.decisionFunction(X);
        assertEquals(X.length, scores.length);
    }

    @Test
    void testLocalOutlierFactorDefaultConstructor() {
        LocalOutlierFactor lof = new LocalOutlierFactor();
        assertNotNull(lof);
    }

    @Test
    void testOneClassSVMFitPredict() {
        double[][] X = createNormalData();
        OneClassSVM svm = new OneClassSVM.Builder()
            .nu(0.1)
            .gamma(0.5)
            .maxIterations(100)
            .build();
        
        svm.fit(X);
        int[] predictions = svm.predict(X);
        
        assertEquals(X.length, predictions.length);
        for (int p : predictions) {
            assertTrue(p == -1 || p == 1);
        }
    }

    @Test
    void testOneClassSVMDecisionFunction() {
        double[][] X = createNormalData();
        OneClassSVM svm = new OneClassSVM();
        svm.fit(X);
        
        double[] scores = svm.decisionFunction(X);
        assertEquals(X.length, scores.length);
    }

    @Test
    void testOneClassSVMDefaultConstructor() {
        OneClassSVM svm = new OneClassSVM();
        assertNotNull(svm);
    }
}
