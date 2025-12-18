package io.github.yasmramos.mindforge.classification;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SGDClassifierTest {

    private double[][] X = {
        {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4},
        {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}
    };
    private int[] y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    @Test
    void testSGDClassifierDefaultConstructor() {
        SGDClassifier clf = new SGDClassifier();
        assertNotNull(clf);
    }

    @Test
    void testSGDClassifierFitPredict() {
        SGDClassifier clf = new SGDClassifier.Builder()
            .loss(SGDClassifier.Loss.HINGE)
            .penalty(SGDClassifier.Penalty.L2)
            .alpha(0.0001)
            .learningRate(0.01)
            .maxIterations(100)
            .randomSeed(42)
            .build();
        
        clf.fit(X, y);
        int[] predictions = clf.predict(X);
        
        assertEquals(X.length, predictions.length);
    }

    @Test
    void testSGDClassifierLogLoss() {
        SGDClassifier clf = new SGDClassifier.Builder()
            .loss(SGDClassifier.Loss.LOG)
            .build();
        
        clf.fit(X, y);
        int[] predictions = clf.predict(X);
        
        assertEquals(X.length, predictions.length);
    }

    @Test
    void testSGDClassifierPerceptronLoss() {
        SGDClassifier clf = new SGDClassifier.Builder()
            .loss(SGDClassifier.Loss.PERCEPTRON)
            .penalty(SGDClassifier.Penalty.NONE)
            .build();
        
        clf.fit(X, y);
        int[] predictions = clf.predict(X);
        
        assertEquals(X.length, predictions.length);
    }

    @Test
    void testSGDClassifierElasticNet() {
        SGDClassifier clf = new SGDClassifier.Builder()
            .penalty(SGDClassifier.Penalty.ELASTICNET)
            .l1Ratio(0.5)
            .build();
        
        clf.fit(X, y);
        int[] predictions = clf.predict(X);
        
        assertEquals(X.length, predictions.length);
    }

    @Test
    void testSGDClassifierMulticlass() {
        int[] yMulti = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
        SGDClassifier clf = new SGDClassifier();
        
        clf.fit(X, yMulti);
        int[] predictions = clf.predict(X);
        
        assertEquals(X.length, predictions.length);
    }
}
