package io.github.yasmramos.mindforge.model_selection;

import io.github.yasmramos.mindforge.classification.LogisticRegression;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

class ModelSelectionTest {

    private double[][] X = {
        {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4},
        {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9},
        {10, 10}, {11, 11}, {12, 12}, {13, 13}, {14, 14}
    };
    private int[] y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1};

    @Test
    void testRandomizedSearchCVDefaultConstructor() {
        RandomizedSearchCV search = new RandomizedSearchCV();
        assertNotNull(search);
    }

    @Test
    void testRandomizedSearchCVWithParams() {
        RandomizedSearchCV search = new RandomizedSearchCV(5, 3, 42);
        assertNotNull(search);
    }

    @Test
    void testLearningCurveDefaultConstructor() {
        LearningCurve lc = new LearningCurve();
        assertNotNull(lc);
    }

    @Test
    void testLearningCurveGenerate() {
        LearningCurve lc = new LearningCurve(3, 42);
        double[] sizes = {0.5, 0.75, 1.0};
        
        lc.generate(() -> new LogisticRegression(), X, y, sizes);
        
        assertNotNull(lc.getTrainSizes());
        assertNotNull(lc.getTrainScoresMean());
        assertNotNull(lc.getTestScoresMean());
        assertEquals(3, lc.getTrainSizes().length);
    }

    @Test
    void testValidationCurveDefaultConstructor() {
        ValidationCurve vc = new ValidationCurve();
        assertNotNull(vc);
    }

    @Test
    void testValidationCurveGenerate() {
        ValidationCurve vc = new ValidationCurve(3, 42);
        double[] paramValues = {0.01, 0.1, 1.0};
        
        vc.generate(
            lr -> new LogisticRegression.Builder().learningRate(lr).build(),
            X, y, paramValues
        );
        
        assertNotNull(vc.getParamRange());
        assertNotNull(vc.getTrainScoresMean());
        assertNotNull(vc.getTestScoresMean());
    }
}
