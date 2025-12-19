package io.github.yasmramos.mindforge.model_selection;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ModelSelectionTest {

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
    void testRandomizedSearchCVTwoParamConstructor() {
        RandomizedSearchCV search = new RandomizedSearchCV(10, 5);
        assertNotNull(search);
    }

    @Test
    void testLearningCurveDefaultConstructor() {
        LearningCurve lc = new LearningCurve();
        assertNotNull(lc);
    }

    @Test
    void testLearningCurveWithParams() {
        LearningCurve lc = new LearningCurve(3, 42);
        assertNotNull(lc);
    }

    @Test
    void testValidationCurveDefaultConstructor() {
        ValidationCurve vc = new ValidationCurve();
        assertNotNull(vc);
    }

    @Test
    void testValidationCurveWithParams() {
        ValidationCurve vc = new ValidationCurve(5, 123);
        assertNotNull(vc);
    }
}
