package io.github.yasmramos.mindforge.classification;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CalibratedClassifierTest {

    @Test
    void testCalibratedClassifierMethodEnum() {
        assertNotNull(CalibratedClassifier.Method.SIGMOID);
        assertNotNull(CalibratedClassifier.Method.ISOTONIC);
        assertEquals(2, CalibratedClassifier.Method.values().length);
    }
}
