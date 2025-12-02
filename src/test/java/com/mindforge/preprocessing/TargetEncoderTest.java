package com.mindforge.preprocessing;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for TargetEncoder.
 */
class TargetEncoderTest {
    
    @Test
    @DisplayName("Default constructor works")
    void testDefaultConstructor() {
        TargetEncoder encoder = new TargetEncoder();
        assertNotNull(encoder);
        assertFalse(encoder.isFitted());
    }
    
    @Test
    @DisplayName("Fit with regression target")
    void testFitRegression() {
        String[][] X = {{"a"}, {"b"}, {"a"}, {"b"}};
        double[] y = {1.0, 2.0, 1.0, 2.0};
        
        TargetEncoder encoder = new TargetEncoder();
        encoder.fit(X, y);
        
        assertTrue(encoder.isFitted());
    }
    
    @Test
    @DisplayName("Fit with classification target")
    void testFitClassification() {
        String[][] X = {{"a"}, {"b"}, {"a"}, {"b"}};
        int[] y = {0, 1, 0, 1};
        
        TargetEncoder encoder = new TargetEncoder();
        encoder.fit(X, y);
        
        assertTrue(encoder.isFitted());
    }
    
    @Test
    @DisplayName("Transform encodes correctly")
    void testTransform() {
        String[][] X = {{"a"}, {"b"}, {"a"}, {"b"}};
        double[] y = {1.0, 2.0, 1.0, 2.0};
        
        TargetEncoder encoder = new TargetEncoder(0.0);
        encoder.fit(X, y);
        double[][] transformed = encoder.transform(X);
        
        assertEquals(4, transformed.length);
        assertEquals(1, transformed[0].length);
        
        // "a" should encode to ~1.0, "b" to ~2.0
        assertEquals(transformed[0][0], transformed[2][0], 0.01);
        assertEquals(transformed[1][0], transformed[3][0], 0.01);
    }
    
    @Test
    @DisplayName("Fit transform works")
    void testFitTransform() {
        String[][] X = {{"cat"}, {"dog"}, {"cat"}, {"dog"}};
        double[] y = {10.0, 20.0, 10.0, 20.0};
        
        TargetEncoder encoder = new TargetEncoder();
        double[][] transformed = encoder.fitTransform(X, y);
        
        assertTrue(encoder.isFitted());
        assertEquals(4, transformed.length);
    }
    
    @Test
    @DisplayName("Smoothing affects encoding")
    void testSmoothing() {
        String[][] X = {{"a"}, {"b"}, {"a"}};
        double[] y = {1.0, 10.0, 1.0};
        
        TargetEncoder noSmooth = new TargetEncoder(0.0);
        TargetEncoder withSmooth = new TargetEncoder(5.0);
        
        double[][] result1 = noSmooth.fitTransform(X, y);
        double[][] result2 = withSmooth.fitTransform(X, y);
        
        // With smoothing, values should be pulled toward global mean
        assertNotEquals(result1[0][0], result2[0][0], 0.01);
    }
    
    @Test
    @DisplayName("Handle unknown categories")
    void testHandleUnknown() {
        String[][] XTrain = {{"a"}, {"b"}};
        double[] y = {1.0, 2.0};
        
        TargetEncoder encoder = new TargetEncoder(0.0, 1, true, 0.0);
        encoder.fit(XTrain, y);
        
        String[][] XTest = {{"a"}, {"c"}}; // "c" is unknown
        double[][] transformed = encoder.transform(XTest);
        
        assertEquals(2, transformed.length);
    }
    
    @Test
    @DisplayName("Null inputs throw exception")
    void testNullInputs() {
        TargetEncoder encoder = new TargetEncoder();
        assertThrows(IllegalArgumentException.class, () -> encoder.fit(null, new double[]{1.0}));
        assertThrows(IllegalArgumentException.class, () -> encoder.fit(new String[][]{{"a"}}, (double[]) null));
    }
    
    @Test
    @DisplayName("Multiple features work")
    void testMultipleFeatures() {
        String[][] X = {
            {"a", "x"}, {"b", "y"}, {"a", "y"}, {"b", "x"}
        };
        double[] y = {1.0, 2.0, 1.5, 2.5};
        
        TargetEncoder encoder = new TargetEncoder();
        double[][] transformed = encoder.fitTransform(X, y);
        
        assertEquals(4, transformed.length);
        assertEquals(2, transformed[0].length);
    }
}
