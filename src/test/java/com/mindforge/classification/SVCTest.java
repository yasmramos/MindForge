package com.mindforge.classification;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Support Vector Classifier (SVC).
 */
public class SVCTest {
    
    private double[][] XBinary;
    private int[] yBinary;
    private double[][] XMulti;
    private int[] yMulti;
    
    @BeforeEach
    public void setUp() {
        // Binary classification dataset (linearly separable)
        XBinary = new double[][] {
            {-2.0, -2.0}, {-1.8, -2.2}, {-2.2, -1.8}, {-1.9, -2.1},
            {2.0, 2.0}, {1.8, 2.2}, {2.2, 1.8}, {1.9, 2.1}
        };
        yBinary = new int[] {0, 0, 0, 0, 1, 1, 1, 1};
        
        // Multiclass dataset
        XMulti = new double[][] {
            {-3.0, 0.0}, {-2.8, 0.2}, {-3.2, -0.2},
            {0.0, 3.0}, {0.2, 2.8}, {-0.2, 3.2},
            {3.0, 0.0}, {2.8, -0.2}, {3.2, 0.2}
        };
        yMulti = new int[] {0, 0, 0, 1, 1, 1, 2, 2, 2};
    }
    
    @Test
    public void testBuilderDefaults() {
        SVC svc = new SVC.Builder().build();
        assertNotNull(svc);
    }
    
    @Test
    public void testBuilderCustom() {
        SVC svc = new SVC.Builder()
            .C(0.5)
            .maxIter(500)
            .tol(1e-4)
            .learningRate(0.001)
            .build();
        assertNotNull(svc);
    }
    
    @Test
    public void testBinaryClassification() {
        SVC svc = new SVC.Builder()
            .C(1.0)
            .maxIter(1000)
            .learningRate(0.01)
            .build();
        
        svc.train(XBinary, yBinary);
        
        assertTrue(svc.isTrained());
        assertEquals(2, svc.getNumClasses());
        
        // Make predictions
        int[] predictions = svc.predict(XBinary);
        assertEquals(XBinary.length, predictions.length);
    }
    
    @Test
    public void testMulticlassClassification() {
        SVC svc = new SVC.Builder()
            .C(1.0)
            .maxIter(1000)
            .learningRate(0.01)
            .build();
        
        svc.train(XMulti, yMulti);
        
        assertTrue(svc.isTrained());
        assertEquals(3, svc.getNumClasses());
        
        int[] predictions = svc.predict(XMulti);
        assertEquals(XMulti.length, predictions.length);
    }
    
    @Test
    public void testDecisionFunction() {
        SVC svc = new SVC.Builder().build();
        svc.train(XBinary, yBinary);
        
        double[] scores = svc.decisionFunction(XBinary[0]);
        assertEquals(2, scores.length);
    }
    
    @Test
    public void testGetters() {
        SVC svc = new SVC.Builder().build();
        svc.train(XBinary, yBinary);
        
        int[] classes = svc.getClasses();
        assertEquals(2, classes.length);
        
        double[][] weights = svc.getWeights();
        assertEquals(2, weights.length);
        assertEquals(2, weights[0].length);
        
        double[] bias = svc.getBias();
        assertEquals(2, bias.length);
    }
    
    // Edge cases and error handling
    
    @Test
    public void testPredictBeforeTraining() {
        SVC svc = new SVC.Builder().build();
        
        assertThrows(IllegalStateException.class, () -> {
            svc.predict(new double[]{1.0, 2.0});
        });
    }
    
    @Test
    public void testDecisionFunctionBeforeTraining() {
        SVC svc = new SVC.Builder().build();
        
        assertThrows(IllegalStateException.class, () -> {
            svc.decisionFunction(new double[]{1.0, 2.0});
        });
    }
    
    @Test
    public void testMismatchedArrays() {
        SVC svc = new SVC.Builder().build();
        double[][] X = {{1.0, 2.0}, {3.0, 4.0}};
        int[] y = {0};
        
        assertThrows(IllegalArgumentException.class, () -> {
            svc.train(X, y);
        });
    }
    
    @Test
    public void testEmptyData() {
        SVC svc = new SVC.Builder().build();
        double[][] X = {};
        int[] y = {};
        
        assertThrows(IllegalArgumentException.class, () -> {
            svc.train(X, y);
        });
    }
    
    @Test
    public void testWrongFeatureCount() {
        SVC svc = new SVC.Builder().build();
        svc.train(XBinary, yBinary);
        
        assertThrows(IllegalArgumentException.class, () -> {
            svc.predict(new double[]{1.0});
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            svc.decisionFunction(new double[]{1.0, 2.0, 3.0});
        });
    }
    
    @Test
    public void testInvalidC() {
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().C(0.0).build();
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().C(-1.0).build();
        });
    }
    
    @Test
    public void testInvalidMaxIter() {
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().maxIter(0).build();
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().maxIter(-100).build();
        });
    }
    
    @Test
    public void testInvalidTol() {
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().tol(0.0).build();
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().tol(-1e-3).build();
        });
    }
    
    @Test
    public void testInvalidLearningRate() {
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().learningRate(0.0).build();
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().learningRate(-0.01).build();
        });
    }
}
