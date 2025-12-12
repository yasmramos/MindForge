package io.github.yasmramos.mindforge.classification;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;

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
    
    // ================== Kernel Tests ==================
    
    @Test
    public void testRBFKernelBinary() {
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.RBF)
            .gamma(0.5)
            .C(1.0)
            .maxIter(1000)
            .build();
        
        svc.train(XBinary, yBinary);
        
        assertTrue(svc.isTrained());
        assertTrue(svc.usesKernelMethod());
        assertEquals(2, svc.getNumClasses());
        assertNotNull(svc.getKernel());
        assertEquals(Kernel.Type.RBF, svc.getKernel().getType());
        
        int[] predictions = svc.predict(XBinary);
        assertEquals(XBinary.length, predictions.length);
    }
    
    @Test
    public void testRBFKernelMulticlass() {
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.RBF)
            .gamma(0.5)
            .C(1.0)
            .build();
        
        svc.train(XMulti, yMulti);
        
        assertTrue(svc.isTrained());
        assertEquals(3, svc.getNumClasses());
        
        int[] predictions = svc.predict(XMulti);
        assertEquals(XMulti.length, predictions.length);
    }
    
    @Test
    public void testPolynomialKernel() {
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.POLYNOMIAL)
            .degree(3)
            .gamma(1.0)
            .coef0(0.0)
            .C(1.0)
            .build();
        
        svc.train(XBinary, yBinary);
        
        assertTrue(svc.isTrained());
        assertTrue(svc.usesKernelMethod());
        assertEquals(Kernel.Type.POLYNOMIAL, svc.getKernel().getType());
        
        int[] predictions = svc.predict(XBinary);
        assertEquals(XBinary.length, predictions.length);
    }
    
    @Test
    public void testPolynomialKernelDegree2() {
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.POLYNOMIAL)
            .degree(2)
            .gamma(0.5)
            .coef0(1.0)
            .C(1.0)
            .build();
        
        svc.train(XBinary, yBinary);
        
        assertTrue(svc.isTrained());
        assertEquals(2, svc.getKernel().getDegree());
        
        int[] predictions = svc.predict(XBinary);
        assertEquals(XBinary.length, predictions.length);
    }
    
    @Test
    public void testSigmoidKernel() {
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.SIGMOID)
            .gamma(0.1)
            .coef0(0.0)
            .C(1.0)
            .build();
        
        svc.train(XBinary, yBinary);
        
        assertTrue(svc.isTrained());
        assertEquals(Kernel.Type.SIGMOID, svc.getKernel().getType());
        
        int[] predictions = svc.predict(XBinary);
        assertEquals(XBinary.length, predictions.length);
    }
    
    @Test
    public void testKernelByString() {
        SVC svcRbf = new SVC.Builder()
            .kernel("rbf")
            .gamma(0.5)
            .build();
        assertEquals(Kernel.Type.RBF, svcRbf.getKernel().getType());
        
        SVC svcPoly = new SVC.Builder()
            .kernel("poly")
            .degree(3)
            .build();
        assertEquals(Kernel.Type.POLYNOMIAL, svcPoly.getKernel().getType());
        
        SVC svcLinear = new SVC.Builder()
            .kernel("linear")
            .build();
        assertEquals(Kernel.Type.LINEAR, svcLinear.getKernel().getType());
        
        SVC svcSigmoid = new SVC.Builder()
            .kernel("sigmoid")
            .build();
        assertEquals(Kernel.Type.SIGMOID, svcSigmoid.getKernel().getType());
    }
    
    @Test
    public void testInvalidKernelString() {
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().kernel("invalid").build();
        });
    }
    
    @Test
    public void testLinearKernelExplicit() {
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.LINEAR)
            .C(1.0)
            .build();
        
        svc.train(XBinary, yBinary);
        
        assertTrue(svc.isTrained());
        assertFalse(svc.usesKernelMethod()); // Linear uses primal form
        
        // Linear kernel should have weights available
        double[][] weights = svc.getWeights();
        assertNotNull(weights);
        assertEquals(2, weights.length);
    }
    
    @Test
    public void testKernelSVMNoWeights() {
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.RBF)
            .gamma(0.5)
            .build();
        
        svc.train(XBinary, yBinary);
        
        // Kernel SVM should not have explicit weights
        assertThrows(IllegalStateException.class, () -> {
            svc.getWeights();
        });
    }
    
    @Test
    public void testSupportVectorCounts() {
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.RBF)
            .gamma(0.5)
            .C(1.0)
            .build();
        
        svc.train(XBinary, yBinary);
        
        int[] svCounts = svc.getNumSupportVectors();
        assertEquals(2, svCounts.length); // 2 classes
        
        int totalSV = svc.getTotalSupportVectors();
        assertTrue(totalSV >= 0);
    }
    
    @Test
    public void testDecisionFunctionKernel() {
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.RBF)
            .gamma(0.5)
            .build();
        
        svc.train(XBinary, yBinary);
        
        double[] scores = svc.decisionFunction(XBinary[0]);
        assertEquals(2, scores.length);
    }
    
    @Test
    public void testGammaAuto() {
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.RBF)
            .gammaAuto()
            .build();
        
        assertNotNull(svc);
    }
    
    @Test
    public void testGammaString() {
        SVC svcAuto = new SVC.Builder()
            .kernel(Kernel.Type.RBF)
            .gamma("auto")
            .build();
        assertNotNull(svcAuto);
        
        SVC svcScale = new SVC.Builder()
            .kernel(Kernel.Type.RBF)
            .gamma("scale")
            .build();
        assertNotNull(svcScale);
        
        SVC svcNumeric = new SVC.Builder()
            .kernel(Kernel.Type.RBF)
            .gamma("0.5")
            .build();
        assertNotNull(svcNumeric);
    }
    
    @Test
    public void testInvalidGammaString() {
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().gamma("invalid").build();
        });
    }
    
    @Test
    public void testInvalidGammaValue() {
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().gamma(0.0).build();
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().gamma(-1.0).build();
        });
    }
    
    @Test
    public void testInvalidDegree() {
        assertThrows(IllegalArgumentException.class, () -> {
            new SVC.Builder().degree(0).build();
        });
    }
    
    // ================== Non-linear Classification Tests ==================
    
    @Test
    public void testXORProblemRBF() {
        // XOR is not linearly separable, but can be solved with RBF kernel
        double[][] X = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        };
        int[] y = {0, 1, 1, 0};
        
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.RBF)
            .gamma(5.0)
            .C(10.0)
            .maxIter(1000)
            .build();
        
        svc.train(X, y);
        
        int[] predictions = svc.predict(X);
        assertEquals(4, predictions.length);
        // RBF should be able to classify XOR correctly
    }
    
    @Test
    public void testCircularDataRBF() {
        // Inner circle (class 0) and outer ring (class 1)
        double[][] X = {
            {0.0, 0.0}, {0.1, 0.1}, {-0.1, 0.1}, {0.1, -0.1},  // Inner
            {2.0, 0.0}, {0.0, 2.0}, {-2.0, 0.0}, {0.0, -2.0},  // Outer
            {1.5, 1.5}, {-1.5, 1.5}, {1.5, -1.5}, {-1.5, -1.5} // Outer
        };
        int[] y = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
        
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.RBF)
            .gamma(0.5)
            .C(1.0)
            .build();
        
        svc.train(X, y);
        
        int[] predictions = svc.predict(X);
        assertEquals(12, predictions.length);
    }
    
    @Test
    public void testPolynomialKernelQuadratic() {
        // Data that is separable with a quadratic decision boundary
        double[][] X = {
            {-2.0, 4.0}, {-1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0}, {2.0, 4.0},  // On/above parabola
            {-1.0, -1.0}, {0.0, -2.0}, {1.0, -1.0}  // Below parabola
        };
        int[] y = {0, 0, 0, 0, 0, 1, 1, 1};
        
        SVC svc = new SVC.Builder()
            .kernel(Kernel.Type.POLYNOMIAL)
            .degree(2)
            .gamma(1.0)
            .coef0(1.0)
            .C(1.0)
            .build();
        
        svc.train(X, y);
        
        int[] predictions = svc.predict(X);
        assertEquals(8, predictions.length);
    }
}
