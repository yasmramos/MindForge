package com.mindforge.classification;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Kernel functions.
 */
public class KernelTest {
    
    private static final double DELTA = 1e-6;
    
    // ================== Linear Kernel Tests ==================
    
    @Test
    public void testLinearKernelDefault() {
        Kernel kernel = new Kernel();
        assertEquals(Kernel.Type.LINEAR, kernel.getType());
    }
    
    @Test
    public void testLinearKernelCompute() {
        Kernel kernel = Kernel.linear();
        
        double[] x = {1.0, 2.0, 3.0};
        double[] y = {4.0, 5.0, 6.0};
        
        // Linear kernel: x · y = 1*4 + 2*5 + 3*6 = 32
        double result = kernel.compute(x, y);
        assertEquals(32.0, result, DELTA);
    }
    
    @Test
    public void testLinearKernelSameVector() {
        Kernel kernel = Kernel.linear();
        
        double[] x = {1.0, 2.0, 3.0};
        
        // K(x, x) = ||x||² = 1 + 4 + 9 = 14
        double result = kernel.compute(x, x);
        assertEquals(14.0, result, DELTA);
    }
    
    @Test
    public void testLinearKernelOrthogonal() {
        Kernel kernel = Kernel.linear();
        
        double[] x = {1.0, 0.0};
        double[] y = {0.0, 1.0};
        
        // Orthogonal vectors have zero dot product
        double result = kernel.compute(x, y);
        assertEquals(0.0, result, DELTA);
    }
    
    // ================== RBF Kernel Tests ==================
    
    @Test
    public void testRBFKernelCreation() {
        Kernel kernel = Kernel.rbf(0.5);
        
        assertEquals(Kernel.Type.RBF, kernel.getType());
        assertEquals(0.5, kernel.getGamma(), DELTA);
    }
    
    @Test
    public void testRBFKernelSameVector() {
        Kernel kernel = Kernel.rbf(0.5);
        
        double[] x = {1.0, 2.0, 3.0};
        
        // K(x, x) = exp(0) = 1 for any x
        double result = kernel.compute(x, x);
        assertEquals(1.0, result, DELTA);
    }
    
    @Test
    public void testRBFKernelCompute() {
        Kernel kernel = Kernel.rbf(0.5);
        
        double[] x = {0.0, 0.0};
        double[] y = {1.0, 1.0};
        
        // ||x - y||² = 1 + 1 = 2
        // K(x, y) = exp(-0.5 * 2) = exp(-1) ≈ 0.3679
        double result = kernel.compute(x, y);
        assertEquals(Math.exp(-1.0), result, DELTA);
    }
    
    @Test
    public void testRBFKernelHighGamma() {
        // High gamma = narrow Gaussian = more localized
        Kernel kernel = Kernel.rbf(10.0);
        
        double[] x = {0.0, 0.0};
        double[] y = {1.0, 0.0};
        
        // ||x - y||² = 1
        // K(x, y) = exp(-10 * 1) = exp(-10) ≈ 0.0000454
        double result = kernel.compute(x, y);
        assertEquals(Math.exp(-10.0), result, DELTA);
    }
    
    @Test
    public void testRBFKernelLowGamma() {
        // Low gamma = wide Gaussian = smoother decision boundary
        Kernel kernel = Kernel.rbf(0.1);
        
        double[] x = {0.0, 0.0};
        double[] y = {1.0, 0.0};
        
        // ||x - y||² = 1
        // K(x, y) = exp(-0.1 * 1) = exp(-0.1) ≈ 0.9048
        double result = kernel.compute(x, y);
        assertEquals(Math.exp(-0.1), result, DELTA);
    }
    
    @Test
    public void testRBFAutoGamma() {
        int nFeatures = 10;
        Kernel kernel = Kernel.rbf(nFeatures);
        
        // Auto gamma should be 1/n_features = 0.1
        assertEquals(0.1, kernel.getGamma(), DELTA);
    }
    
    // ================== Polynomial Kernel Tests ==================
    
    @Test
    public void testPolynomialKernelCreation() {
        Kernel kernel = Kernel.polynomial(3);
        
        assertEquals(Kernel.Type.POLYNOMIAL, kernel.getType());
        assertEquals(3, kernel.getDegree());
    }
    
    @Test
    public void testPolynomialKernelDegree1() {
        // Degree 1 polynomial is essentially linear
        Kernel kernel = Kernel.polynomial(1, 1.0, 0.0);
        
        double[] x = {1.0, 2.0};
        double[] y = {3.0, 4.0};
        
        // (1*3 + 2*4)^1 = 11
        double result = kernel.compute(x, y);
        assertEquals(11.0, result, DELTA);
    }
    
    @Test
    public void testPolynomialKernelDegree2() {
        Kernel kernel = Kernel.polynomial(2, 1.0, 0.0);
        
        double[] x = {1.0, 2.0};
        double[] y = {3.0, 4.0};
        
        // (1*3 + 2*4)^2 = 11^2 = 121
        double result = kernel.compute(x, y);
        assertEquals(121.0, result, DELTA);
    }
    
    @Test
    public void testPolynomialKernelDegree3() {
        Kernel kernel = Kernel.polynomial(3, 1.0, 0.0);
        
        double[] x = {1.0, 2.0};
        double[] y = {3.0, 4.0};
        
        // (1*3 + 2*4)^3 = 11^3 = 1331
        double result = kernel.compute(x, y);
        assertEquals(1331.0, result, DELTA);
    }
    
    @Test
    public void testPolynomialKernelWithCoef0() {
        Kernel kernel = Kernel.polynomial(2, 1.0, 1.0);
        
        double[] x = {1.0, 2.0};
        double[] y = {3.0, 4.0};
        
        // (1*3 + 2*4 + 1)^2 = 12^2 = 144
        double result = kernel.compute(x, y);
        assertEquals(144.0, result, DELTA);
    }
    
    @Test
    public void testPolynomialKernelWithGamma() {
        Kernel kernel = Kernel.polynomial(2, 0.5, 1.0);
        
        double[] x = {2.0, 4.0};
        double[] y = {4.0, 6.0};
        
        // dot = 2*4 + 4*6 = 32
        // (0.5 * 32 + 1)^2 = 17^2 = 289
        double result = kernel.compute(x, y);
        assertEquals(289.0, result, DELTA);
    }
    
    // ================== Sigmoid Kernel Tests ==================
    
    @Test
    public void testSigmoidKernelCreation() {
        Kernel kernel = Kernel.sigmoid(0.1, 0.5);
        
        assertEquals(Kernel.Type.SIGMOID, kernel.getType());
        assertEquals(0.1, kernel.getGamma(), DELTA);
        assertEquals(0.5, kernel.getCoef0(), DELTA);
    }
    
    @Test
    public void testSigmoidKernelCompute() {
        Kernel kernel = Kernel.sigmoid(0.1, 0.0);
        
        double[] x = {1.0, 0.0};
        double[] y = {0.0, 1.0};
        
        // tanh(0.1 * 0 + 0) = tanh(0) = 0
        double result = kernel.compute(x, y);
        assertEquals(0.0, result, DELTA);
    }
    
    @Test
    public void testSigmoidKernelPositive() {
        Kernel kernel = Kernel.sigmoid(0.5, 0.0);
        
        double[] x = {1.0, 1.0};
        double[] y = {1.0, 1.0};
        
        // tanh(0.5 * 2 + 0) = tanh(1) ≈ 0.7616
        double result = kernel.compute(x, y);
        assertEquals(Math.tanh(1.0), result, DELTA);
    }
    
    // ================== Kernel Matrix Tests ==================
    
    @Test
    public void testKernelMatrixLinear() {
        Kernel kernel = Kernel.linear();
        
        double[][] X = {
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 1.0}
        };
        
        double[][] K = kernel.computeMatrix(X);
        
        assertEquals(3, K.length);
        assertEquals(3, K[0].length);
        
        // Check diagonal (K(x, x))
        assertEquals(1.0, K[0][0], DELTA);
        assertEquals(1.0, K[1][1], DELTA);
        assertEquals(2.0, K[2][2], DELTA);
        
        // Check symmetry
        assertEquals(K[0][1], K[1][0], DELTA);
        assertEquals(K[0][2], K[2][0], DELTA);
        assertEquals(K[1][2], K[2][1], DELTA);
    }
    
    @Test
    public void testKernelMatrixRBF() {
        Kernel kernel = Kernel.rbf(1.0);
        
        double[][] X = {
            {0.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0}
        };
        
        double[][] K = kernel.computeMatrix(X);
        
        // Diagonal should be 1.0 for RBF
        assertEquals(1.0, K[0][0], DELTA);
        assertEquals(1.0, K[1][1], DELTA);
        assertEquals(1.0, K[2][2], DELTA);
        
        // K[0][1] = K[0][2] = exp(-1)
        assertEquals(Math.exp(-1.0), K[0][1], DELTA);
        assertEquals(Math.exp(-1.0), K[0][2], DELTA);
        
        // K[1][2] = exp(-2) (distance = sqrt(2))
        assertEquals(Math.exp(-2.0), K[1][2], DELTA);
    }
    
    @Test
    public void testComputeRow() {
        Kernel kernel = Kernel.linear();
        
        double[] x = {1.0, 1.0};
        double[][] X = {
            {1.0, 0.0},
            {0.0, 1.0},
            {2.0, 2.0}
        };
        
        double[] row = kernel.computeRow(x, X);
        
        assertEquals(3, row.length);
        assertEquals(1.0, row[0], DELTA);  // 1*1 + 1*0 = 1
        assertEquals(1.0, row[1], DELTA);  // 1*0 + 1*1 = 1
        assertEquals(4.0, row[2], DELTA);  // 1*2 + 1*2 = 4
    }
    
    // ================== Factory Methods Tests ==================
    
    @Test
    public void testFactoryMethods() {
        Kernel linear = Kernel.linear();
        assertEquals(Kernel.Type.LINEAR, linear.getType());
        
        Kernel rbf = Kernel.rbf(0.5);
        assertEquals(Kernel.Type.RBF, rbf.getType());
        
        Kernel poly = Kernel.polynomial(3);
        assertEquals(Kernel.Type.POLYNOMIAL, poly.getType());
        
        Kernel sigmoid = Kernel.sigmoid(0.1, 0.5);
        assertEquals(Kernel.Type.SIGMOID, sigmoid.getType());
    }
    
    @Test
    public void testDefaultGamma() {
        double gamma = Kernel.defaultGamma(10);
        assertEquals(0.1, gamma, DELTA);
        
        gamma = Kernel.defaultGamma(100);
        assertEquals(0.01, gamma, DELTA);
    }
    
    // ================== Error Handling Tests ==================
    
    @Test
    public void testInvalidGamma() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Kernel(Kernel.Type.RBF, 0.0);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new Kernel(Kernel.Type.RBF, -1.0);
        });
    }
    
    @Test
    public void testInvalidDegree() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Kernel(Kernel.Type.POLYNOMIAL, 1.0, 0.0, 0);
        });
    }
    
    @Test
    public void testDimensionMismatch() {
        Kernel kernel = Kernel.linear();
        
        double[] x = {1.0, 2.0};
        double[] y = {1.0, 2.0, 3.0};
        
        assertThrows(IllegalArgumentException.class, () -> {
            kernel.compute(x, y);
        });
    }
    
    // ================== ToString Tests ==================
    
    @Test
    public void testToString() {
        assertEquals("Kernel(type=LINEAR)", Kernel.linear().toString());
        assertTrue(Kernel.rbf(0.5).toString().contains("RBF"));
        assertTrue(Kernel.polynomial(3).toString().contains("POLYNOMIAL"));
        assertTrue(Kernel.sigmoid(0.1, 0.5).toString().contains("SIGMOID"));
    }
}
