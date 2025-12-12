package io.github.yasmramos.mindforge.decomposition;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for PCA (Principal Component Analysis).
 */
@DisplayName("PCA Tests")
class PCATest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Should create with specified number of components")
        void testNComponents() {
            PCA pca = new PCA(3);
            assertEquals(3, pca.getNComponents());
            assertFalse(pca.isFitted());
        }
        
        @Test
        @DisplayName("Should create with default all components")
        void testDefaultConstructor() {
            PCA pca = new PCA();
            assertEquals(-1, pca.getNComponents());
        }
        
        @Test
        @DisplayName("Should accept -1 for all components")
        void testAllComponents() {
            PCA pca = new PCA(-1);
            assertEquals(-1, pca.getNComponents());
        }
        
        @Test
        @DisplayName("Should reject invalid nComponents")
        void testInvalidNComponents() {
            assertThrows(IllegalArgumentException.class, () -> new PCA(0));
            assertThrows(IllegalArgumentException.class, () -> new PCA(-2));
        }
    }
    
    @Nested
    @DisplayName("Fit Tests")
    class FitTests {
        
        @Test
        @DisplayName("Should fit and compute components")
        void testFitComputes() {
            double[][] X = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0},
                {10.0, 11.0, 12.0}
            };
            
            PCA pca = new PCA(2);
            pca.fit(X);
            
            assertTrue(pca.isFitted());
            assertEquals(2, pca.getNumberOfComponents());
            assertEquals(3, pca.getNumberOfFeatures());
        }
        
        @Test
        @DisplayName("Should compute mean correctly")
        void testMeanComputation() {
            double[][] X = {
                {2.0, 4.0},
                {4.0, 6.0},
                {6.0, 8.0}
            };
            
            PCA pca = new PCA();
            pca.fit(X);
            
            double[] mean = pca.getMean();
            assertEquals(4.0, mean[0], 1e-10);
            assertEquals(6.0, mean[1], 1e-10);
        }
        
        @Test
        @DisplayName("Should return this for method chaining")
        void testMethodChaining() {
            double[][] X = {{1.0, 2.0}, {3.0, 4.0}};
            PCA pca = new PCA(1);
            assertSame(pca, pca.fit(X));
        }
        
        @Test
        @DisplayName("Should reject null input")
        void testNullInput() {
            PCA pca = new PCA(2);
            assertThrows(IllegalArgumentException.class, () -> pca.fit(null));
        }
        
        @Test
        @DisplayName("Should reject empty input")
        void testEmptyInput() {
            PCA pca = new PCA(2);
            assertThrows(IllegalArgumentException.class, () -> pca.fit(new double[0][]));
        }
        
        @Test
        @DisplayName("Should limit components to min(n_samples, n_features)")
        void testComponentLimit() {
            double[][] X = {
                {1.0, 2.0, 3.0, 4.0, 5.0},
                {2.0, 3.0, 4.0, 5.0, 6.0},
                {3.0, 4.0, 5.0, 6.0, 7.0}
            };
            
            PCA pca = new PCA(10); // Request more than possible
            pca.fit(X);
            
            // Should be limited to min(3 samples, 5 features) = 3
            assertTrue(pca.getNumberOfComponents() <= 3);
        }
    }
    
    @Nested
    @DisplayName("Transform Tests")
    class TransformTests {
        
        @Test
        @DisplayName("Should transform data to lower dimensions")
        void testTransformReducesDimensions() {
            double[][] X = {
                {1.0, 2.0, 3.0, 4.0},
                {2.0, 3.0, 4.0, 5.0},
                {3.0, 4.0, 5.0, 6.0},
                {4.0, 5.0, 6.0, 7.0}
            };
            
            PCA pca = new PCA(2);
            double[][] result = pca.fitTransform(X);
            
            assertEquals(4, result.length);
            assertEquals(2, result[0].length);
        }
        
        @Test
        @DisplayName("Should throw if not fitted")
        void testTransformNotFitted() {
            PCA pca = new PCA(2);
            double[][] X = {{1.0, 2.0}};
            
            assertThrows(IllegalStateException.class, () -> pca.transform(X));
        }
        
        @Test
        @DisplayName("Should throw if feature count mismatch")
        void testTransformFeatureMismatch() {
            double[][] X_train = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            double[][] X_test = {{1.0, 2.0}};
            
            PCA pca = new PCA(2);
            pca.fit(X_train);
            
            assertThrows(IllegalArgumentException.class, () -> pca.transform(X_test));
        }
        
        @Test
        @DisplayName("Should preserve correct output shape")
        void testOutputShape() {
            double[][] X = new double[100][10];
            java.util.Random random = new java.util.Random(42);
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 10; j++) {
                    X[i][j] = random.nextGaussian();
                }
            }
            
            PCA pca = new PCA(5);
            double[][] result = pca.fitTransform(X);
            
            assertEquals(100, result.length);
            assertEquals(5, result[0].length);
        }
    }
    
    @Nested
    @DisplayName("Inverse Transform Tests")
    class InverseTransformTests {
        
        @Test
        @DisplayName("Should reconstruct data approximately")
        void testInverseTransformReconstructs() {
            double[][] X = {
                {1.0, 2.0},
                {3.0, 4.0},
                {5.0, 6.0},
                {7.0, 8.0}
            };
            
            // Keep all components - should reconstruct perfectly
            PCA pca = new PCA(-1);
            double[][] transformed = pca.fitTransform(X);
            double[][] reconstructed = pca.inverseTransform(transformed);
            
            for (int i = 0; i < X.length; i++) {
                for (int j = 0; j < X[0].length; j++) {
                    assertEquals(X[i][j], reconstructed[i][j], 1e-6);
                }
            }
        }
        
        @Test
        @DisplayName("Should throw if dimensions mismatch")
        void testInverseTransformDimensionMismatch() {
            double[][] X = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            
            PCA pca = new PCA(2);
            pca.fit(X);
            
            // Wrong number of components
            double[][] wrong = {{1.0, 2.0, 3.0}};
            assertThrows(IllegalArgumentException.class, () -> pca.inverseTransform(wrong));
        }
        
        @Test
        @DisplayName("Should throw if not fitted")
        void testInverseTransformNotFitted() {
            PCA pca = new PCA(2);
            assertThrows(IllegalStateException.class, () -> 
                pca.inverseTransform(new double[][]{{1.0, 2.0}}));
        }
    }
    
    @Nested
    @DisplayName("Variance Tests")
    class VarianceTests {
        
        @Test
        @DisplayName("Should compute explained variance")
        void testExplainedVariance() {
            double[][] X = generateRandomData(100, 5, 42);
            
            PCA pca = new PCA(3);
            pca.fit(X);
            
            double[] variance = pca.getExplainedVariance();
            assertEquals(3, variance.length);
            
            // Variances should be positive
            for (double v : variance) {
                assertTrue(v >= 0);
            }
            
            // Variances should be in decreasing order
            for (int i = 1; i < variance.length; i++) {
                assertTrue(variance[i-1] >= variance[i] - 1e-10);
            }
        }
        
        @Test
        @DisplayName("Should compute explained variance ratio")
        void testExplainedVarianceRatio() {
            double[][] X = generateRandomData(100, 5, 42);
            
            PCA pca = new PCA(-1);
            pca.fit(X);
            
            double[] ratio = pca.getExplainedVarianceRatio();
            
            // Ratios should be between 0 and 1
            for (double r : ratio) {
                assertTrue(r >= 0 && r <= 1);
            }
            
            // Sum should be approximately 1 when keeping all components
            double sum = 0;
            for (double r : ratio) {
                sum += r;
            }
            assertEquals(1.0, sum, 0.01);
        }
        
        @Test
        @DisplayName("Should compute cumulative explained variance ratio")
        void testCumulativeVarianceRatio() {
            double[][] X = generateRandomData(50, 5, 42);
            
            PCA pca = new PCA(3);
            pca.fit(X);
            
            double[] cumulative = pca.getCumulativeExplainedVarianceRatio();
            
            // Should be increasing
            for (int i = 1; i < cumulative.length; i++) {
                assertTrue(cumulative[i] >= cumulative[i-1]);
            }
            
            // Should be between 0 and 1
            for (double c : cumulative) {
                assertTrue(c >= 0 && c <= 1);
            }
        }
        
        @Test
        @DisplayName("Should have positive singular values")
        void testSingularValues() {
            double[][] X = generateRandomData(50, 5, 42);
            
            PCA pca = new PCA(3);
            pca.fit(X);
            
            double[] sv = pca.getSingularValues();
            assertEquals(3, sv.length);
            
            // Should be positive and decreasing
            for (int i = 0; i < sv.length; i++) {
                assertTrue(sv[i] >= 0);
                if (i > 0) {
                    assertTrue(sv[i-1] >= sv[i] - 1e-10);
                }
            }
        }
    }
    
    @Nested
    @DisplayName("Components Tests")
    class ComponentsTests {
        
        @Test
        @DisplayName("Should return orthogonal components")
        void testOrthogonalComponents() {
            double[][] X = generateRandomData(100, 5, 42);
            
            PCA pca = new PCA(3);
            pca.fit(X);
            
            double[][] components = pca.getComponents();
            
            // Check orthogonality (dot product should be ~0)
            for (int i = 0; i < components.length; i++) {
                for (int j = i + 1; j < components.length; j++) {
                    double dot = 0.0;
                    for (int k = 0; k < components[i].length; k++) {
                        dot += components[i][k] * components[j][k];
                    }
                    assertEquals(0.0, dot, 1e-6);
                }
            }
        }
        
        @Test
        @DisplayName("Should return unit-length components")
        void testUnitComponents() {
            double[][] X = generateRandomData(100, 5, 42);
            
            PCA pca = new PCA(3);
            pca.fit(X);
            
            double[][] components = pca.getComponents();
            
            for (double[] component : components) {
                double norm = 0.0;
                for (double v : component) {
                    norm += v * v;
                }
                assertEquals(1.0, Math.sqrt(norm), 1e-6);
            }
        }
        
        @Test
        @DisplayName("Should have correct component dimensions")
        void testComponentDimensions() {
            double[][] X = {{1.0, 2.0, 3.0, 4.0}, {2.0, 3.0, 4.0, 5.0}, {3.0, 4.0, 5.0, 6.0}};
            
            PCA pca = new PCA(2);
            pca.fit(X);
            
            double[][] components = pca.getComponents();
            assertEquals(2, components.length);
            assertEquals(4, components[0].length);
        }
    }
    
    @Nested
    @DisplayName("Getter Tests")
    class GetterTests {
        
        @Test
        @DisplayName("Should throw on getters before fit")
        void testGettersBeforeFit() {
            PCA pca = new PCA(2);
            
            assertThrows(IllegalStateException.class, () -> pca.getComponents());
            assertThrows(IllegalStateException.class, () -> pca.getExplainedVariance());
            assertThrows(IllegalStateException.class, () -> pca.getExplainedVarianceRatio());
            assertThrows(IllegalStateException.class, () -> pca.getSingularValues());
            assertThrows(IllegalStateException.class, () -> pca.getMean());
            assertThrows(IllegalStateException.class, () -> pca.getNumberOfComponents());
            assertThrows(IllegalStateException.class, () -> pca.getNumberOfFeatures());
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle single feature")
        void testSingleFeature() {
            double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}};
            
            PCA pca = new PCA(1);
            double[][] result = pca.fitTransform(X);
            
            assertEquals(4, result.length);
            assertEquals(1, result[0].length);
        }
        
        @Test
        @DisplayName("Should handle two samples")
        void testTwoSamples() {
            double[][] X = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            
            PCA pca = new PCA(1);
            double[][] result = pca.fitTransform(X);
            
            assertEquals(2, result.length);
            assertEquals(1, result[0].length);
        }
        
        @Test
        @DisplayName("Should handle perfectly correlated features")
        void testCorrelatedFeatures() {
            double[][] X = new double[10][3];
            for (int i = 0; i < 10; i++) {
                X[i][0] = i;
                X[i][1] = 2 * i;  // Perfectly correlated
                X[i][2] = 3 * i;  // Perfectly correlated
            }
            
            PCA pca = new PCA(2);
            double[][] result = pca.fitTransform(X);
            
            assertEquals(10, result.length);
            // First component should capture most variance
            assertTrue(pca.getExplainedVarianceRatio()[0] > 0.9);
        }
        
        @Test
        @DisplayName("Should handle negative values")
        void testNegativeValues() {
            double[][] X = {
                {-1.0, -2.0},
                {-3.0, -4.0},
                {1.0, 2.0},
                {3.0, 4.0}
            };
            
            PCA pca = new PCA(2);
            pca.fit(X);
            
            assertTrue(pca.isFitted());
        }
    }
    
    @Nested
    @DisplayName("ToString Tests")
    class ToStringTests {
        
        @Test
        @DisplayName("Should show unfitted state")
        void testToStringUnfitted() {
            PCA pca = new PCA(3);
            String str = pca.toString();
            
            assertTrue(str.contains("n_components=3"));
            assertTrue(str.contains("false"));
        }
        
        @Test
        @DisplayName("Should show fitted state with variance")
        void testToStringFitted() {
            double[][] X = generateRandomData(50, 5, 42);
            
            PCA pca = new PCA(3);
            pca.fit(X);
            
            String str = pca.toString();
            assertTrue(str.contains("n_components=3"));
            assertTrue(str.contains("n_features=5"));
            assertTrue(str.contains("explained_variance="));
        }
    }
    
    @Nested
    @DisplayName("Large Dataset Tests")
    class LargeDatasetTests {
        
        @Test
        @DisplayName("Should handle large number of features")
        void testManyFeatures() {
            double[][] X = generateRandomData(200, 100, 42);
            
            PCA pca = new PCA(10);
            double[][] result = pca.fitTransform(X);
            
            assertEquals(200, result.length);
            assertEquals(10, result[0].length);
        }
        
        @Test
        @DisplayName("Should handle large number of samples")
        void testManySamples() {
            double[][] X = generateRandomData(1000, 20, 42);
            
            PCA pca = new PCA(5);
            double[][] result = pca.fitTransform(X);
            
            assertEquals(1000, result.length);
            assertEquals(5, result[0].length);
        }
    }
    
    // Helper method to generate random data
    private double[][] generateRandomData(int nSamples, int nFeatures, long seed) {
        double[][] X = new double[nSamples][nFeatures];
        java.util.Random random = new java.util.Random(seed);
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = random.nextGaussian() * (j + 1);
            }
        }
        
        return X;
    }
}
