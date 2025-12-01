package com.mindforge.feature;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for VarianceThreshold feature selector.
 */
@DisplayName("VarianceThreshold Tests")
class VarianceThresholdTest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Should create with default threshold of 0.0")
        void testDefaultConstructor() {
            VarianceThreshold selector = new VarianceThreshold();
            assertEquals(0.0, selector.getThreshold());
            assertFalse(selector.isFitted());
        }
        
        @Test
        @DisplayName("Should create with custom threshold")
        void testCustomThreshold() {
            VarianceThreshold selector = new VarianceThreshold(0.5);
            assertEquals(0.5, selector.getThreshold());
        }
        
        @Test
        @DisplayName("Should reject negative threshold")
        void testNegativeThreshold() {
            assertThrows(IllegalArgumentException.class, () -> 
                new VarianceThreshold(-0.1));
        }
        
        @Test
        @DisplayName("Should accept zero threshold")
        void testZeroThreshold() {
            VarianceThreshold selector = new VarianceThreshold(0.0);
            assertEquals(0.0, selector.getThreshold());
        }
    }
    
    @Nested
    @DisplayName("Fit Tests")
    class FitTests {
        
        @Test
        @DisplayName("Should fit and calculate variances correctly")
        void testFitCalculatesVariances() {
            double[][] X = {
                {0.0, 2.0, 0.0, 3.0},
                {0.0, 1.0, 4.0, 3.0},
                {0.0, 1.0, 1.0, 3.0}
            };
            
            VarianceThreshold selector = new VarianceThreshold();
            selector.fit(X);
            
            assertTrue(selector.isFitted());
            double[] variances = selector.getVariances();
            assertEquals(4, variances.length);
            
            // First feature has 0 variance (all 0s)
            assertEquals(0.0, variances[0], 1e-10);
            // Last feature has 0 variance (all 3s)
            assertEquals(0.0, variances[3], 1e-10);
        }
        
        @Test
        @DisplayName("Should remove zero-variance features with default threshold")
        void testRemoveZeroVarianceFeatures() {
            double[][] X = {
                {0.0, 2.0, 0.0, 3.0},
                {0.0, 1.0, 4.0, 3.0},
                {0.0, 1.0, 1.0, 3.0}
            };
            
            VarianceThreshold selector = new VarianceThreshold();
            double[][] result = selector.fitTransform(X);
            
            // Should keep only features 1 and 2 (indices)
            assertEquals(3, result.length);
            assertEquals(2, result[0].length);
        }
        
        @Test
        @DisplayName("Should handle single sample")
        void testSingleSample() {
            double[][] X = {{1.0, 2.0, 3.0}};
            
            VarianceThreshold selector = new VarianceThreshold();
            // All variances will be 0, should throw
            assertThrows(IllegalStateException.class, () -> selector.fit(X));
        }
        
        @Test
        @DisplayName("Should return this for method chaining")
        void testMethodChaining() {
            double[][] X = {{1.0, 2.0}, {3.0, 4.0}};
            VarianceThreshold selector = new VarianceThreshold();
            assertSame(selector, selector.fit(X));
        }
        
        @Test
        @DisplayName("Should reject null input")
        void testNullInput() {
            VarianceThreshold selector = new VarianceThreshold();
            assertThrows(IllegalArgumentException.class, () -> selector.fit(null));
        }
        
        @Test
        @DisplayName("Should reject empty input")
        void testEmptyInput() {
            VarianceThreshold selector = new VarianceThreshold();
            assertThrows(IllegalArgumentException.class, () -> selector.fit(new double[0][]));
        }
    }
    
    @Nested
    @DisplayName("Transform Tests")
    class TransformTests {
        
        @Test
        @DisplayName("Should transform data to selected features")
        void testTransform() {
            double[][] X = {
                {0.0, 2.0, 0.0},
                {0.0, 1.0, 4.0},
                {0.0, 1.0, 1.0}
            };
            
            VarianceThreshold selector = new VarianceThreshold();
            selector.fit(X);
            double[][] result = selector.transform(X);
            
            assertEquals(3, result.length);
            assertEquals(2, result[0].length); // Features 1 and 2
        }
        
        @Test
        @DisplayName("Should throw if not fitted")
        void testTransformNotFitted() {
            VarianceThreshold selector = new VarianceThreshold();
            double[][] X = {{1.0, 2.0}};
            
            assertThrows(IllegalStateException.class, () -> selector.transform(X));
        }
        
        @Test
        @DisplayName("Should throw if feature count mismatch")
        void testTransformFeatureMismatch() {
            double[][] X_train = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            double[][] X_test = {{1.0, 2.0}};
            
            VarianceThreshold selector = new VarianceThreshold();
            selector.fit(X_train);
            
            assertThrows(IllegalArgumentException.class, () -> selector.transform(X_test));
        }
        
        @Test
        @DisplayName("Should preserve selected feature values correctly")
        void testTransformPreservesValues() {
            double[][] X = {
                {5.0, 1.0, 2.0},  // First column constant
                {5.0, 3.0, 4.0},
                {5.0, 5.0, 6.0}
            };
            
            VarianceThreshold selector = new VarianceThreshold();
            selector.fit(X);
            double[][] result = selector.transform(X);
            
            // Should have removed first column
            assertEquals(1.0, result[0][0], 1e-10);
            assertEquals(2.0, result[0][1], 1e-10);
            assertEquals(3.0, result[1][0], 1e-10);
            assertEquals(4.0, result[1][1], 1e-10);
        }
    }
    
    @Nested
    @DisplayName("Threshold Behavior Tests")
    class ThresholdBehaviorTests {
        
        @Test
        @DisplayName("Should filter by custom threshold")
        void testCustomThresholdFilter() {
            double[][] X = {
                {1.0, 1.0, 10.0},
                {1.1, 2.0, 20.0},
                {0.9, 3.0, 30.0}
            };
            
            // High threshold should remove low-variance features
            VarianceThreshold selector = new VarianceThreshold(5.0);
            selector.fit(X);
            
            // Only the third feature should remain (high variance)
            assertEquals(1, selector.getNumberOfSelectedFeatures());
        }
        
        @Test
        @DisplayName("Should throw if no features meet threshold")
        void testNoFeaturesAboveThreshold() {
            double[][] X = {
                {1.0, 2.0},
                {1.1, 2.1},
                {0.9, 1.9}
            };
            
            VarianceThreshold selector = new VarianceThreshold(100.0);
            assertThrows(IllegalStateException.class, () -> selector.fit(X));
        }
    }
    
    @Nested
    @DisplayName("Getter Tests")
    class GetterTests {
        
        @Test
        @DisplayName("Should return correct selected feature indices")
        void testGetSelectedFeatureIndices() {
            double[][] X = {
                {0.0, 1.0, 0.0, 2.0},
                {0.0, 2.0, 0.0, 3.0},
                {0.0, 3.0, 0.0, 4.0}
            };
            
            VarianceThreshold selector = new VarianceThreshold();
            selector.fit(X);
            
            int[] selected = selector.getSelectedFeatureIndices();
            assertArrayEquals(new int[]{1, 3}, selected);
        }
        
        @Test
        @DisplayName("Should return correct support mask")
        void testGetSupport() {
            double[][] X = {
                {0.0, 1.0, 0.0},
                {0.0, 2.0, 0.0},
                {0.0, 3.0, 0.0}
            };
            
            VarianceThreshold selector = new VarianceThreshold();
            selector.fit(X);
            
            boolean[] support = selector.getSupport();
            assertFalse(support[0]);
            assertTrue(support[1]);
            assertFalse(support[2]);
        }
        
        @Test
        @DisplayName("Should throw on getters before fit")
        void testGettersBeforeFit() {
            VarianceThreshold selector = new VarianceThreshold();
            
            assertThrows(IllegalStateException.class, () -> selector.getVariances());
            assertThrows(IllegalStateException.class, () -> selector.getSelectedFeatureIndices());
            assertThrows(IllegalStateException.class, () -> selector.getSupport());
            assertThrows(IllegalStateException.class, () -> selector.getNumberOfSelectedFeatures());
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle all features passing threshold")
        void testAllFeaturesPass() {
            double[][] X = {
                {1.0, 10.0, 100.0},
                {2.0, 20.0, 200.0},
                {3.0, 30.0, 300.0}
            };
            
            VarianceThreshold selector = new VarianceThreshold();
            selector.fit(X);
            
            assertEquals(3, selector.getNumberOfSelectedFeatures());
        }
        
        @Test
        @DisplayName("Should handle large dataset")
        void testLargeDataset() {
            int nSamples = 1000;
            int nFeatures = 50;
            double[][] X = new double[nSamples][nFeatures];
            
            java.util.Random random = new java.util.Random(42);
            for (int i = 0; i < nSamples; i++) {
                for (int j = 0; j < nFeatures; j++) {
                    // Some features with low variance, some with high
                    if (j % 5 == 0) {
                        X[i][j] = 1.0 + random.nextDouble() * 0.01; // Low variance
                    } else {
                        X[i][j] = random.nextDouble() * 100; // High variance
                    }
                }
            }
            
            VarianceThreshold selector = new VarianceThreshold(0.1);
            double[][] result = selector.fitTransform(X);
            
            // Should have removed some low-variance features
            assertTrue(result[0].length < nFeatures);
            assertTrue(result[0].length > 0);
        }
        
        @Test
        @DisplayName("Should handle negative values")
        void testNegativeValues() {
            double[][] X = {
                {-1.0, -10.0},
                {-2.0, -20.0},
                {-3.0, -30.0}
            };
            
            VarianceThreshold selector = new VarianceThreshold();
            selector.fit(X);
            
            double[] variances = selector.getVariances();
            assertTrue(variances[0] > 0);
            assertTrue(variances[1] > 0);
        }
    }
    
    @Nested
    @DisplayName("ToString Tests")
    class ToStringTests {
        
        @Test
        @DisplayName("Should show unfitted state")
        void testToStringUnfitted() {
            VarianceThreshold selector = new VarianceThreshold(0.5);
            String str = selector.toString();
            
            assertTrue(str.contains("0.5"));
            assertTrue(str.contains("false"));
        }
        
        @Test
        @DisplayName("Should show fitted state with details")
        void testToStringFitted() {
            double[][] X = {{1.0, 2.0, 3.0}, {4.0, 5.0, 3.0}};
            VarianceThreshold selector = new VarianceThreshold();
            selector.fit(X);
            
            String str = selector.toString();
            assertTrue(str.contains("n_features_in=3"));
            assertTrue(str.contains("n_features_out=2"));
        }
    }
}
