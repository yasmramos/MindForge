package io.github.yasmramos.mindforge.feature;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for RFE (Recursive Feature Elimination) feature selector.
 */
@DisplayName("RFE Tests")
class RFETest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Should create with default step of 1")
        void testDefaultStep() {
            RFE rfe = new RFE(5);
            assertEquals(5, rfe.getNFeaturesToSelect());
            assertEquals(1, rfe.getStep());
            assertFalse(rfe.isFitted());
        }
        
        @Test
        @DisplayName("Should create with custom step")
        void testCustomStep() {
            RFE rfe = new RFE(5, 2);
            assertEquals(5, rfe.getNFeaturesToSelect());
            assertEquals(2, rfe.getStep());
        }
        
        @Test
        @DisplayName("Should accept -1 for half of features")
        void testAutoSelect() {
            RFE rfe = new RFE(-1);
            assertEquals(-1, rfe.getNFeaturesToSelect());
        }
        
        @Test
        @DisplayName("Should reject invalid nFeaturesToSelect")
        void testInvalidNFeatures() {
            assertThrows(IllegalArgumentException.class, () -> new RFE(0));
            assertThrows(IllegalArgumentException.class, () -> new RFE(-2));
        }
        
        @Test
        @DisplayName("Should reject invalid step")
        void testInvalidStep() {
            assertThrows(IllegalArgumentException.class, () -> new RFE(5, 0));
            assertThrows(IllegalArgumentException.class, () -> new RFE(5, -1));
        }
    }
    
    @Nested
    @DisplayName("Fit Tests")
    class FitTests {
        
        @Test
        @DisplayName("Should fit and calculate rankings")
        void testFitCalculatesRankings() {
            double[][] X = {
                {1.0, 0.1, 2.0, 0.2},
                {1.1, 0.15, 2.1, 0.25},
                {5.0, 0.12, 2.0, 0.18},
                {5.1, 0.11, 2.1, 0.22}
            };
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(2);
            rfe.fit(X, y);
            
            assertTrue(rfe.isFitted());
            int[] ranking = rfe.getRanking();
            assertEquals(4, ranking.length);
        }
        
        @Test
        @DisplayName("Should select best features based on correlation")
        void testSelectsBestFeatures() {
            // Feature 0: highly correlated with y (good predictor)
            // Features 1-3: noise
            double[][] X = {
                {0.0, 0.5, 0.3, 0.7},
                {0.1, 0.4, 0.6, 0.2},
                {0.2, 0.6, 0.4, 0.5},
                {1.0, 0.5, 0.5, 0.4},
                {1.1, 0.3, 0.7, 0.6},
                {1.2, 0.7, 0.3, 0.3}
            };
            int[] y = {0, 0, 0, 1, 1, 1};
            
            RFE rfe = new RFE(1);
            rfe.fit(X, y);
            
            int[] selected = rfe.getSelectedFeatureIndices();
            assertEquals(1, selected.length);
            assertEquals(0, selected[0]); // Feature 0 should be selected
        }
        
        @Test
        @DisplayName("Should return this for method chaining")
        void testMethodChaining() {
            double[][] X = {{1.0, 2.0}, {3.0, 4.0}};
            int[] y = {0, 1};
            
            RFE rfe = new RFE(1);
            assertSame(rfe, rfe.fit(X, y));
        }
    }
    
    @Nested
    @DisplayName("Step Parameter Tests")
    class StepParameterTests {
        
        @Test
        @DisplayName("Should eliminate multiple features per step")
        void testMultipleStep() {
            double[][] X = new double[10][6];
            int[] y = new int[10];
            
            java.util.Random random = new java.util.Random(42);
            for (int i = 0; i < 10; i++) {
                y[i] = i < 5 ? 0 : 1;
                for (int j = 0; j < 6; j++) {
                    X[i][j] = (j == 0 ? y[i] * 10 : 0) + random.nextGaussian();
                }
            }
            
            RFE rfe = new RFE(2, 2); // Select 2, remove 2 at a time
            rfe.fit(X, y);
            
            assertEquals(2, rfe.getNumberOfSelectedFeatures());
        }
        
        @Test
        @DisplayName("Should handle step larger than remaining features")
        void testLargeStep() {
            double[][] X = {{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {0.0, 2.1, 3.1, 4.1}, {0.1, 6.1, 7.1, 8.1}};
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(1, 10); // Step larger than features
            rfe.fit(X, y);
            
            assertEquals(1, rfe.getNumberOfSelectedFeatures());
        }
    }
    
    @Nested
    @DisplayName("Transform Tests")
    class TransformTests {
        
        @Test
        @DisplayName("Should transform data to selected features")
        void testTransform() {
            double[][] X = {
                {1.0, 2.0, 3.0},
                {1.1, 2.1, 3.1},
                {5.0, 2.2, 3.2},
                {5.1, 2.3, 3.3}
            };
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(1);
            double[][] result = rfe.fitTransform(X, y);
            
            assertEquals(4, result.length);
            assertEquals(1, result[0].length);
        }
        
        @Test
        @DisplayName("Should throw if not fitted")
        void testTransformNotFitted() {
            RFE rfe = new RFE(2);
            double[][] X = {{1.0, 2.0}};
            
            assertThrows(IllegalStateException.class, () -> rfe.transform(X));
        }
        
        @Test
        @DisplayName("Should throw if feature count mismatch")
        void testTransformFeatureMismatch() {
            double[][] X_train = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            int[] y = {0, 1};
            double[][] X_test = {{1.0, 2.0}};
            
            RFE rfe = new RFE(2);
            rfe.fit(X_train, y);
            
            assertThrows(IllegalArgumentException.class, () -> rfe.transform(X_test));
        }
        
        @Test
        @DisplayName("Should preserve selected feature values")
        void testTransformPreservesValues() {
            double[][] X = {
                {100.0, 1.0, 2.0},
                {101.0, 1.1, 2.1},
                {0.0, 1.0, 2.0},
                {1.0, 1.1, 2.1}
            };
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(1);
            double[][] result = rfe.fitTransform(X, y);
            
            // Feature 0 should be selected (best discriminator)
            assertEquals(100.0, result[0][0], 1e-10);
            assertEquals(101.0, result[1][0], 1e-10);
        }
    }
    
    @Nested
    @DisplayName("Ranking Tests")
    class RankingTests {
        
        @Test
        @DisplayName("Should assign ranking 1 to selected features")
        void testSelectedFeaturesRanking() {
            double[][] X = {
                {1.0, 0.1, 0.2},
                {1.1, 0.15, 0.25},
                {5.0, 0.12, 0.18},
                {5.1, 0.11, 0.22}
            };
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(1);
            rfe.fit(X, y);
            
            int[] ranking = rfe.getRanking();
            int[] selected = rfe.getSelectedFeatureIndices();
            
            // Selected features should have ranking 1
            for (int idx : selected) {
                assertEquals(1, ranking[idx]);
            }
        }
        
        @Test
        @DisplayName("Should assign higher rankings to eliminated features")
        void testEliminatedFeaturesRanking() {
            double[][] X = {
                {1.0, 0.1, 0.2},
                {1.1, 0.15, 0.25},
                {5.0, 0.12, 0.18},
                {5.1, 0.11, 0.22}
            };
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(1);
            rfe.fit(X, y);
            
            int[] ranking = rfe.getRanking();
            boolean[] support = rfe.getSupport();
            
            // Non-selected features should have ranking > 1
            for (int i = 0; i < ranking.length; i++) {
                if (!support[i]) {
                    assertTrue(ranking[i] > 1);
                }
            }
        }
    }
    
    @Nested
    @DisplayName("Feature Importance Tests")
    class FeatureImportanceTests {
        
        @Test
        @DisplayName("Should calculate feature importances")
        void testFeatureImportances() {
            double[][] X = {
                {1.0, 0.5, 2.0},
                {1.1, 0.4, 2.1},
                {5.0, 0.6, 2.0},
                {5.1, 0.5, 2.1}
            };
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(2);
            rfe.fit(X, y);
            
            double[] importances = rfe.getFeatureImportances();
            assertEquals(3, importances.length);
            
            // Feature 0 should have highest importance
            assertTrue(importances[0] > importances[1]);
            assertTrue(importances[0] > importances[2]);
        }
        
        @Test
        @DisplayName("Should have non-negative importances")
        void testNonNegativeImportances() {
            double[][] X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(1);
            rfe.fit(X, y);
            
            double[] importances = rfe.getFeatureImportances();
            for (double imp : importances) {
                assertTrue(imp >= 0);
            }
        }
    }
    
    @Nested
    @DisplayName("Getter Tests")
    class GetterTests {
        
        @Test
        @DisplayName("Should return support mask")
        void testGetSupport() {
            double[][] X = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {0.0, 2.1, 3.1}, {0.1, 5.1, 6.1}};
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(2);
            rfe.fit(X, y);
            
            boolean[] support = rfe.getSupport();
            assertEquals(3, support.length);
            
            // Count selected
            int selected = 0;
            for (boolean b : support) {
                if (b) selected++;
            }
            assertEquals(2, selected);
        }
        
        @Test
        @DisplayName("Should return sorted selected indices")
        void testSortedSelectedIndices() {
            double[][] X = {{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {0.0, 2.1, 3.1, 4.1}, {0.1, 6.1, 7.1, 8.1}};
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(2);
            rfe.fit(X, y);
            
            int[] selected = rfe.getSelectedFeatureIndices();
            for (int i = 1; i < selected.length; i++) {
                assertTrue(selected[i] > selected[i-1]);
            }
        }
        
        @Test
        @DisplayName("Should throw on getters before fit")
        void testGettersBeforeFit() {
            RFE rfe = new RFE(2);
            
            assertThrows(IllegalStateException.class, () -> rfe.getRanking());
            assertThrows(IllegalStateException.class, () -> rfe.getFeatureImportances());
            assertThrows(IllegalStateException.class, () -> rfe.getSelectedFeatureIndices());
            assertThrows(IllegalStateException.class, () -> rfe.getSupport());
            assertThrows(IllegalStateException.class, () -> rfe.getNumberOfSelectedFeatures());
        }
    }
    
    @Nested
    @DisplayName("Input Validation Tests")
    class InputValidationTests {
        
        @Test
        @DisplayName("Should reject null X")
        void testNullX() {
            RFE rfe = new RFE(2);
            assertThrows(IllegalArgumentException.class, () -> rfe.fit(null, new int[]{0, 1}));
        }
        
        @Test
        @DisplayName("Should reject null y")
        void testNullY() {
            RFE rfe = new RFE(2);
            assertThrows(IllegalArgumentException.class, () -> rfe.fit(new double[][]{{1.0}}, null));
        }
        
        @Test
        @DisplayName("Should reject mismatched X and y lengths")
        void testMismatchedLengths() {
            RFE rfe = new RFE(2);
            double[][] X = {{1.0}, {2.0}, {3.0}};
            int[] y = {0, 1};
            
            assertThrows(IllegalArgumentException.class, () -> rfe.fit(X, y));
        }
        
        @Test
        @DisplayName("Should reject empty input")
        void testEmptyInput() {
            RFE rfe = new RFE(2);
            assertThrows(IllegalArgumentException.class, () -> rfe.fit(new double[0][], new int[0]));
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle nFeaturesToSelect greater than n_features")
        void testNFeaturesGreaterThanFeatures() {
            double[][] X = {{1.0, 2.0}, {3.0, 4.0}, {0.0, 2.1}, {0.1, 4.1}};
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(10); // More than 2 features
            double[][] result = rfe.fitTransform(X, y);
            
            // Should select all available features
            assertEquals(2, result[0].length);
        }
        
        @Test
        @DisplayName("Should handle auto selection (-1)")
        void testAutoSelection() {
            double[][] X = {{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {0.0, 2.1, 3.1, 4.1}, {0.1, 6.1, 7.1, 8.1}};
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(-1); // Auto = n_features / 2
            double[][] result = rfe.fitTransform(X, y);
            
            assertEquals(2, result[0].length); // 4 / 2 = 2
        }
        
        @Test
        @DisplayName("Should handle single feature selection")
        void testSingleFeatureSelection() {
            double[][] X = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            int[] y = {0, 1};
            
            RFE rfe = new RFE(1);
            rfe.fit(X, y);
            
            assertEquals(1, rfe.getNumberOfSelectedFeatures());
        }
        
        @Test
        @DisplayName("Should handle constant target")
        void testConstantTarget() {
            double[][] X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
            int[] y = {0, 0, 0}; // All same class
            
            RFE rfe = new RFE(1);
            rfe.fit(X, y);
            
            // Should still work, just with lower correlations
            assertEquals(1, rfe.getNumberOfSelectedFeatures());
        }
        
        @Test
        @DisplayName("Should handle multi-class targets")
        void testMultiClass() {
            double[][] X = {
                {1.0, 0.1},
                {1.1, 0.2},
                {2.0, 0.15},
                {2.1, 0.25},
                {3.0, 0.12},
                {3.1, 0.22}
            };
            int[] y = {0, 0, 1, 1, 2, 2};
            
            RFE rfe = new RFE(1);
            rfe.fit(X, y);
            
            assertTrue(rfe.isFitted());
        }
    }
    
    @Nested
    @DisplayName("ToString Tests")
    class ToStringTests {
        
        @Test
        @DisplayName("Should show unfitted state")
        void testToStringUnfitted() {
            RFE rfe = new RFE(5, 2);
            String str = rfe.toString();
            
            assertTrue(str.contains("nFeaturesToSelect=5"));
            assertTrue(str.contains("step=2"));
            assertTrue(str.contains("false"));
        }
        
        @Test
        @DisplayName("Should show fitted state")
        void testToStringFitted() {
            double[][] X = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {0.0, 2.1, 3.1}, {0.1, 5.1, 6.1}};
            int[] y = {0, 0, 1, 1};
            
            RFE rfe = new RFE(2);
            rfe.fit(X, y);
            
            String str = rfe.toString();
            assertTrue(str.contains("n_features_in=3"));
            assertTrue(str.contains("n_features_out=2"));
        }
    }
    
    @Nested
    @DisplayName("Large Dataset Tests")
    class LargeDatasetTests {
        
        @Test
        @DisplayName("Should handle large number of features efficiently")
        void testManyFeatures() {
            int nSamples = 100;
            int nFeatures = 100;
            double[][] X = new double[nSamples][nFeatures];
            int[] y = new int[nSamples];
            
            java.util.Random random = new java.util.Random(42);
            for (int i = 0; i < nSamples; i++) {
                y[i] = i < nSamples / 2 ? 0 : 1;
                for (int j = 0; j < nFeatures; j++) {
                    // First 5 features are informative
                    if (j < 5) {
                        X[i][j] = y[i] * 10.0 + random.nextGaussian();
                    } else {
                        X[i][j] = random.nextGaussian();
                    }
                }
            }
            
            RFE rfe = new RFE(10, 5); // Select 10, remove 5 at a time
            double[][] result = rfe.fitTransform(X, y);
            
            assertEquals(nSamples, result.length);
            assertEquals(10, result[0].length);
        }
    }
}
