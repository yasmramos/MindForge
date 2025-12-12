package io.github.yasmramos.mindforge.feature;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for SelectKBest feature selector.
 */
@DisplayName("SelectKBest Tests")
class SelectKBestTest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Should create with default F_CLASSIF scoring")
        void testDefaultScoring() {
            SelectKBest selector = new SelectKBest(5);
            assertEquals(SelectKBest.ScoreFunction.F_CLASSIF, selector.getScoreFunction());
            assertEquals(5, selector.getK());
        }
        
        @Test
        @DisplayName("Should create with custom scoring function")
        void testCustomScoring() {
            SelectKBest selector = new SelectKBest(SelectKBest.ScoreFunction.CHI2, 3);
            assertEquals(SelectKBest.ScoreFunction.CHI2, selector.getScoreFunction());
            assertEquals(3, selector.getK());
        }
        
        @Test
        @DisplayName("Should accept -1 for all features")
        void testSelectAll() {
            SelectKBest selector = new SelectKBest(SelectKBest.ScoreFunction.F_CLASSIF, -1);
            assertEquals(-1, selector.getK());
        }
        
        @Test
        @DisplayName("Should reject invalid k values")
        void testInvalidK() {
            assertThrows(IllegalArgumentException.class, () -> new SelectKBest(0));
            assertThrows(IllegalArgumentException.class, () -> new SelectKBest(-2));
        }
    }
    
    @Nested
    @DisplayName("F_CLASSIF Tests")
    class FClassifTests {
        
        @Test
        @DisplayName("Should select features with highest F-values")
        void testFClassifSelection() {
            // Feature 0: perfectly separates classes (high F-value)
            // Feature 1: random noise (low F-value)
            // Feature 2: somewhat separates classes (medium F-value)
            double[][] X = {
                {1.0, 0.5, 1.2},
                {1.1, 0.3, 1.1},
                {1.2, 0.8, 1.3},
                {5.0, 0.4, 3.8},
                {5.1, 0.6, 4.0},
                {5.2, 0.2, 3.9}
            };
            int[] y = {0, 0, 0, 1, 1, 1};
            
            SelectKBest selector = new SelectKBest(SelectKBest.ScoreFunction.F_CLASSIF, 2);
            selector.fit(X, y);
            
            double[] scores = selector.getScores();
            // Feature 0 and 2 should have higher scores than feature 1
            assertTrue(scores[0] > scores[1]);
            assertTrue(scores[2] > scores[1]);
        }
        
        @Test
        @DisplayName("Should transform to k features")
        void testTransformToKFeatures() {
            double[][] X = {
                {1.0, 2.0, 3.0, 4.0},
                {1.1, 2.1, 3.1, 4.1},
                {5.0, 2.2, 3.2, 4.2},
                {5.1, 2.3, 3.3, 4.3}
            };
            int[] y = {0, 0, 1, 1};
            
            SelectKBest selector = new SelectKBest(2);
            double[][] result = selector.fitTransform(X, y);
            
            assertEquals(4, result.length);
            assertEquals(2, result[0].length);
        }
    }
    
    @Nested
    @DisplayName("CHI2 Tests")
    class Chi2Tests {
        
        @Test
        @DisplayName("Should work with non-negative features")
        void testChi2NonNegative() {
            double[][] X = {
                {1.0, 0.0, 2.0},
                {2.0, 0.0, 1.0},
                {0.0, 3.0, 0.5},
                {0.5, 4.0, 0.3}
            };
            int[] y = {0, 0, 1, 1};
            
            SelectKBest selector = new SelectKBest(SelectKBest.ScoreFunction.CHI2, 2);
            selector.fit(X, y);
            
            assertTrue(selector.isFitted());
            assertEquals(2, selector.getSelectedFeatureIndices().length);
        }
        
        @Test
        @DisplayName("Should reject negative features for CHI2")
        void testChi2RejectsNegative() {
            double[][] X = {
                {1.0, -0.5},
                {2.0, 0.5}
            };
            int[] y = {0, 1};
            
            SelectKBest selector = new SelectKBest(SelectKBest.ScoreFunction.CHI2, 1);
            assertThrows(IllegalArgumentException.class, () -> selector.fit(X, y));
        }
    }
    
    @Nested
    @DisplayName("MUTUAL_INFO Tests")
    class MutualInfoTests {
        
        @Test
        @DisplayName("Should calculate mutual information")
        void testMutualInfo() {
            // Create data where feature 0 is highly informative
            double[][] X = {
                {0.1, 0.5},
                {0.2, 0.6},
                {0.15, 0.4},
                {0.9, 0.55},
                {0.85, 0.45},
                {0.95, 0.5}
            };
            int[] y = {0, 0, 0, 1, 1, 1};
            
            SelectKBest selector = new SelectKBest(SelectKBest.ScoreFunction.MUTUAL_INFO, 1);
            selector.fit(X, y);
            
            double[] scores = selector.getScores();
            // Feature 0 should have higher MI with y
            assertTrue(scores[0] >= scores[1]);
        }
    }
    
    @Nested
    @DisplayName("Transform Tests")
    class TransformTests {
        
        @Test
        @DisplayName("Should throw if not fitted")
        void testTransformNotFitted() {
            SelectKBest selector = new SelectKBest(2);
            double[][] X = {{1.0, 2.0}};
            
            assertThrows(IllegalStateException.class, () -> selector.transform(X));
        }
        
        @Test
        @DisplayName("Should throw if feature count mismatch")
        void testTransformFeatureMismatch() {
            double[][] X_train = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            int[] y = {0, 1};
            double[][] X_test = {{1.0, 2.0}};
            
            SelectKBest selector = new SelectKBest(2);
            selector.fit(X_train, y);
            
            assertThrows(IllegalArgumentException.class, () -> selector.transform(X_test));
        }
        
        @Test
        @DisplayName("Should preserve correct feature values")
        void testTransformPreservesValues() {
            double[][] X = {
                {100.0, 1.0, 2.0},  // Feature 0 has high variance/scores
                {101.0, 1.1, 2.1},
                {0.0, 1.0, 2.0},
                {1.0, 1.1, 2.1}
            };
            int[] y = {0, 0, 1, 1};
            
            SelectKBest selector = new SelectKBest(1);
            double[][] result = selector.fitTransform(X, y);
            
            // Should select feature 0 (best discriminator)
            assertEquals(4, result.length);
            assertEquals(1, result[0].length);
        }
    }
    
    @Nested
    @DisplayName("Getter Tests")
    class GetterTests {
        
        @Test
        @DisplayName("Should return scores after fit")
        void testGetScores() {
            double[][] X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};
            int[] y = {0, 0, 1, 1};
            
            SelectKBest selector = new SelectKBest(1);
            selector.fit(X, y);
            
            double[] scores = selector.getScores();
            assertEquals(2, scores.length);
        }
        
        @Test
        @DisplayName("Should return p-values after fit")
        void testGetPValues() {
            double[][] X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};
            int[] y = {0, 0, 1, 1};
            
            SelectKBest selector = new SelectKBest(SelectKBest.ScoreFunction.F_CLASSIF, 1);
            selector.fit(X, y);
            
            double[] pValues = selector.getPValues();
            assertEquals(2, pValues.length);
            for (double p : pValues) {
                assertTrue(p >= 0 && p <= 1);
            }
        }
        
        @Test
        @DisplayName("Should return support mask")
        void testGetSupport() {
            double[][] X = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {0.0, 2.1, 3.1}, {0.1, 5.1, 6.1}};
            int[] y = {0, 0, 1, 1};
            
            SelectKBest selector = new SelectKBest(2);
            selector.fit(X, y);
            
            boolean[] support = selector.getSupport();
            assertEquals(3, support.length);
            
            // Count selected
            int selected = 0;
            for (boolean b : support) {
                if (b) selected++;
            }
            assertEquals(2, selected);
        }
        
        @Test
        @DisplayName("Should throw on getters before fit")
        void testGettersBeforeFit() {
            SelectKBest selector = new SelectKBest(2);
            
            assertThrows(IllegalStateException.class, () -> selector.getScores());
            assertThrows(IllegalStateException.class, () -> selector.getPValues());
            assertThrows(IllegalStateException.class, () -> selector.getSelectedFeatureIndices());
            assertThrows(IllegalStateException.class, () -> selector.getSupport());
        }
    }
    
    @Nested
    @DisplayName("Input Validation Tests")
    class InputValidationTests {
        
        @Test
        @DisplayName("Should reject null X")
        void testNullX() {
            SelectKBest selector = new SelectKBest(2);
            assertThrows(IllegalArgumentException.class, () -> selector.fit(null, new int[]{0, 1}));
        }
        
        @Test
        @DisplayName("Should reject null y")
        void testNullY() {
            SelectKBest selector = new SelectKBest(2);
            assertThrows(IllegalArgumentException.class, () -> selector.fit(new double[][]{{1.0}}, null));
        }
        
        @Test
        @DisplayName("Should reject mismatched X and y lengths")
        void testMismatchedLengths() {
            SelectKBest selector = new SelectKBest(2);
            double[][] X = {{1.0}, {2.0}, {3.0}};
            int[] y = {0, 1};
            
            assertThrows(IllegalArgumentException.class, () -> selector.fit(X, y));
        }
        
        @Test
        @DisplayName("Should reject empty input")
        void testEmptyInput() {
            SelectKBest selector = new SelectKBest(2);
            assertThrows(IllegalArgumentException.class, () -> selector.fit(new double[0][], new int[0]));
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle k greater than n_features")
        void testKGreaterThanFeatures() {
            double[][] X = {{1.0, 2.0}, {3.0, 4.0}, {0.0, 2.1}, {0.1, 4.1}};
            int[] y = {0, 0, 1, 1};
            
            SelectKBest selector = new SelectKBest(10); // k > 2 features
            double[][] result = selector.fitTransform(X, y);
            
            // Should select all available features
            assertEquals(2, result[0].length);
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
            
            SelectKBest selector = new SelectKBest(1);
            selector.fit(X, y);
            
            assertTrue(selector.isFitted());
        }
        
        @Test
        @DisplayName("Should handle identical feature values")
        void testIdenticalFeatures() {
            double[][] X = {
                {1.0, 5.0, 5.0},
                {2.0, 5.0, 5.0},
                {3.0, 5.0, 5.0},
                {4.0, 5.0, 5.0}
            };
            int[] y = {0, 0, 1, 1};
            
            SelectKBest selector = new SelectKBest(1);
            selector.fit(X, y);
            
            // Should select feature 0 (only one with variance)
            int[] selected = selector.getSelectedFeatureIndices();
            assertEquals(0, selected[0]);
        }
        
        @Test
        @DisplayName("Should maintain feature order in selection")
        void testFeatureOrder() {
            double[][] X = {
                {1.0, 100.0, 10.0, 1000.0},
                {2.0, 99.0, 11.0, 999.0},
                {10.0, 1.0, 100.0, 10.0},
                {11.0, 2.0, 99.0, 11.0}
            };
            int[] y = {0, 0, 1, 1};
            
            SelectKBest selector = new SelectKBest(2);
            selector.fit(X, y);
            
            int[] selected = selector.getSelectedFeatureIndices();
            // Should be sorted
            assertTrue(selected[0] < selected[1]);
        }
    }
    
    @Nested
    @DisplayName("ToString Tests")
    class ToStringTests {
        
        @Test
        @DisplayName("Should show unfitted state")
        void testToStringUnfitted() {
            SelectKBest selector = new SelectKBest(SelectKBest.ScoreFunction.CHI2, 3);
            String str = selector.toString();
            
            assertTrue(str.contains("CHI2"));
            assertTrue(str.contains("k=3"));
            assertTrue(str.contains("false"));
        }
        
        @Test
        @DisplayName("Should show fitted state")
        void testToStringFitted() {
            double[][] X = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {0.0, 2.1, 3.1}, {0.1, 5.1, 6.1}};
            int[] y = {0, 0, 1, 1};
            
            SelectKBest selector = new SelectKBest(2);
            selector.fit(X, y);
            
            String str = selector.toString();
            assertTrue(str.contains("n_features_in=3"));
            assertTrue(str.contains("n_features_out=2"));
        }
    }
    
    @Nested
    @DisplayName("Large Dataset Tests")
    class LargeDatasetTests {
        
        @Test
        @DisplayName("Should handle large number of features")
        void testManyFeatures() {
            int nSamples = 100;
            int nFeatures = 200;
            double[][] X = new double[nSamples][nFeatures];
            int[] y = new int[nSamples];
            
            java.util.Random random = new java.util.Random(42);
            for (int i = 0; i < nSamples; i++) {
                y[i] = i < nSamples / 2 ? 0 : 1;
                for (int j = 0; j < nFeatures; j++) {
                    // First 10 features are informative
                    if (j < 10) {
                        X[i][j] = y[i] * 5.0 + random.nextGaussian();
                    } else {
                        X[i][j] = random.nextGaussian();
                    }
                }
            }
            
            SelectKBest selector = new SelectKBest(10);
            double[][] result = selector.fitTransform(X, y);
            
            assertEquals(nSamples, result.length);
            assertEquals(10, result[0].length);
        }
    }
}
