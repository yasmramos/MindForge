package com.mindforge.validation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("ROC Curve Tests")
class ROCCurveTest {
    
    @Nested
    @DisplayName("AUC Calculation Tests")
    class AUCCalculationTests {
        
        @Test
        @DisplayName("Perfect classifier should have AUC close to 1.0")
        void testPerfectAUC() {
            int[] yTrue = {0, 0, 0, 1, 1, 1};
            double[] yScores = {0.1, 0.2, 0.3, 0.7, 0.8, 0.9};
            
            ROCCurve roc = new ROCCurve(yTrue, yScores);
            double auc = roc.getAUC();
            
            assertEquals(1.0, auc, 0.01, "Perfect classifier should have AUC close to 1.0");
        }
        
        @Test
        @DisplayName("Worst classifier should have AUC close to 0.0")
        void testWorstAUC() {
            int[] yTrue = {0, 0, 0, 1, 1, 1};
            double[] yScores = {0.9, 0.8, 0.7, 0.1, 0.2, 0.3}; // Reversed scores
            
            ROCCurve roc = new ROCCurve(yTrue, yScores);
            double auc = roc.getAUC();
            
            assertEquals(0.0, auc, 0.01, "Worst classifier should have AUC close to 0.0");
        }
        
        @Test
        @DisplayName("AUC should be between 0 and 1")
        void testAUCRange() {
            int[] yTrue = {0, 0, 1, 1, 0, 1, 0, 1};
            double[] yScores = {0.2, 0.4, 0.6, 0.8, 0.3, 0.7, 0.1, 0.9};
            
            ROCCurve roc = new ROCCurve(yTrue, yScores);
            double auc = roc.getAUC();
            
            assertTrue(auc >= 0 && auc <= 1, "AUC should be between 0 and 1");
        }
        
        @Test
        @DisplayName("Static rocAucScore should work")
        void testStaticRocAucScore() {
            int[] yTrue = {0, 0, 0, 1, 1, 1};
            double[] yScores = {0.1, 0.2, 0.3, 0.7, 0.8, 0.9};
            
            double auc = ROCCurve.rocAucScore(yTrue, yScores);
            
            assertEquals(1.0, auc, 0.01, "Perfect classifier should have AUC close to 1.0");
        }
    }
    
    @Nested
    @DisplayName("ROC Curve Points Tests")
    class ROCCurvePointsTests {
        
        @Test
        @DisplayName("Should calculate TPR and FPR arrays")
        void testTPRFPRArrays() {
            int[] yTrue = {0, 0, 1, 1};
            double[] yScores = {0.1, 0.4, 0.6, 0.9};
            
            ROCCurve roc = new ROCCurve(yTrue, yScores);
            double[] tpr = roc.getTPR();
            double[] fpr = roc.getFPR();
            
            assertNotNull(tpr);
            assertNotNull(fpr);
            assertEquals(tpr.length, fpr.length, "TPR and FPR arrays should have same length");
        }
        
        @Test
        @DisplayName("TPR and FPR should be between 0 and 1")
        void testTPRFPRRange() {
            int[] yTrue = {0, 0, 1, 1, 0, 1};
            double[] yScores = {0.1, 0.2, 0.7, 0.8, 0.3, 0.9};
            
            ROCCurve roc = new ROCCurve(yTrue, yScores);
            double[] tpr = roc.getTPR();
            double[] fpr = roc.getFPR();
            
            for (double val : tpr) {
                assertTrue(val >= 0 && val <= 1, "TPR should be between 0 and 1");
            }
            for (double val : fpr) {
                assertTrue(val >= 0 && val <= 1, "FPR should be between 0 and 1");
            }
        }
        
        @Test
        @DisplayName("Should return thresholds")
        void testThresholds() {
            int[] yTrue = {0, 0, 1, 1};
            double[] yScores = {0.1, 0.4, 0.6, 0.9};
            
            ROCCurve roc = new ROCCurve(yTrue, yScores);
            double[] thresholds = roc.getThresholds();
            
            assertNotNull(thresholds);
            assertTrue(thresholds.length > 0, "Should have at least one threshold");
        }
        
        @Test
        @DisplayName("ROC curve should start at (0,0) and end at (1,1)")
        void testROCCurveEndpoints() {
            int[] yTrue = {0, 0, 1, 1};
            double[] yScores = {0.1, 0.4, 0.6, 0.9};
            
            ROCCurve roc = new ROCCurve(yTrue, yScores);
            double[] tpr = roc.getTPR();
            double[] fpr = roc.getFPR();
            
            // First point should be (0, 0)
            assertEquals(0.0, fpr[0], 0.001);
            assertEquals(0.0, tpr[0], 0.001);
            
            // Last point should be (1, 1)
            assertEquals(1.0, fpr[fpr.length - 1], 0.001);
            assertEquals(1.0, tpr[tpr.length - 1], 0.001);
        }
    }
    
    @Nested
    @DisplayName("Optimal Threshold Tests")
    class OptimalThresholdTests {
        
        @Test
        @DisplayName("Should find optimal threshold")
        void testOptimalThreshold() {
            int[] yTrue = {0, 0, 0, 1, 1, 1};
            double[] yScores = {0.1, 0.2, 0.3, 0.7, 0.8, 0.9};
            
            ROCCurve roc = new ROCCurve(yTrue, yScores);
            double optimalThreshold = roc.getOptimalThreshold();
            
            assertNotNull(optimalThreshold);
            // For a perfect classifier, optimal threshold should be between the classes
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should throw exception for mismatched lengths")
        void testMismatchedLengths() {
            int[] yTrue = {0, 1, 0};
            double[] yScores = {0.1, 0.9};
            
            assertThrows(IllegalArgumentException.class, () -> {
                new ROCCurve(yTrue, yScores);
            });
        }
        
        @Test
        @DisplayName("Should handle tied scores")
        void testTiedScores() {
            int[] yTrue = {0, 0, 1, 1};
            double[] yScores = {0.5, 0.5, 0.5, 0.5};
            
            ROCCurve roc = new ROCCurve(yTrue, yScores);
            double auc = roc.getAUC();
            
            assertTrue(auc >= 0 && auc <= 1, "AUC should be valid even with tied scores");
        }
        
        @Test
        @DisplayName("Should generate string representation")
        void testToString() {
            int[] yTrue = {0, 0, 1, 1};
            double[] yScores = {0.1, 0.4, 0.6, 0.9};
            
            ROCCurve roc = new ROCCurve(yTrue, yScores);
            String str = roc.toString();
            
            assertNotNull(str);
            assertTrue(str.contains("AUC"));
        }
    }
}
