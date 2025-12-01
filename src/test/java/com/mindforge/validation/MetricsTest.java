package com.mindforge.validation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Metrics Tests")
class MetricsTest {
    
    @Nested
    @DisplayName("Accuracy Tests")
    class AccuracyTests {
        
        @Test
        @DisplayName("Should calculate accuracy correctly")
        void testAccuracy() {
            int[] yTrue = {0, 1, 0, 1, 0, 1};
            int[] yPred = {0, 1, 0, 1, 1, 0};
            
            double accuracy = Metrics.accuracy(yTrue, yPred);
            assertEquals(0.6667, accuracy, 0.01, "Accuracy should be 4/6");
        }
        
        @Test
        @DisplayName("Should return 1.0 for perfect predictions")
        void testPerfectAccuracy() {
            int[] yTrue = {0, 1, 2, 1, 0};
            int[] yPred = {0, 1, 2, 1, 0};
            
            double accuracy = Metrics.accuracy(yTrue, yPred);
            assertEquals(1.0, accuracy, 0.001, "Perfect predictions should have accuracy 1.0");
        }
        
        @Test
        @DisplayName("Should return 0.0 for all wrong predictions")
        void testZeroAccuracy() {
            int[] yTrue = {0, 0, 0};
            int[] yPred = {1, 1, 1};
            
            double accuracy = Metrics.accuracy(yTrue, yPred);
            assertEquals(0.0, accuracy, 0.001, "All wrong predictions should have accuracy 0.0");
        }
        
        @Test
        @DisplayName("Should throw exception for mismatched lengths")
        void testAccuracyMismatchedLengths() {
            int[] yTrue = {0, 1, 0};
            int[] yPred = {0, 1};
            
            assertThrows(IllegalArgumentException.class, () -> {
                Metrics.accuracy(yTrue, yPred);
            }, "Should throw exception for mismatched lengths");
        }
    }
    
    @Nested
    @DisplayName("MSE Tests")
    class MSETests {
        
        @Test
        @DisplayName("Should return 0.0 for perfect predictions")
        void testMSEPerfect() {
            double[] yTrue = {1.0, 2.0, 3.0, 4.0};
            double[] yPred = {1.0, 2.0, 3.0, 4.0};
            
            double mse = Metrics.mse(yTrue, yPred);
            assertEquals(0.0, mse, 0.001, "Perfect predictions should have MSE 0.0");
        }
        
        @Test
        @DisplayName("Should calculate MSE correctly")
        void testMSE() {
            double[] yTrue = {1.0, 2.0, 3.0, 4.0};
            double[] yPred2 = {2.0, 3.0, 4.0, 5.0};
            double mse = Metrics.mse(yTrue, yPred2);
            assertEquals(1.0, mse, 0.001, "Constant error of 1 should give MSE 1.0");
        }
        
        @Test
        @DisplayName("Should throw exception for mismatched lengths")
        void testMSEMismatchedLengths() {
            double[] yTrue = {1.0, 2.0};
            double[] yPred = {1.0};
            
            assertThrows(IllegalArgumentException.class, () -> {
                Metrics.mse(yTrue, yPred);
            }, "Should throw exception for mismatched lengths");
        }
    }
    
    @Nested
    @DisplayName("RMSE Tests")
    class RMSETests {
        
        @Test
        @DisplayName("Should calculate RMSE correctly")
        void testRMSE() {
            double[] yTrue = {1.0, 2.0, 3.0, 4.0};
            double[] yPred = {2.0, 3.0, 4.0, 5.0};
            
            double rmse = Metrics.rmse(yTrue, yPred);
            assertEquals(1.0, rmse, 0.001, "RMSE should be square root of MSE");
        }
        
        @Test
        @DisplayName("Should return 0.0 for perfect predictions")
        void testRMSEPerfect() {
            double[] yTrue = {1.0, 2.0, 3.0};
            double[] yPred = {1.0, 2.0, 3.0};
            
            double rmse = Metrics.rmse(yTrue, yPred);
            assertEquals(0.0, rmse, 0.001, "Perfect predictions should have RMSE 0.0");
        }
    }
    
    @Nested
    @DisplayName("MAE Tests")
    class MAETests {
        
        @Test
        @DisplayName("Should calculate MAE correctly")
        void testMAE() {
            double[] yTrue = {1.0, 2.0, 3.0, 4.0};
            double[] yPred = {2.0, 3.0, 4.0, 5.0};
            
            double mae = Metrics.mae(yTrue, yPred);
            assertEquals(1.0, mae, 0.001, "Constant error of 1 should give MAE 1.0");
        }
        
        @Test
        @DisplayName("Should handle mixed positive and negative errors")
        void testMAEMixedErrors() {
            double[] yTrue = {1.0, 2.0, 3.0, 4.0};
            double[] yPred = {2.0, 1.0, 4.0, 3.0};
            
            double mae = Metrics.mae(yTrue, yPred);
            assertEquals(1.0, mae, 0.001, "MAE should be average of absolute errors");
        }
        
        @Test
        @DisplayName("Should throw exception for mismatched lengths")
        void testMAEMismatchedLengths() {
            double[] yTrue = {1.0, 2.0, 3.0};
            double[] yPred = {1.0, 2.0};
            
            assertThrows(IllegalArgumentException.class, () -> {
                Metrics.mae(yTrue, yPred);
            }, "Should throw exception for mismatched lengths");
        }
    }
    
    @Nested
    @DisplayName("R2 Score Tests")
    class R2ScoreTests {
        
        @Test
        @DisplayName("Should return 1.0 for perfect predictions")
        void testR2ScorePerfect() {
            double[] yTrue = {1.0, 2.0, 3.0, 4.0, 5.0};
            double[] yPred = {1.0, 2.0, 3.0, 4.0, 5.0};
            
            double r2 = Metrics.r2Score(yTrue, yPred);
            assertEquals(1.0, r2, 0.001, "Perfect predictions should have R² = 1.0");
        }
        
        @Test
        @DisplayName("Should return 0.0 for predicting the mean")
        void testR2ScorePredictMean() {
            double[] yTrue = {1.0, 2.0, 3.0, 4.0, 5.0};
            double[] yPred = {3.0, 3.0, 3.0, 3.0, 3.0}; // Mean of yTrue
            
            double r2 = Metrics.r2Score(yTrue, yPred);
            assertEquals(0.0, r2, 0.001, "Predicting mean should give R² = 0.0");
        }
        
        @Test
        @DisplayName("Should return negative for worse than mean predictions")
        void testR2ScoreNegative() {
            double[] yTrue = {1.0, 2.0, 3.0, 4.0, 5.0};
            double[] yPred = {5.0, 4.0, 3.0, 2.0, 1.0}; // Reversed
            
            double r2 = Metrics.r2Score(yTrue, yPred);
            assertTrue(r2 < 0, "Worse than mean predictions should give negative R²");
        }
        
        @Test
        @DisplayName("Should return 1.0 when all true values are the same")
        void testR2ScoreConstantTrue() {
            double[] yTrue = {3.0, 3.0, 3.0, 3.0};
            double[] yPred = {3.0, 3.0, 3.0, 3.0};
            
            double r2 = Metrics.r2Score(yTrue, yPred);
            assertEquals(1.0, r2, 0.001, "Constant true values with perfect prediction should give R² = 1.0");
        }
        
        @Test
        @DisplayName("Should throw exception for mismatched lengths")
        void testR2ScoreMismatchedLengths() {
            double[] yTrue = {1.0, 2.0};
            double[] yPred = {1.0};
            
            assertThrows(IllegalArgumentException.class, () -> {
                Metrics.r2Score(yTrue, yPred);
            }, "Should throw exception for mismatched lengths");
        }
    }
    
    @Nested
    @DisplayName("Precision Tests")
    class PrecisionTests {
        
        @Test
        @DisplayName("Should calculate precision correctly")
        void testPrecision() {
            int[] yTrue = {1, 1, 0, 1, 0, 0};
            int[] yPred = {1, 0, 0, 1, 0, 1};
            
            double precision = Metrics.precision(yTrue, yPred, 1);
            assertEquals(0.6667, precision, 0.01, "Precision should be 2/3");
        }
        
        @Test
        @DisplayName("Should return 1.0 for perfect precision")
        void testPerfectPrecision() {
            int[] yTrue = {1, 1, 0, 0};
            int[] yPred = {1, 1, 0, 0};
            
            double precision = Metrics.precision(yTrue, yPred, 1);
            assertEquals(1.0, precision, 0.001, "Perfect predictions should have precision 1.0");
        }
        
        @Test
        @DisplayName("Should return 0.0 when no positive predictions")
        void testPrecisionNoPositivePredictions() {
            int[] yTrue = {1, 1, 1};
            int[] yPred = {0, 0, 0};
            
            double precision = Metrics.precision(yTrue, yPred, 1);
            assertEquals(0.0, precision, 0.001, "No positive predictions should give precision 0.0");
        }
        
        @Test
        @DisplayName("Should throw exception for mismatched lengths")
        void testPrecisionMismatchedLengths() {
            int[] yTrue = {1, 0};
            int[] yPred = {1};
            
            assertThrows(IllegalArgumentException.class, () -> {
                Metrics.precision(yTrue, yPred, 1);
            }, "Should throw exception for mismatched lengths");
        }
    }
    
    @Nested
    @DisplayName("Recall Tests")
    class RecallTests {
        
        @Test
        @DisplayName("Should calculate recall correctly")
        void testRecall() {
            int[] yTrue = {1, 1, 0, 1, 0, 0};
            int[] yPred = {1, 0, 0, 1, 0, 1};
            
            double recall = Metrics.recall(yTrue, yPred, 1);
            assertEquals(0.6667, recall, 0.01, "Recall should be 2/3");
        }
        
        @Test
        @DisplayName("Should return 1.0 for perfect recall")
        void testPerfectRecall() {
            int[] yTrue = {1, 1, 0, 0};
            int[] yPred = {1, 1, 1, 1};
            
            double recall = Metrics.recall(yTrue, yPred, 1);
            assertEquals(1.0, recall, 0.001, "All positives predicted should have recall 1.0");
        }
        
        @Test
        @DisplayName("Should return 0.0 when no actual positives")
        void testRecallNoActualPositives() {
            int[] yTrue = {0, 0, 0};
            int[] yPred = {1, 1, 1};
            
            double recall = Metrics.recall(yTrue, yPred, 1);
            assertEquals(0.0, recall, 0.001, "No actual positives should give recall 0.0");
        }
        
        @Test
        @DisplayName("Should throw exception for mismatched lengths")
        void testRecallMismatchedLengths() {
            int[] yTrue = {1, 0, 1};
            int[] yPred = {1, 0};
            
            assertThrows(IllegalArgumentException.class, () -> {
                Metrics.recall(yTrue, yPred, 1);
            }, "Should throw exception for mismatched lengths");
        }
    }
    
    @Nested
    @DisplayName("F1 Score Tests")
    class F1ScoreTests {
        
        @Test
        @DisplayName("Should return 1.0 for perfect F1 score")
        void testF1ScorePerfect() {
            int[] yTrue = {1, 1, 0, 1, 0, 0};
            int[] yPred = {1, 1, 0, 1, 0, 0};
            
            double f1 = Metrics.f1Score(yTrue, yPred, 1);
            assertEquals(1.0, f1, 0.001, "Perfect predictions should have F1 = 1.0");
        }
        
        @Test
        @DisplayName("Should return 0.0 when precision and recall are both 0")
        void testF1ScoreZero() {
            int[] yTrue = {0, 0, 0};
            int[] yPred = {0, 0, 0};
            
            double f1 = Metrics.f1Score(yTrue, yPred, 1);
            assertEquals(0.0, f1, 0.001, "No positives should give F1 = 0.0");
        }
        
        @Test
        @DisplayName("Should calculate F1 as harmonic mean of precision and recall")
        void testF1ScoreCalculation() {
            int[] yTrue = {1, 1, 0, 0, 1, 0};
            int[] yPred = {1, 0, 1, 0, 1, 0};
            
            double precision = Metrics.precision(yTrue, yPred, 1);
            double recall = Metrics.recall(yTrue, yPred, 1);
            double expectedF1 = 2 * precision * recall / (precision + recall);
            
            double f1 = Metrics.f1Score(yTrue, yPred, 1);
            assertEquals(expectedF1, f1, 0.001, "F1 should be harmonic mean of precision and recall");
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle single element arrays")
        void testSingleElement() {
            int[] yTrue = {1};
            int[] yPred = {1};
            
            assertEquals(1.0, Metrics.accuracy(yTrue, yPred), 0.001);
            assertEquals(1.0, Metrics.precision(yTrue, yPred, 1), 0.001);
            assertEquals(1.0, Metrics.recall(yTrue, yPred, 1), 0.001);
        }
        
        @Test
        @DisplayName("Should handle multi-class for accuracy")
        void testMultiClass() {
            int[] yTrue = {0, 1, 2, 3, 4};
            int[] yPred = {0, 1, 2, 3, 4};
            
            assertEquals(1.0, Metrics.accuracy(yTrue, yPred), 0.001);
        }
    }
}
