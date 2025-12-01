package com.mindforge.regression;

import com.mindforge.validation.Metrics;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("LinearRegression Tests")
class LinearRegressionTest {
    
    @Nested
    @DisplayName("Basic Training and Prediction")
    class BasicTests {
        
        @Test
        @DisplayName("Should fit simple linear relationship y = 2x")
        void testSimpleLinearRelationship() {
            double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
            double[] y = {2.0, 4.0, 6.0, 8.0, 10.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            double prediction = lr.predict(new double[]{6.0});
            assertEquals(12.0, prediction, 1.0, "Should predict approximately 12 for input 6");
            
            double[] predictions = lr.predict(X);
            double rmse = Metrics.rmse(y, predictions);
            assertTrue(rmse < 0.5, "RMSE should be low on simple linear data");
        }
        
        @Test
        @DisplayName("Should achieve high R² score on linear data")
        void testR2Score() {
            double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
            double[] y = {2.1, 4.2, 5.9, 8.1, 10.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            double[] predictions = lr.predict(X);
            double r2 = Metrics.r2Score(y, predictions);
            assertTrue(r2 > 0.95, "R² score should be high for near-linear data");
        }
        
        @Test
        @DisplayName("Should handle multivariate regression")
        void testMultivariateRegression() {
            // y = 2*x1 + 3*x2 + 1
            double[][] X = {
                {1.0, 1.0}, {2.0, 1.0}, {1.0, 2.0}, {2.0, 2.0},
                {3.0, 1.0}, {1.0, 3.0}, {3.0, 3.0}
            };
            double[] y = {6.0, 8.0, 9.0, 11.0, 10.0, 12.0, 16.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            double[] predictions = lr.predict(X);
            double rmse = Metrics.rmse(y, predictions);
            assertTrue(rmse < 1.0, "RMSE should be low on multivariate linear data");
        }
        
        @Test
        @DisplayName("Should handle data with intercept")
        void testWithIntercept() {
            // y = x + 5
            double[][] X = {{0.0}, {1.0}, {2.0}, {3.0}, {4.0}};
            double[] y = {5.0, 6.0, 7.0, 8.0, 9.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            double prediction = lr.predict(new double[]{5.0});
            assertEquals(10.0, prediction, 0.5, "Should predict approximately 10 for input 5");
        }
    }
    
    @Nested
    @DisplayName("Model State Tests")
    class ModelStateTests {
        
        @Test
        @DisplayName("Should report unfitted state initially")
        void testIsFittedInitially() {
            LinearRegression lr = new LinearRegression();
            assertFalse(lr.isFitted(), "Model should not be fitted initially");
        }
        
        @Test
        @DisplayName("Should report fitted state after training")
        void testIsFittedAfterTraining() {
            LinearRegression lr = new LinearRegression();
            double[][] X = {{1.0}, {2.0}};
            double[] y = {1.0, 2.0};
            lr.train(X, y);
            
            assertTrue(lr.isFitted(), "Model should be fitted after training");
        }
        
        @Test
        @DisplayName("Should return learned weights")
        void testGetWeights() {
            double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
            double[] y = {2.0, 4.0, 6.0, 8.0, 10.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            double[] weights = lr.getWeights();
            assertNotNull(weights, "Weights should not be null");
            assertEquals(1, weights.length, "Should have one weight for one feature");
            assertTrue(Math.abs(weights[0] - 2.0) < 0.5, "Weight should be approximately 2");
        }
        
        @Test
        @DisplayName("Should return learned bias")
        void testGetBias() {
            // y = x + 5
            double[][] X = {{0.0}, {1.0}, {2.0}, {3.0}, {4.0}};
            double[] y = {5.0, 6.0, 7.0, 8.0, 9.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            double bias = lr.getBias();
            assertTrue(Math.abs(bias - 5.0) < 0.5, "Bias should be approximately 5");
        }
        
        @Test
        @DisplayName("Should return copy of weights, not reference")
        void testWeightsImmutability() {
            double[][] X = {{1.0}, {2.0}, {3.0}};
            double[] y = {2.0, 4.0, 6.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            double[] weights1 = lr.getWeights();
            double originalWeight = weights1[0];
            weights1[0] = 999.0;
            
            double[] weights2 = lr.getWeights();
            assertEquals(originalWeight, weights2[0], 0.001, "Modifying returned array should not affect model");
        }
    }
    
    @Nested
    @DisplayName("Exception Tests")
    class ExceptionTests {
        
        @Test
        @DisplayName("Should throw exception when predicting before training")
        void testPredictBeforeTraining() {
            LinearRegression lr = new LinearRegression();
            assertThrows(IllegalStateException.class, () -> {
                lr.predict(new double[]{1.0});
            }, "Should throw exception when predicting before training");
        }
        
        @Test
        @DisplayName("Should throw exception when getting weights before training")
        void testGetWeightsBeforeTraining() {
            LinearRegression lr = new LinearRegression();
            assertThrows(IllegalStateException.class, () -> {
                lr.getWeights();
            }, "Should throw exception when getting weights before training");
        }
        
        @Test
        @DisplayName("Should throw exception when getting bias before training")
        void testGetBiasBeforeTraining() {
            LinearRegression lr = new LinearRegression();
            assertThrows(IllegalStateException.class, () -> {
                lr.getBias();
            }, "Should throw exception when getting bias before training");
        }
        
        @Test
        @DisplayName("Should throw exception for mismatched data and target length")
        void testMismatchedDataTargetLength() {
            LinearRegression lr = new LinearRegression();
            double[][] X = {{1.0}, {2.0}, {3.0}};
            double[] y = {1.0, 2.0}; // Different length
            
            assertThrows(IllegalArgumentException.class, () -> {
                lr.train(X, y);
            }, "Should throw exception for mismatched lengths");
        }
        
        @Test
        @DisplayName("Should throw exception for empty training data")
        void testEmptyTrainingData() {
            LinearRegression lr = new LinearRegression();
            double[][] X = {};
            double[] y = {};
            
            assertThrows(IllegalArgumentException.class, () -> {
                lr.train(X, y);
            }, "Should throw exception for empty data");
        }
        
        @Test
        @DisplayName("Should throw exception for input dimension mismatch during prediction")
        void testInputDimensionMismatch() {
            double[][] X = {{1.0, 2.0}, {2.0, 3.0}};
            double[] y = {1.0, 2.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            assertThrows(IllegalArgumentException.class, () -> {
                lr.predict(new double[]{1.0}); // Should be 2 features, not 1
            }, "Should throw exception for wrong input dimension");
        }
    }
    
    @Nested
    @DisplayName("Batch Prediction Tests")
    class BatchPredictionTests {
        
        @Test
        @DisplayName("Should predict multiple samples at once")
        void testBatchPrediction() {
            double[][] X_train = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
            double[] y_train = {2.0, 4.0, 6.0, 8.0, 10.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X_train, y_train);
            
            double[][] X_test = {{1.5}, {2.5}, {3.5}};
            double[] predictions = lr.predict(X_test);
            
            assertEquals(3, predictions.length, "Should return prediction for each sample");
            assertTrue(Math.abs(predictions[0] - 3.0) < 0.5, "First prediction should be ~3");
            assertTrue(Math.abs(predictions[1] - 5.0) < 0.5, "Second prediction should be ~5");
            assertTrue(Math.abs(predictions[2] - 7.0) < 0.5, "Third prediction should be ~7");
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle minimal training data (2 samples)")
        void testMinimalData() {
            double[][] X = {{0.0}, {1.0}};
            double[] y = {0.0, 1.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            assertTrue(lr.isFitted(), "Should fit with minimal data");
            double prediction = lr.predict(new double[]{0.5});
            assertTrue(Math.abs(prediction - 0.5) < 0.3, "Should predict approximately 0.5");
        }
        
        @Test
        @DisplayName("Should handle negative values")
        void testNegativeValues() {
            double[][] X = {{-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}};
            double[] y = {-4.0, -2.0, 0.0, 2.0, 4.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            double prediction = lr.predict(new double[]{-3.0});
            assertEquals(-6.0, prediction, 1.0, "Should handle negative values correctly");
        }
        
        @Test
        @DisplayName("Should handle zero values")
        void testZeroValues() {
            double[][] X = {{0.0}, {1.0}, {2.0}};
            double[] y = {0.0, 0.0, 0.0}; // All zeros
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            double prediction = lr.predict(new double[]{5.0});
            assertEquals(0.0, prediction, 0.1, "Should predict near zero for constant zero target");
        }
        
        @Test
        @DisplayName("Should handle large values")
        void testLargeValues() {
            // Scale values to avoid gradient descent issues
            double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
            double[] y = {10.0, 20.0, 30.0, 40.0, 50.0};
            
            LinearRegression lr = new LinearRegression();
            lr.train(X, y);
            
            double[] predictions = lr.predict(X);
            double rmse = Metrics.rmse(y, predictions);
            assertTrue(rmse < 5.0, "Should handle values with reasonable error");
        }
    }
}
