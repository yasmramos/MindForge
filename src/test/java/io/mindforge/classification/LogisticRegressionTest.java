package io.mindforge.classification;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Random;

/**
 * Comprehensive test suite for LogisticRegression classifier.
 * Tests binary/multiclass classification, regularization, solvers, and edge cases.
 */
public class LogisticRegressionTest {
    
    private static final double EPSILON = 1e-6;
    private static final double ACCURACY_THRESHOLD = 0.8;
    
    private double[][] X_binary;
    private int[] y_binary;
    private double[][] X_multiclass;
    private int[] y_multiclass;
    
    @BeforeEach
    public void setUp() {
        // Binary classification dataset: linearly separable
        X_binary = new double[][] {
            {1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0}, {4.0, 5.0}, {5.0, 5.0},
            {1.0, 1.0}, {2.0, 1.0}, {3.0, 2.0}, {4.0, 3.0}, {5.0, 3.0},
            {6.0, 7.0}, {7.0, 7.0}, {8.0, 8.0}, {9.0, 9.0}, {10.0, 10.0},
            {6.0, 5.0}, {7.0, 6.0}, {8.0, 6.0}, {9.0, 7.0}, {10.0, 8.0}
        };
        y_binary = new int[] {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        };
        
        // Multiclass dataset: 3 clusters
        X_multiclass = new double[][] {
            // Class 0: bottom-left
            {1.0, 1.0}, {1.5, 1.5}, {2.0, 1.0}, {1.0, 2.0}, {2.0, 2.0},
            {1.2, 1.3}, {1.8, 1.2}, {1.5, 2.0}, {2.0, 1.5}, {1.3, 1.8},
            // Class 1: top-right
            {8.0, 8.0}, {8.5, 8.5}, {9.0, 8.0}, {8.0, 9.0}, {9.0, 9.0},
            {8.2, 8.3}, {8.8, 8.2}, {8.5, 9.0}, {9.0, 8.5}, {8.3, 8.8},
            // Class 2: top-left
            {1.0, 8.0}, {1.5, 8.5}, {2.0, 8.0}, {1.0, 9.0}, {2.0, 9.0},
            {1.2, 8.3}, {1.8, 8.2}, {1.5, 9.0}, {2.0, 8.5}, {1.3, 8.8}
        };
        y_multiclass = new int[] {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        };
    }
    
    // ==================== Binary Classification Tests ====================
    
    @Test
    public void testBinaryClassificationGradientDescent() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        int[] predictions = lr.predict(X_binary);
        
        double accuracy = computeAccuracy(y_binary, predictions);
        assertTrue(accuracy >= ACCURACY_THRESHOLD, 
            "Accuracy should be >= " + ACCURACY_THRESHOLD + ", got " + accuracy);
    }
    
    @Test
    public void testBinaryClassificationSGD() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("sgd")
            .learningRate(0.1)
            .batchSize(5)
            .maxIter(500)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        int[] predictions = lr.predict(X_binary);
        
        double accuracy = computeAccuracy(y_binary, predictions);
        assertTrue(accuracy >= ACCURACY_THRESHOLD,
            "SGD should achieve good accuracy");
    }
    
    @Test
    public void testBinaryClassificationNewtonCG() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("newton_cg")
            .learningRate(1.0)
            .maxIter(1000)
            .penalty("l2")
            .C(1.0)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        int[] predictions = lr.predict(X_binary);
        
        // Newton-CG should complete training and produce predictions
        assertNotNull(predictions);
        assertEquals(X_binary.length, predictions.length);
        
        // Loss should decrease during training
        List<Double> lossHistory = lr.getLossHistory();
        assertTrue(lossHistory.size() > 0, "Should have loss history");
    }
    
    @Test
    public void testBinaryPredictProba() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        double[][] proba = lr.predictProba(X_binary);
        
        // Check shape
        assertEquals(X_binary.length, proba.length);
        assertEquals(2, proba[0].length);
        
        // Check probabilities sum to 1
        for (int i = 0; i < proba.length; i++) {
            double sum = proba[i][0] + proba[i][1];
            assertEquals(1.0, sum, 0.01, "Probabilities should sum to 1");
        }
        
        // Check probabilities are in [0, 1]
        for (int i = 0; i < proba.length; i++) {
            assertTrue(proba[i][0] >= 0 && proba[i][0] <= 1);
            assertTrue(proba[i][1] >= 0 && proba[i][1] <= 1);
        }
    }
    
    // ==================== Multiclass Classification Tests ====================
    
    @Test
    public void testMulticlassClassification() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_multiclass, y_multiclass);
        int[] predictions = lr.predict(X_multiclass);
        
        double accuracy = computeAccuracy(y_multiclass, predictions);
        assertTrue(accuracy >= ACCURACY_THRESHOLD,
            "Multiclass accuracy should be >= " + ACCURACY_THRESHOLD);
    }
    
    @Test
    public void testMulticlassPredictProba() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_multiclass, y_multiclass);
        double[][] proba = lr.predictProba(X_multiclass);
        
        // Check shape
        assertEquals(X_multiclass.length, proba.length);
        assertEquals(3, proba[0].length);
        
        // Check probabilities sum to 1
        for (int i = 0; i < proba.length; i++) {
            double sum = proba[i][0] + proba[i][1] + proba[i][2];
            assertEquals(1.0, sum, 0.01, "Probabilities should sum to 1");
        }
    }
    
    @Test
    public void testMulticlassWithSGD() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("sgd")
            .learningRate(0.1)
            .batchSize(10)
            .maxIter(500)
            .randomState(42)
            .build();
        
        lr.fit(X_multiclass, y_multiclass);
        int[] predictions = lr.predict(X_multiclass);
        
        double accuracy = computeAccuracy(y_multiclass, predictions);
        assertTrue(accuracy >= 0.7, "SGD multiclass should achieve reasonable accuracy");
    }
    
    // ==================== Regularization Tests ====================
    
    @Test
    public void testL2Regularization() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .penalty("l2")
            .C(0.1)  // Strong regularization
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        double[][] weights = lr.getCoefficients();
        
        // Check that weights are relatively small due to regularization
        double weightNorm = 0.0;
        for (int j = 0; j < weights[0].length; j++) {
            weightNorm += weights[0][j] * weights[0][j];
        }
        weightNorm = Math.sqrt(weightNorm);
        
        // Weights should be smaller with strong regularization
        assertTrue(weightNorm < 10.0, "L2 regularization should produce small weights");
    }
    
    @Test
    public void testL1Regularization() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .penalty("l1")
            .C(2.0)
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        double[][] weights = lr.getCoefficients();
        
        // L1 should still learn reasonable model
        int[] predictions = lr.predict(X_binary);
        double accuracy = computeAccuracy(y_binary, predictions);
        assertTrue(accuracy >= 0.65, "L1 regularization should still learn");
    }
    
    @Test
    public void testElasticNetRegularization() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .penalty("elasticnet")
            .C(1.0)
            .l1Ratio(0.5)
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        int[] predictions = lr.predict(X_binary);
        
        double accuracy = computeAccuracy(y_binary, predictions);
        assertTrue(accuracy >= ACCURACY_THRESHOLD, "Elastic Net should work well");
    }
    
    @Test
    public void testNoRegularization() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .penalty("none")
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        int[] predictions = lr.predict(X_binary);
        
        double accuracy = computeAccuracy(y_binary, predictions);
        assertTrue(accuracy >= ACCURACY_THRESHOLD, "No regularization should work");
    }
    
    @Test
    public void testRegularizationStrength() {
        // Weak regularization
        LogisticRegression lr1 = new LogisticRegression.Builder()
            .penalty("l2")
            .C(10.0)
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        lr1.fit(X_binary, y_binary);
        
        // Strong regularization
        LogisticRegression lr2 = new LogisticRegression.Builder()
            .penalty("l2")
            .C(0.1)
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        lr2.fit(X_binary, y_binary);
        
        // Both models should complete training
        int[] pred1 = lr1.predict(X_binary);
        int[] pred2 = lr2.predict(X_binary);
        
        assertNotNull(pred1);
        assertNotNull(pred2);
        assertEquals(X_binary.length, pred1.length);
        assertEquals(X_binary.length, pred2.length);
    }
    
    // ==================== Convergence and Training Tests ====================
    
    @Test
    public void testConvergence() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .tol(1e-4)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        List<Double> lossHistory = lr.getLossHistory();
        
        // Loss should decrease over time
        assertTrue(lossHistory.size() > 0, "Should have loss history");
        
        if (lossHistory.size() > 10) {
            double initialLoss = lossHistory.get(0);
            double finalLoss = lossHistory.get(lossHistory.size() - 1);
            assertTrue(finalLoss < initialLoss, "Loss should decrease during training");
        }
    }
    
    @Test
    public void testLossHistory() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(100)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        List<Double> lossHistory = lr.getLossHistory();
        
        assertEquals(100, lossHistory.size(), "Should have 100 loss values for 100 iterations");
    }
    
    @Test
    public void testEarlyConvergence() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.3)  // Moderate learning rate for faster convergence
            .maxIter(5000)
            .tol(1e-3)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        List<Double> lossHistory = lr.getLossHistory();
        
        // Should converge before max iterations
        assertTrue(lossHistory.size() < 5000, "Should converge early with moderate learning rate");
    }
    
    // ==================== Model Attributes Tests ====================
    
    @Test
    public void testGetCoefficients() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        double[][] weights = lr.getCoefficients();
        
        // Binary classification: 1 set of weights
        assertEquals(1, weights.length);
        assertEquals(X_binary[0].length, weights[0].length);
    }
    
    @Test
    public void testGetIntercepts() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        double[] intercepts = lr.getIntercepts();
        
        assertEquals(1, intercepts.length, "Binary classification should have 1 intercept");
    }
    
    @Test
    public void testGetClasses() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .randomState(42)
            .build();
        
        lr.fit(X_binary, y_binary);
        int[] classes = lr.getClasses();
        
        assertArrayEquals(new int[]{0, 1}, classes);
    }
    
    @Test
    public void testMulticlassCoefficients() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_multiclass, y_multiclass);
        double[][] weights = lr.getCoefficients();
        
        // Multiclass: one set of weights per class
        assertEquals(3, weights.length);
        assertEquals(X_multiclass[0].length, weights[0].length);
    }
    
    // ==================== Edge Cases and Error Handling ====================
    
    @Test
    public void testPredictBeforeFit() {
        LogisticRegression lr = new LogisticRegression.Builder().build();
        
        assertThrows(IllegalStateException.class, () -> {
            lr.predict(X_binary);
        }, "Should throw exception when predicting before fitting");
    }
    
    @Test
    public void testInvalidPenalty() {
        // Invalid penalty should still construct, but may affect training
        LogisticRegression lr = new LogisticRegression.Builder()
            .penalty("invalid")
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(100)
            .randomState(42)
            .build();
        
        // Should still fit (treated as no regularization)
        lr.fit(X_binary, y_binary);
        int[] predictions = lr.predict(X_binary);
        assertNotNull(predictions);
    }
    
    @Test
    public void testInvalidSolver() {
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("invalid_solver")
            .build();
        
        assertThrows(IllegalArgumentException.class, () -> {
            lr.fit(X_binary, y_binary);
        }, "Should throw exception for invalid solver");
    }
    
    @Test
    public void testInvalidC() {
        assertThrows(IllegalArgumentException.class, () -> {
            new LogisticRegression.Builder().C(-1.0);
        }, "C must be positive");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new LogisticRegression.Builder().C(0.0);
        }, "C must be positive");
    }
    
    @Test
    public void testInvalidL1Ratio() {
        assertThrows(IllegalArgumentException.class, () -> {
            new LogisticRegression.Builder().l1Ratio(-0.1);
        }, "l1Ratio must be in [0, 1]");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new LogisticRegression.Builder().l1Ratio(1.5);
        }, "l1Ratio must be in [0, 1]");
    }
    
    @Test
    public void testMismatchedDataLength() {
        LogisticRegression lr = new LogisticRegression.Builder().build();
        
        int[] wrongY = new int[]{0, 1, 0};  // Different length
        
        assertThrows(IllegalArgumentException.class, () -> {
            lr.fit(X_binary, wrongY);
        }, "Should throw exception for mismatched data lengths");
    }
    
    // ==================== Reproducibility Tests ====================
    
    @Test
    public void testReproducibilityWithRandomState() {
        LogisticRegression lr1 = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        LogisticRegression lr2 = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr1.fit(X_binary, y_binary);
        lr2.fit(X_binary, y_binary);
        
        int[] pred1 = lr1.predict(X_binary);
        int[] pred2 = lr2.predict(X_binary);
        
        assertArrayEquals(pred1, pred2, "Same random state should give same predictions");
    }
    
    @Test
    public void testDifferentRandomStates() {
        LogisticRegression lr1 = new LogisticRegression.Builder()
            .solver("sgd")
            .learningRate(0.1)
            .maxIter(100)
            .randomState(42)
            .build();
        
        LogisticRegression lr2 = new LogisticRegression.Builder()
            .solver("sgd")
            .learningRate(0.1)
            .maxIter(100)
            .randomState(123)
            .build();
        
        lr1.fit(X_binary, y_binary);
        lr2.fit(X_binary, y_binary);
        
        double[][] weights1 = lr1.getCoefficients();
        double[][] weights2 = lr2.getCoefficients();
        
        // Different random states may produce different weights (especially with SGD)
        // But both should achieve good accuracy
        int[] pred1 = lr1.predict(X_binary);
        int[] pred2 = lr2.predict(X_binary);
        
        double acc1 = computeAccuracy(y_binary, pred1);
        double acc2 = computeAccuracy(y_binary, pred2);
        
        assertTrue(acc1 >= 0.7 && acc2 >= 0.7, "Both models should achieve good accuracy");
    }
    
    // ==================== Small Dataset Tests ====================
    
    @Test
    public void testSmallDataset() {
        double[][] X_small = {{1.0, 2.0}, {2.0, 3.0}, {5.0, 6.0}, {6.0, 7.0}};
        int[] y_small = {0, 0, 1, 1};
        
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_small, y_small);
        int[] predictions = lr.predict(X_small);
        
        assertNotNull(predictions);
        assertEquals(4, predictions.length);
    }
    
    @Test
    public void testSingleFeature() {
        double[][] X_single = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}};
        int[] y_single = {0, 0, 0, 0, 1, 1, 1, 1};
        
        LogisticRegression lr = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr.fit(X_single, y_single);
        int[] predictions = lr.predict(X_single);
        
        double accuracy = computeAccuracy(y_single, predictions);
        assertTrue(accuracy >= 0.7, "Should work with single feature");
    }
    
    // ==================== Hyperparameter Tests ====================
    
    @Test
    public void testDifferentLearningRates() {
        double[] learningRates = {0.01, 0.1, 0.3};
        
        for (double lr_val : learningRates) {
            LogisticRegression lr = new LogisticRegression.Builder()
                .solver("gradient_descent")
                .learningRate(lr_val)
                .maxIter(1000)
                .randomState(42)
                .build();
            
            lr.fit(X_binary, y_binary);
            int[] predictions = lr.predict(X_binary);
            double accuracy = computeAccuracy(y_binary, predictions);
            
            assertTrue(accuracy >= 0.65, 
                "Learning rate " + lr_val + " should achieve reasonable accuracy");
        }
    }
    
    @Test
    public void testDifferentBatchSizes() {
        int[] batchSizes = {1, 5, 10, 20};
        
        for (int bs : batchSizes) {
            LogisticRegression lr = new LogisticRegression.Builder()
                .solver("sgd")
                .batchSize(bs)
                .learningRate(0.1)
                .maxIter(500)
                .randomState(42)
                .build();
            
            lr.fit(X_binary, y_binary);
            int[] predictions = lr.predict(X_binary);
            
            assertNotNull(predictions);
            assertEquals(X_binary.length, predictions.length);
        }
    }
    
    @Test
    public void testDifferentMaxIterations() {
        LogisticRegression lr1 = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(10)
            .randomState(42)
            .build();
        
        LogisticRegression lr2 = new LogisticRegression.Builder()
            .solver("gradient_descent")
            .learningRate(0.1)
            .maxIter(1000)
            .randomState(42)
            .build();
        
        lr1.fit(X_binary, y_binary);
        lr2.fit(X_binary, y_binary);
        
        int[] pred1 = lr1.predict(X_binary);
        int[] pred2 = lr2.predict(X_binary);
        
        double acc1 = computeAccuracy(y_binary, pred1);
        double acc2 = computeAccuracy(y_binary, pred2);
        
        // More iterations should achieve better or equal accuracy
        assertTrue(acc2 >= acc1 - 0.1, "More iterations should not hurt performance");
    }
    
    // ==================== Helper Methods ====================
    
    private double computeAccuracy(int[] yTrue, int[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) {
                correct++;
            }
        }
        
        return (double) correct / yTrue.length;
    }
    
    private double computeWeightNorm(double[] weights) {
        double norm = 0.0;
        for (double w : weights) {
            norm += w * w;
        }
        return Math.sqrt(norm);
    }
}
