package com.mindforge.classification;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Gaussian Naive Bayes classifier.
 */
public class GaussianNaiveBayesTest {
    
    private double[][] X;
    private int[] y;
    private double[][] XBinary;
    private int[] yBinary;
    
    @BeforeEach
    public void setUp() {
        // Binary classification dataset
        // Class 0: mean around (-2, -2), Class 1: mean around (2, 2)
        XBinary = new double[][] {
            {-2.0, -2.0}, {-1.8, -2.2}, {-2.2, -1.8}, {-1.9, -2.1},
            {-2.1, -1.9}, {-1.7, -2.3}, {-2.3, -1.7}, {-1.6, -2.4},
            {2.0, 2.0}, {1.8, 2.2}, {2.2, 1.8}, {1.9, 2.1},
            {2.1, 1.9}, {1.7, 2.3}, {2.3, 1.7}, {1.6, 2.4}
        };
        
        yBinary = new int[] {
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1
        };
        
        // Multiclass dataset (3 classes)
        // Class 0: mean around (-3, 0), Class 1: mean around (0, 3), Class 2: mean around (3, 0)
        X = new double[][] {
            {-3.0, 0.0}, {-2.8, 0.2}, {-3.2, -0.2}, {-2.9, 0.1}, {-3.1, -0.1},
            {0.0, 3.0}, {0.2, 2.8}, {-0.2, 3.2}, {0.1, 2.9}, {-0.1, 3.1},
            {3.0, 0.0}, {2.8, -0.2}, {3.2, 0.2}, {2.9, -0.1}, {3.1, 0.1}
        };
        
        y = new int[] {
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2
        };
    }
    
    @Test
    public void testBinaryClassificationBasic() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XBinary, yBinary);
        
        assertEquals(2, gnb.getNumClasses());
        assertTrue(gnb.isTrained());
        
        // Test predictions on training data
        int[] predictions = gnb.predict(XBinary);
        int correct = 0;
        for (int i = 0; i < yBinary.length; i++) {
            if (predictions[i] == yBinary[i]) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / yBinary.length;
        assertTrue(accuracy >= 0.9, "Accuracy should be high on training data");
    }
    
    @Test
    public void testBinaryClassificationPrediction() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XBinary, yBinary);
        
        // Test clear cases
        assertEquals(0, gnb.predict(new double[]{-2.0, -2.0}));
        assertEquals(1, gnb.predict(new double[]{2.0, 2.0}));
        
        // Test intermediate points
        int pred1 = gnb.predict(new double[]{-1.0, -1.0});
        int pred2 = gnb.predict(new double[]{1.0, 1.0});
        
        // Should predict different classes for opposite quadrants
        assertNotEquals(pred1, pred2);
    }
    
    @Test
    public void testMulticlassClassification() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(X, y);
        
        assertEquals(3, gnb.getNumClasses());
        
        // Test predictions
        assertEquals(0, gnb.predict(new double[]{-3.0, 0.0}));
        assertEquals(1, gnb.predict(new double[]{0.0, 3.0}));
        assertEquals(2, gnb.predict(new double[]{3.0, 0.0}));
    }
    
    @Test
    public void testMulticlassAccuracy() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(X, y);
        
        int[] predictions = gnb.predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / y.length;
        assertTrue(accuracy >= 0.8, "Multiclass accuracy should be high");
    }
    
    @Test
    public void testPredictProba() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XBinary, yBinary);
        
        double[] probs = gnb.predictProba(new double[]{-2.0, -2.0});
        
        // Check probabilities sum to 1
        double sum = 0.0;
        for (double p : probs) {
            sum += p;
        }
        assertEquals(1.0, sum, 1e-6);
        
        // Check all probabilities are valid
        for (double p : probs) {
            assertTrue(p >= 0.0 && p <= 1.0);
        }
        
        // Check that class 0 has higher probability for point near class 0 mean
        assertTrue(probs[0] > probs[1]);
    }
    
    @Test
    public void testPredictProbaMultiple() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(X, y);
        
        double[][] probabilities = gnb.predictProba(X);
        
        assertEquals(X.length, probabilities.length);
        
        for (double[] probs : probabilities) {
            assertEquals(3, probs.length);
            
            double sum = 0.0;
            for (double p : probs) {
                assertTrue(p >= 0.0 && p <= 1.0);
                sum += p;
            }
            assertEquals(1.0, sum, 1e-6);
        }
    }
    
    @Test
    public void testPredictProbaConsistency() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XBinary, yBinary);
        
        double[] x = {-2.0, -2.0};
        
        // Prediction should match highest probability
        int prediction = gnb.predict(x);
        double[] probs = gnb.predictProba(x);
        
        int maxIdx = 0;
        double maxProb = probs[0];
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }
        
        assertEquals(prediction, gnb.getClasses()[maxIdx]);
    }
    
    @Test
    public void testGetClassPriors() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XBinary, yBinary);
        
        double[] priors = gnb.getClassPriors();
        
        assertEquals(2, priors.length);
        
        // With balanced dataset, priors should be 0.5
        assertEquals(0.5, priors[0], 1e-6);
        assertEquals(0.5, priors[1], 1e-6);
        
        // Sum should be 1
        double sum = 0.0;
        for (double p : priors) {
            sum += p;
        }
        assertEquals(1.0, sum, 1e-6);
    }
    
    @Test
    public void testGetClassPriorsImbalanced() {
        // Create imbalanced dataset: 10 samples class 0, 5 samples class 1
        double[][] XImb = new double[15][2];
        int[] yImb = new int[15];
        
        for (int i = 0; i < 10; i++) {
            XImb[i] = new double[]{-2.0 + i * 0.1, -2.0};
            yImb[i] = 0;
        }
        
        for (int i = 10; i < 15; i++) {
            XImb[i] = new double[]{2.0 + (i - 10) * 0.1, 2.0};
            yImb[i] = 1;
        }
        
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XImb, yImb);
        
        double[] priors = gnb.getClassPriors();
        
        assertEquals(10.0 / 15.0, priors[0], 1e-6);
        assertEquals(5.0 / 15.0, priors[1], 1e-6);
    }
    
    @Test
    public void testGetMeans() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XBinary, yBinary);
        
        double[][] means = gnb.getMeans();
        
        assertEquals(2, means.length); // 2 classes
        assertEquals(2, means[0].length); // 2 features
        
        // Class 0 mean should be around (-2, -2)
        assertTrue(Math.abs(means[0][0] - (-2.0)) < 0.5);
        assertTrue(Math.abs(means[0][1] - (-2.0)) < 0.5);
        
        // Class 1 mean should be around (2, 2)
        assertTrue(Math.abs(means[1][0] - 2.0) < 0.5);
        assertTrue(Math.abs(means[1][1] - 2.0) < 0.5);
    }
    
    @Test
    public void testGetVariances() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XBinary, yBinary);
        
        double[][] variances = gnb.getVariances();
        
        assertEquals(2, variances.length); // 2 classes
        assertEquals(2, variances[0].length); // 2 features
        
        // Variances should be positive
        for (double[] classVar : variances) {
            for (double v : classVar) {
                assertTrue(v > 0.0);
            }
        }
    }
    
    @Test
    public void testGetClasses() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(X, y);
        
        int[] classes = gnb.getClasses();
        
        assertEquals(3, classes.length);
        assertArrayEquals(new int[]{0, 1, 2}, classes);
    }
    
    @Test
    public void testCustomEpsilon() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes(1e-6);
        gnb.train(XBinary, yBinary);
        
        int prediction = gnb.predict(new double[]{-2.0, -2.0});
        assertEquals(0, prediction);
    }
    
    @Test
    public void testNonStandardLabels() {
        // Use labels 5 and 10 instead of 0 and 1
        double[][] XNS = new double[][]{
            {-2.0, -2.0}, {-1.8, -2.2}, {-2.2, -1.8},
            {2.0, 2.0}, {1.8, 2.2}, {2.2, 1.8}
        };
        int[] yNS = {5, 5, 5, 10, 10, 10};
        
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XNS, yNS);
        
        assertEquals(2, gnb.getNumClasses());
        
        int pred1 = gnb.predict(new double[]{-2.0, -2.0});
        int pred2 = gnb.predict(new double[]{2.0, 2.0});
        
        assertTrue(pred1 == 5 || pred1 == 10);
        assertTrue(pred2 == 5 || pred2 == 10);
        assertNotEquals(pred1, pred2);
    }
    
    // Edge cases and error handling
    
    @Test
    public void testPredictBeforeTraining() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        
        assertThrows(IllegalStateException.class, () -> {
            gnb.predict(new double[]{1.0, 2.0});
        });
    }
    
    @Test
    public void testPredictProbaBeforeTraining() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        
        assertThrows(IllegalStateException.class, () -> {
            gnb.predictProba(new double[]{1.0, 2.0});
        });
    }
    
    @Test
    public void testTrainWithMismatchedArrays() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        double[][] X = {{1.0, 2.0}, {3.0, 4.0}};
        int[] y = {0};
        
        assertThrows(IllegalArgumentException.class, () -> {
            gnb.train(X, y);
        });
    }
    
    @Test
    public void testTrainWithEmptyData() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        double[][] X = {};
        int[] y = {};
        
        assertThrows(IllegalArgumentException.class, () -> {
            gnb.train(X, y);
        });
    }
    
    @Test
    public void testPredictWithWrongFeatureCount() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XBinary, yBinary);
        
        assertThrows(IllegalArgumentException.class, () -> {
            gnb.predict(new double[]{1.0}); // Only 1 feature, expected 2
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            gnb.predict(new double[]{1.0, 2.0, 3.0}); // 3 features, expected 2
        });
    }
    
    @Test
    public void testPredictProbaWithWrongFeatureCount() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XBinary, yBinary);
        
        assertThrows(IllegalArgumentException.class, () -> {
            gnb.predictProba(new double[]{1.0});
        });
    }
    
    @Test
    public void testInvalidEpsilon() {
        assertThrows(IllegalArgumentException.class, () -> {
            new GaussianNaiveBayes(0.0);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new GaussianNaiveBayes(-1e-9);
        });
    }
    
    @Test
    public void testSingleSamplePerClass() {
        double[][] XSingle = {{-1.0, -1.0}, {1.0, 1.0}};
        int[] ySingle = {0, 1};
        
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XSingle, ySingle);
        
        assertEquals(2, gnb.getNumClasses());
        
        // Should still make reasonable predictions
        int[] predictions = gnb.predict(XSingle);
        assertEquals(2, predictions.length);
    }
    
    @Test
    public void testConsistentPredictions() {
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XBinary, yBinary);
        
        double[] x = {-2.0, -2.0};
        
        // Multiple predictions should be consistent
        int pred1 = gnb.predict(x);
        int pred2 = gnb.predict(x);
        int pred3 = gnb.predict(x);
        
        assertEquals(pred1, pred2);
        assertEquals(pred2, pred3);
    }
}
