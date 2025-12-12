package io.github.yasmramos.mindforge.classification;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Bernoulli Naive Bayes classifier.
 */
public class BernoulliNaiveBayesTest {
    
    private double[][] XBinary;
    private int[] yBinary;
    
    @BeforeEach
    public void setUp() {
        // Binary feature dataset (word presence/absence)
        // Features represent whether specific words appear in documents
        // Class 0: spam, Class 1: not spam
        
        XBinary = new double[][] {
            // Spam documents (class 0)
            {1.0, 1.0, 0.0, 1.0}, // "free" "win" "meeting" "click"
            {1.0, 1.0, 0.0, 0.0},
            {1.0, 0.0, 0.0, 1.0},
            {1.0, 1.0, 1.0, 1.0},
            
            // Not spam documents (class 1)
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0, 1.0},
            {0.0, 0.0, 1.0, 0.0},
            {1.0, 0.0, 1.0, 0.0}
        };
        
        yBinary = new int[] {0, 0, 0, 0, 1, 1, 1, 1};
    }
    
    @Test
    public void testBasicTrainAndPredict() {
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        bnb.train(XBinary, yBinary);
        
        assertTrue(bnb.isTrained());
        assertEquals(2, bnb.getNumClasses());
        
        // Document with spam words should be class 0
        int pred1 = bnb.predict(new double[]{1.0, 1.0, 0.0, 1.0});
        assertEquals(0, pred1);
        
        // Document with non-spam words should be class 1
        int pred2 = bnb.predict(new double[]{0.0, 0.0, 1.0, 0.0});
        assertEquals(1, pred2);
    }
    
    @Test
    public void testTrainingDataAccuracy() {
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        bnb.train(XBinary, yBinary);
        
        int[] predictions = bnb.predict(XBinary);
        int correct = 0;
        for (int i = 0; i < yBinary.length; i++) {
            if (predictions[i] == yBinary[i]) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / yBinary.length;
        assertTrue(accuracy >= 0.6, "Accuracy should be reasonable");
    }
    
    @Test
    public void testPredictProba() {
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        bnb.train(XBinary, yBinary);
        
        double[] probs = bnb.predictProba(new double[]{1.0, 1.0, 0.0, 0.0});
        
        // Probabilities should sum to 1
        double sum = 0.0;
        for (double p : probs) {
            sum += p;
        }
        assertEquals(1.0, sum, 1e-6);
        
        // All probabilities should be valid
        for (double p : probs) {
            assertTrue(p >= 0.0 && p <= 1.0);
        }
    }
    
    @Test
    public void testPredictProbaMultiple() {
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        bnb.train(XBinary, yBinary);
        
        double[][] probabilities = bnb.predictProba(XBinary);
        
        assertEquals(XBinary.length, probabilities.length);
        
        for (double[] probs : probabilities) {
            assertEquals(2, probs.length);
            
            double sum = 0.0;
            for (double p : probs) {
                assertTrue(p >= 0.0 && p <= 1.0);
                sum += p;
            }
            assertEquals(1.0, sum, 1e-6);
        }
    }
    
    @Test
    public void testBinarization() {
        // Test with continuous features that need binarization
        double[][] XContinuous = {
            {5.0, 3.0, 0.0, 8.0}, // Will be binarized to {1, 1, 0, 1} with threshold 0.5
            {0.1, 0.2, 3.0, 0.3},
            {6.0, 4.0, 0.0, 5.0},
            {0.0, 0.0, 4.0, 0.0}
        };
        int[] y = {0, 1, 0, 1};
        
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes(1.0, 0.5, true);
        bnb.train(XContinuous, y);
        
        assertTrue(bnb.isTrained());
        assertEquals(0.5, bnb.getBinarize());
        
        int[] predictions = bnb.predict(XContinuous);
        assertEquals(4, predictions.length);
    }
    
    @Test
    public void testNoBinarization() {
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes(1.0, -1.0, true);
        bnb.train(XBinary, yBinary);
        
        assertEquals(-1.0, bnb.getBinarize());
        assertTrue(bnb.isTrained());
    }
    
    @Test
    public void testUniformPriors() {
        // Test with fit_prior=false (uniform priors)
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes(1.0, -1.0, false);
        bnb.train(XBinary, yBinary);
        
        double[] priors = bnb.getClassPriors();
        
        // All priors should be equal (1/numClasses)
        assertEquals(0.5, priors[0], 1e-6);
        assertEquals(0.5, priors[1], 1e-6);
    }
    
    @Test
    public void testFittedPriors() {
        // Test with fit_prior=true (learn from data)
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes(1.0, -1.0, true);
        bnb.train(XBinary, yBinary);
        
        double[] priors = bnb.getClassPriors();
        
        // Priors should match class distribution (50-50 in this case)
        assertEquals(0.5, priors[0], 1e-6);
        assertEquals(0.5, priors[1], 1e-6);
    }
    
    @Test
    public void testImbalancedPriors() {
        double[][] X = {
            {1.0, 0.0}, {1.0, 1.0}, {1.0, 0.0},
            {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0},
            {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}
        };
        int[] y = {0, 0, 0, 1, 1, 1, 1, 1, 1}; // 3 class 0, 6 class 1
        
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes(1.0, -1.0, true);
        bnb.train(X, y);
        
        double[] priors = bnb.getClassPriors();
        assertEquals(3.0 / 9.0, priors[0], 1e-6);
        assertEquals(6.0 / 9.0, priors[1], 1e-6);
    }
    
    @Test
    public void testGetters() {
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes(0.5, 0.3, true);
        bnb.train(XBinary, yBinary);
        
        assertEquals(0.5, bnb.getAlpha());
        assertEquals(0.3, bnb.getBinarize());
        assertEquals(2, bnb.getNumClasses());
        
        int[] classes = bnb.getClasses();
        assertEquals(2, classes.length);
        
        double[] priors = bnb.getClassPriors();
        assertEquals(2, priors.length);
        
        double[][] probs = bnb.getFeatureProbs();
        assertEquals(2, probs.length);
        assertEquals(4, probs[0].length);
        
        // Feature probabilities should be between 0 and 1
        for (double[] classProbs : probs) {
            for (double p : classProbs) {
                assertTrue(p >= 0.0 && p <= 1.0);
            }
        }
    }
    
    @Test
    public void testDifferentAlphaValues() {
        BernoulliNaiveBayes bnb0 = new BernoulliNaiveBayes(0.0);
        bnb0.train(XBinary, yBinary);
        assertTrue(bnb0.isTrained());
        
        BernoulliNaiveBayes bnb1 = new BernoulliNaiveBayes(1.0);
        bnb1.train(XBinary, yBinary);
        assertTrue(bnb1.isTrained());
        
        BernoulliNaiveBayes bnb05 = new BernoulliNaiveBayes(0.5);
        bnb05.train(XBinary, yBinary);
        assertTrue(bnb05.isTrained());
    }
    
    @Test
    public void testFeatureAbsencePenalty() {
        // This tests that Bernoulli considers feature absence
        // (unlike Multinomial which only considers presence)
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        bnb.train(XBinary, yBinary);
        
        // Document with no features
        double[] allZeros = {0.0, 0.0, 0.0, 0.0};
        int prediction = bnb.predict(allZeros);
        
        // Should still make a prediction based on feature absence
        assertTrue(prediction == 0 || prediction == 1);
        
        double[] probs = bnb.predictProba(allZeros);
        double sum = 0.0;
        for (double p : probs) {
            sum += p;
        }
        assertEquals(1.0, sum, 1e-6);
    }
    
    // Edge cases and error handling
    
    @Test
    public void testPredictBeforeTraining() {
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        
        assertThrows(IllegalStateException.class, () -> {
            bnb.predict(new double[]{1.0, 0.0, 1.0, 0.0});
        });
    }
    
    @Test
    public void testPredictProbaBeforeTraining() {
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        
        assertThrows(IllegalStateException.class, () -> {
            bnb.predictProba(new double[]{1.0, 0.0, 1.0, 0.0});
        });
    }
    
    @Test
    public void testNegativeAlpha() {
        assertThrows(IllegalArgumentException.class, () -> {
            new BernoulliNaiveBayes(-1.0);
        });
    }
    
    @Test
    public void testMismatchedArrays() {
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        double[][] X = {{1.0, 0.0}, {0.0, 1.0}};
        int[] y = {0};
        
        assertThrows(IllegalArgumentException.class, () -> {
            bnb.train(X, y);
        });
    }
    
    @Test
    public void testEmptyData() {
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        double[][] X = {};
        int[] y = {};
        
        assertThrows(IllegalArgumentException.class, () -> {
            bnb.train(X, y);
        });
    }
    
    @Test
    public void testWrongFeatureCount() {
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        bnb.train(XBinary, yBinary);
        
        assertThrows(IllegalArgumentException.class, () -> {
            bnb.predict(new double[]{1.0, 0.0}); // Only 2 features, expected 4
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            bnb.predictProba(new double[]{1.0, 0.0, 1.0}); // Only 3 features, expected 4
        });
    }
    
    @Test
    public void testMulticlass() {
        double[][] X = {
            {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0},
            {0.0, 1.0, 0.0}, {0.0, 1.0, 1.0},
            {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}
        };
        int[] y = {0, 0, 1, 1, 2, 2};
        
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        bnb.train(X, y);
        
        assertEquals(3, bnb.getNumClasses());
        
        int[] predictions = bnb.predict(X);
        assertEquals(6, predictions.length);
    }
    
    @Test
    public void testNonStandardLabels() {
        double[][] X = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}};
        int[] y = {5, 10, 5, 10};
        
        BernoulliNaiveBayes bnb = new BernoulliNaiveBayes();
        bnb.train(X, y);
        
        assertEquals(2, bnb.getNumClasses());
        
        int[] predictions = bnb.predict(X);
        for (int pred : predictions) {
            assertTrue(pred == 5 || pred == 10);
        }
    }
}
