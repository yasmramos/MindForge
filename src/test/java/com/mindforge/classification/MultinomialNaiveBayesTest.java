package com.mindforge.classification;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Multinomial Naive Bayes classifier.
 */
public class MultinomialNaiveBayesTest {
    
    private double[][] XCounts;
    private int[] yCounts;
    
    @BeforeEach
    public void setUp() {
        // Simple word count dataset for document classification
        // Features represent word frequencies
        // Class 0: documents about sports
        // Class 1: documents about technology
        
        XCounts = new double[][] {
            // Sports documents (class 0)
            {5.0, 2.0, 0.0, 1.0}, // "game" "ball" "code" "python"
            {4.0, 3.0, 0.0, 0.0},
            {6.0, 1.0, 0.0, 2.0},
            {3.0, 4.0, 1.0, 0.0},
            
            // Technology documents (class 1)
            {0.0, 0.0, 5.0, 4.0},
            {1.0, 0.0, 6.0, 3.0},
            {0.0, 0.0, 4.0, 5.0},
            {0.0, 1.0, 7.0, 2.0}
        };
        
        yCounts = new int[] {0, 0, 0, 0, 1, 1, 1, 1};
    }
    
    @Test
    public void testBasicTrainAndPredict() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        mnb.train(XCounts, yCounts);
        
        assertTrue(mnb.isTrained());
        assertEquals(2, mnb.getNumClasses());
        
        // Document with sports words should be class 0
        int pred1 = mnb.predict(new double[]{5.0, 3.0, 0.0, 0.0});
        assertEquals(0, pred1);
        
        // Document with tech words should be class 1
        int pred2 = mnb.predict(new double[]{0.0, 0.0, 6.0, 4.0});
        assertEquals(1, pred2);
    }
    
    @Test
    public void testTrainingDataAccuracy() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        mnb.train(XCounts, yCounts);
        
        int[] predictions = mnb.predict(XCounts);
        int correct = 0;
        for (int i = 0; i < yCounts.length; i++) {
            if (predictions[i] == yCounts[i]) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / yCounts.length;
        assertTrue(accuracy >= 0.7, "Accuracy should be reasonable");
    }
    
    @Test
    public void testPredictProba() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        mnb.train(XCounts, yCounts);
        
        double[] probs = mnb.predictProba(new double[]{5.0, 3.0, 0.0, 0.0});
        
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
        
        // Sports document should have higher probability for class 0
        assertTrue(probs[0] > probs[1]);
    }
    
    @Test
    public void testPredictProbaMultiple() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        mnb.train(XCounts, yCounts);
        
        double[][] probabilities = mnb.predictProba(XCounts);
        
        assertEquals(XCounts.length, probabilities.length);
        
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
    public void testDifferentAlphaValues() {
        // Test with no smoothing
        MultinomialNaiveBayes mnb0 = new MultinomialNaiveBayes(0.0);
        mnb0.train(XCounts, yCounts);
        assertTrue(mnb0.isTrained());
        
        // Test with Laplace smoothing
        MultinomialNaiveBayes mnb1 = new MultinomialNaiveBayes(1.0);
        mnb1.train(XCounts, yCounts);
        assertTrue(mnb1.isTrained());
        
        // Test with different smoothing
        MultinomialNaiveBayes mnb05 = new MultinomialNaiveBayes(0.5);
        mnb05.train(XCounts, yCounts);
        assertTrue(mnb05.isTrained());
        
        // All should make predictions
        assertNotNull(mnb0.predict(XCounts));
        assertNotNull(mnb1.predict(XCounts));
        assertNotNull(mnb05.predict(XCounts));
    }
    
    @Test
    public void testGetters() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes(1.0);
        mnb.train(XCounts, yCounts);
        
        assertEquals(1.0, mnb.getAlpha());
        assertEquals(2, mnb.getNumClasses());
        
        int[] classes = mnb.getClasses();
        assertEquals(2, classes.length);
        
        double[] priors = mnb.getClassPriors();
        assertEquals(2, priors.length);
        assertEquals(0.5, priors[0], 1e-6);
        assertEquals(0.5, priors[1], 1e-6);
        
        double[][] logProbs = mnb.getFeatureLogProbs();
        assertEquals(2, logProbs.length);
        assertEquals(4, logProbs[0].length);
    }
    
    @Test
    public void testZeroFeatures() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        mnb.train(XCounts, yCounts);
        
        // Document with all zeros should still make a prediction
        int prediction = mnb.predict(new double[]{0.0, 0.0, 0.0, 0.0});
        assertTrue(prediction == 0 || prediction == 1);
    }
    
    @Test
    public void testSparseFeatures() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        mnb.train(XCounts, yCounts);
        
        // Document with mostly zeros (sparse)
        int prediction = mnb.predict(new double[]{0.0, 0.0, 10.0, 0.0});
        assertEquals(1, prediction); // Should be tech class
    }
    
    @Test
    public void testConsistencyWithAlpha() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes(1.0);
        assertEquals(1.0, mnb.getAlpha());
    }
    
    // Edge cases and error handling
    
    @Test
    public void testPredictBeforeTraining() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        
        assertThrows(IllegalStateException.class, () -> {
            mnb.predict(new double[]{1.0, 2.0, 3.0, 4.0});
        });
    }
    
    @Test
    public void testPredictProbaBeforeTraining() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        
        assertThrows(IllegalStateException.class, () -> {
            mnb.predictProba(new double[]{1.0, 2.0, 3.0, 4.0});
        });
    }
    
    @Test
    public void testNegativeAlpha() {
        assertThrows(IllegalArgumentException.class, () -> {
            new MultinomialNaiveBayes(-1.0);
        });
    }
    
    @Test
    public void testNegativeFeatureValue() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        
        double[][] XNeg = {{-1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}};
        int[] yNeg = {0, 1};
        
        assertThrows(IllegalArgumentException.class, () -> {
            mnb.train(XNeg, yNeg);
        });
    }
    
    @Test
    public void testNegativeFeatureInPredict() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        mnb.train(XCounts, yCounts);
        
        assertThrows(IllegalArgumentException.class, () -> {
            mnb.predict(new double[]{-1.0, 2.0, 3.0, 4.0});
        });
    }
    
    @Test
    public void testMismatchedArrays() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        double[][] X = {{1.0, 2.0}, {3.0, 4.0}};
        int[] y = {0};
        
        assertThrows(IllegalArgumentException.class, () -> {
            mnb.train(X, y);
        });
    }
    
    @Test
    public void testEmptyData() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        double[][] X = {};
        int[] y = {};
        
        assertThrows(IllegalArgumentException.class, () -> {
            mnb.train(X, y);
        });
    }
    
    @Test
    public void testWrongFeatureCount() {
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        mnb.train(XCounts, yCounts);
        
        assertThrows(IllegalArgumentException.class, () -> {
            mnb.predict(new double[]{1.0, 2.0}); // Only 2 features, expected 4
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            mnb.predictProba(new double[]{1.0, 2.0, 3.0}); // Only 3 features, expected 4
        });
    }
    
    @Test
    public void testNonStandardLabels() {
        double[][] X = {{5.0, 2.0}, {1.0, 5.0}, {6.0, 1.0}, {0.0, 6.0}};
        int[] y = {10, 20, 10, 20};
        
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        mnb.train(X, y);
        
        assertEquals(2, mnb.getNumClasses());
        
        int[] predictions = mnb.predict(X);
        for (int pred : predictions) {
            assertTrue(pred == 10 || pred == 20);
        }
    }
    
    @Test
    public void testMulticlass() {
        double[][] X = {
            {5.0, 0.0, 0.0}, {4.0, 1.0, 0.0},
            {0.0, 5.0, 0.0}, {0.0, 4.0, 1.0},
            {0.0, 0.0, 5.0}, {1.0, 0.0, 4.0}
        };
        int[] y = {0, 0, 1, 1, 2, 2};
        
        MultinomialNaiveBayes mnb = new MultinomialNaiveBayes();
        mnb.train(X, y);
        
        assertEquals(3, mnb.getNumClasses());
        
        assertEquals(0, mnb.predict(new double[]{5.0, 0.0, 0.0}));
        assertEquals(1, mnb.predict(new double[]{0.0, 5.0, 0.0}));
        assertEquals(2, mnb.predict(new double[]{0.0, 0.0, 5.0}));
    }
}
