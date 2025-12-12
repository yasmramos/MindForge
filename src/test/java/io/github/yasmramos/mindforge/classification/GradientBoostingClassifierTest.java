package io.github.yasmramos.mindforge.classification;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Comprehensive tests for GradientBoostingClassifier.
 */
class GradientBoostingClassifierTest {
    
    private double[][] binaryFeatures;
    private int[] binaryLabels;
    private double[][] multiclassFeatures;
    private int[] multiclassLabels;
    
    @BeforeEach
    void setUp() {
        // Binary classification dataset (XOR-like)
        binaryFeatures = new double[][] {
            {0, 0}, {0, 1}, {1, 0}, {1, 1},
            {0.1, 0.1}, {0.1, 0.9}, {0.9, 0.1}, {0.9, 0.9},
            {0.2, 0.2}, {0.2, 0.8}, {0.8, 0.2}, {0.8, 0.8}
        };
        binaryLabels = new int[] {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0};
        
        // Multiclass classification dataset (3 clusters)
        multiclassFeatures = new double[][] {
            // Class 0: bottom-left cluster
            {0.1, 0.1}, {0.2, 0.2}, {0.15, 0.15}, {0.1, 0.2}, {0.2, 0.1},
            // Class 1: top-right cluster
            {0.8, 0.8}, {0.9, 0.9}, {0.85, 0.85}, {0.8, 0.9}, {0.9, 0.8},
            // Class 2: top-left cluster
            {0.1, 0.8}, {0.2, 0.9}, {0.15, 0.85}, {0.1, 0.9}, {0.2, 0.8}
        };
        multiclassLabels = new int[] {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
    }
    
    // ==================== Constructor Tests ====================
    
    @Test
    @DisplayName("Default constructor creates valid classifier")
    void testDefaultConstructor() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier();
        
        assertEquals(100, gb.getNEstimators());
        assertEquals(0.1, gb.getLearningRate(), 1e-10);
        assertEquals(3, gb.getMaxDepth());
        assertEquals(1.0, gb.getSubsample(), 1e-10);
        assertFalse(gb.isTrained());
    }
    
    @Test
    @DisplayName("Custom constructor sets parameters correctly")
    void testCustomConstructor() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier(50, 0.05, 5, 0.8, 42);
        
        assertEquals(50, gb.getNEstimators());
        assertEquals(0.05, gb.getLearningRate(), 1e-10);
        assertEquals(5, gb.getMaxDepth());
        assertEquals(0.8, gb.getSubsample(), 1e-10);
    }
    
    @Test
    @DisplayName("Builder pattern creates classifier correctly")
    void testBuilderPattern() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(75)
            .learningRate(0.15)
            .maxDepth(4)
            .subsample(0.9)
            .randomState(123)
            .build();
        
        assertEquals(75, gb.getNEstimators());
        assertEquals(0.15, gb.getLearningRate(), 1e-10);
        assertEquals(4, gb.getMaxDepth());
        assertEquals(0.9, gb.getSubsample(), 1e-10);
    }
    
    @Test
    @DisplayName("Invalid nEstimators throws exception")
    void testInvalidNEstimators() {
        assertThrows(IllegalArgumentException.class, () -> 
            new GradientBoostingClassifier(0, 0.1, 3, 1.0, null));
        assertThrows(IllegalArgumentException.class, () -> 
            new GradientBoostingClassifier(-5, 0.1, 3, 1.0, null));
    }
    
    @Test
    @DisplayName("Invalid learning rate throws exception")
    void testInvalidLearningRate() {
        assertThrows(IllegalArgumentException.class, () -> 
            new GradientBoostingClassifier(100, 0, 3, 1.0, null));
        assertThrows(IllegalArgumentException.class, () -> 
            new GradientBoostingClassifier(100, -0.1, 3, 1.0, null));
        assertThrows(IllegalArgumentException.class, () -> 
            new GradientBoostingClassifier(100, 1.5, 3, 1.0, null));
    }
    
    @Test
    @DisplayName("Invalid maxDepth throws exception")
    void testInvalidMaxDepth() {
        assertThrows(IllegalArgumentException.class, () -> 
            new GradientBoostingClassifier(100, 0.1, 0, 1.0, null));
        assertThrows(IllegalArgumentException.class, () -> 
            new GradientBoostingClassifier(100, 0.1, -1, 1.0, null));
    }
    
    @Test
    @DisplayName("Invalid subsample throws exception")
    void testInvalidSubsample() {
        assertThrows(IllegalArgumentException.class, () -> 
            new GradientBoostingClassifier(100, 0.1, 3, 0, null));
        assertThrows(IllegalArgumentException.class, () -> 
            new GradientBoostingClassifier(100, 0.1, 3, -0.5, null));
        assertThrows(IllegalArgumentException.class, () -> 
            new GradientBoostingClassifier(100, 0.1, 3, 1.5, null));
    }
    
    // ==================== Training Tests ====================
    
    @Test
    @DisplayName("Fit with binary classification data")
    void testFitBinary() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(50)
            .learningRate(0.1)
            .maxDepth(3)
            .randomState(42)
            .build();
        
        assertDoesNotThrow(() -> gb.fit(binaryFeatures, binaryLabels));
        assertTrue(gb.isTrained());
        assertEquals(50, gb.getNumTrees());
        
        int[] classes = gb.getClasses();
        assertNotNull(classes);
        assertEquals(2, classes.length);
    }
    
    @Test
    @DisplayName("Fit with multiclass classification data")
    void testFitMulticlass() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(30)
            .learningRate(0.1)
            .maxDepth(3)
            .randomState(42)
            .build();
        
        assertDoesNotThrow(() -> gb.fit(multiclassFeatures, multiclassLabels));
        assertTrue(gb.isTrained());
        // 3 classes * 30 estimators = 90 trees
        assertEquals(90, gb.getNumTrees());
        
        int[] classes = gb.getClasses();
        assertNotNull(classes);
        assertEquals(3, classes.length);
    }
    
    @Test
    @DisplayName("Fit with null features throws exception")
    void testFitNullFeatures() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier();
        assertThrows(IllegalArgumentException.class, () -> gb.fit(null, binaryLabels));
    }
    
    @Test
    @DisplayName("Fit with null labels throws exception")
    void testFitNullLabels() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier();
        assertThrows(IllegalArgumentException.class, () -> gb.fit(binaryFeatures, null));
    }
    
    @Test
    @DisplayName("Fit with mismatched lengths throws exception")
    void testFitMismatchedLengths() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier();
        int[] wrongLabels = new int[] {0, 1, 0};
        assertThrows(IllegalArgumentException.class, () -> gb.fit(binaryFeatures, wrongLabels));
    }
    
    @Test
    @DisplayName("Fit with empty data throws exception")
    void testFitEmptyData() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier();
        assertThrows(IllegalArgumentException.class, () -> gb.fit(new double[0][0], new int[0]));
    }
    
    @Test
    @DisplayName("Fit with single class throws exception")
    void testFitSingleClass() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier();
        int[] singleClassLabels = new int[] {0, 0, 0, 0};
        double[][] features = new double[][] {{1,2}, {3,4}, {5,6}, {7,8}};
        assertThrows(IllegalArgumentException.class, () -> gb.fit(features, singleClassLabels));
    }
    
    // ==================== Prediction Tests ====================
    
    @Test
    @DisplayName("Predict without training throws exception")
    void testPredictWithoutTraining() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier();
        assertThrows(IllegalStateException.class, () -> gb.predict(binaryFeatures));
    }
    
    @Test
    @DisplayName("Predict returns valid labels for binary classification")
    void testPredictBinary() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(50)
            .learningRate(0.1)
            .maxDepth(3)
            .randomState(42)
            .build();
        
        gb.fit(binaryFeatures, binaryLabels);
        int[] predictions = gb.predict(binaryFeatures);
        
        assertEquals(binaryFeatures.length, predictions.length);
        
        // Check predictions are valid labels
        Set<Integer> validLabels = new HashSet<>(Arrays.asList(0, 1));
        for (int pred : predictions) {
            assertTrue(validLabels.contains(pred));
        }
    }
    
    @Test
    @DisplayName("Predict returns valid labels for multiclass classification")
    void testPredictMulticlass() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(50)
            .learningRate(0.1)
            .maxDepth(3)
            .randomState(42)
            .build();
        
        gb.fit(multiclassFeatures, multiclassLabels);
        int[] predictions = gb.predict(multiclassFeatures);
        
        assertEquals(multiclassFeatures.length, predictions.length);
        
        // Check predictions are valid labels
        Set<Integer> validLabels = new HashSet<>(Arrays.asList(0, 1, 2));
        for (int pred : predictions) {
            assertTrue(validLabels.contains(pred));
        }
    }
    
    @Test
    @DisplayName("Single sample prediction works")
    void testPredictSingleSample() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(30)
            .randomState(42)
            .build();
        
        gb.fit(binaryFeatures, binaryLabels);
        
        int prediction = gb.predict(new double[] {0.5, 0.5});
        assertTrue(prediction == 0 || prediction == 1);
    }
    
    @Test
    @DisplayName("Training data prediction accuracy is reasonable")
    void testTrainingAccuracy() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(100)
            .learningRate(0.1)
            .maxDepth(5)
            .randomState(42)
            .build();
        
        // Simple linearly separable data
        double[][] features = new double[][] {
            {0, 0}, {0, 1}, {1, 0}, {1, 1},
            {2, 2}, {2, 3}, {3, 2}, {3, 3}
        };
        int[] labels = new int[] {0, 0, 0, 0, 1, 1, 1, 1};
        
        gb.fit(features, labels);
        int[] predictions = gb.predict(features);
        
        int correct = 0;
        for (int i = 0; i < labels.length; i++) {
            if (predictions[i] == labels[i]) correct++;
        }
        
        double accuracy = (double) correct / labels.length;
        assertTrue(accuracy >= 0.8, "Training accuracy should be at least 80%");
    }
    
    // ==================== Probability Prediction Tests ====================
    
    @Test
    @DisplayName("predictProba without training throws exception")
    void testPredictProbaWithoutTraining() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier();
        assertThrows(IllegalStateException.class, () -> gb.predictProba(binaryFeatures));
    }
    
    @Test
    @DisplayName("predictProba returns valid probabilities for binary classification")
    void testPredictProbaBinary() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(50)
            .randomState(42)
            .build();
        
        gb.fit(binaryFeatures, binaryLabels);
        double[][] probas = gb.predictProba(binaryFeatures);
        
        assertEquals(binaryFeatures.length, probas.length);
        assertEquals(2, probas[0].length);
        
        for (double[] proba : probas) {
            // Check probabilities are in [0, 1]
            for (double p : proba) {
                assertTrue(p >= 0 && p <= 1, "Probability should be in [0, 1]");
            }
            // Check probabilities sum to 1
            double sum = Arrays.stream(proba).sum();
            assertEquals(1.0, sum, 1e-6, "Probabilities should sum to 1");
        }
    }
    
    @Test
    @DisplayName("predictProba returns valid probabilities for multiclass classification")
    void testPredictProbaMulticlass() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(50)
            .randomState(42)
            .build();
        
        gb.fit(multiclassFeatures, multiclassLabels);
        double[][] probas = gb.predictProba(multiclassFeatures);
        
        assertEquals(multiclassFeatures.length, probas.length);
        assertEquals(3, probas[0].length);
        
        for (double[] proba : probas) {
            // Check probabilities are in [0, 1]
            for (double p : proba) {
                assertTrue(p >= 0 && p <= 1, "Probability should be in [0, 1]");
            }
            // Check probabilities sum to 1
            double sum = Arrays.stream(proba).sum();
            assertEquals(1.0, sum, 1e-6, "Probabilities should sum to 1");
        }
    }
    
    @Test
    @DisplayName("Single sample predictProba works")
    void testPredictProbaSingleSample() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(30)
            .randomState(42)
            .build();
        
        gb.fit(binaryFeatures, binaryLabels);
        
        double[] proba = gb.predictProba(new double[] {0.5, 0.5});
        assertEquals(2, proba.length);
        assertEquals(1.0, Arrays.stream(proba).sum(), 1e-6);
    }
    
    // ==================== Subsampling Tests ====================
    
    @Test
    @DisplayName("Stochastic gradient boosting with subsampling works")
    void testSubsampling() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(50)
            .learningRate(0.1)
            .maxDepth(3)
            .subsample(0.5)
            .randomState(42)
            .build();
        
        assertDoesNotThrow(() -> gb.fit(binaryFeatures, binaryLabels));
        assertTrue(gb.isTrained());
        
        int[] predictions = gb.predict(binaryFeatures);
        assertEquals(binaryFeatures.length, predictions.length);
    }
    
    // ==================== Reproducibility Tests ====================
    
    @Test
    @DisplayName("Same random state produces same results")
    void testReproducibility() {
        GradientBoostingClassifier gb1 = new GradientBoostingClassifier.Builder()
            .nEstimators(30)
            .randomState(42)
            .build();
        
        GradientBoostingClassifier gb2 = new GradientBoostingClassifier.Builder()
            .nEstimators(30)
            .randomState(42)
            .build();
        
        gb1.fit(binaryFeatures, binaryLabels);
        gb2.fit(binaryFeatures, binaryLabels);
        
        int[] pred1 = gb1.predict(binaryFeatures);
        int[] pred2 = gb2.predict(binaryFeatures);
        
        assertArrayEquals(pred1, pred2, "Same random state should produce same predictions");
    }
    
    @Test
    @DisplayName("Different random states may produce different results")
    void testDifferentRandomStates() {
        // Use subsampling to introduce randomness
        GradientBoostingClassifier gb1 = new GradientBoostingClassifier.Builder()
            .nEstimators(30)
            .subsample(0.5)
            .randomState(42)
            .build();
        
        GradientBoostingClassifier gb2 = new GradientBoostingClassifier.Builder()
            .nEstimators(30)
            .subsample(0.5)
            .randomState(123)
            .build();
        
        gb1.fit(binaryFeatures, binaryLabels);
        gb2.fit(binaryFeatures, binaryLabels);
        
        double[][] proba1 = gb1.predictProba(binaryFeatures);
        double[][] proba2 = gb2.predictProba(binaryFeatures);
        
        // At least some probabilities should differ
        boolean someDifferent = false;
        for (int i = 0; i < proba1.length && !someDifferent; i++) {
            for (int j = 0; j < proba1[i].length; j++) {
                if (Math.abs(proba1[i][j] - proba2[i][j]) > 1e-6) {
                    someDifferent = true;
                    break;
                }
            }
        }
        assertTrue(someDifferent, "Different random states should produce different results with subsampling");
    }
    
    // ==================== Edge Cases ====================
    
    @Test
    @DisplayName("Works with minimum estimators")
    void testMinimumEstimators() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(1)
            .randomState(42)
            .build();
        
        assertDoesNotThrow(() -> gb.fit(binaryFeatures, binaryLabels));
        int[] predictions = gb.predict(binaryFeatures);
        assertEquals(binaryFeatures.length, predictions.length);
    }
    
    @Test
    @DisplayName("Works with depth 1 (stumps)")
    void testDepthOne() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(50)
            .maxDepth(1)
            .randomState(42)
            .build();
        
        assertDoesNotThrow(() -> gb.fit(binaryFeatures, binaryLabels));
        int[] predictions = gb.predict(binaryFeatures);
        assertEquals(binaryFeatures.length, predictions.length);
    }
    
    @Test
    @DisplayName("Works with high depth")
    void testHighDepth() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(20)
            .maxDepth(10)
            .randomState(42)
            .build();
        
        assertDoesNotThrow(() -> gb.fit(binaryFeatures, binaryLabels));
        int[] predictions = gb.predict(binaryFeatures);
        assertEquals(binaryFeatures.length, predictions.length);
    }
    
    @Test
    @DisplayName("Works with very small learning rate")
    void testSmallLearningRate() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(100)
            .learningRate(0.001)
            .randomState(42)
            .build();
        
        assertDoesNotThrow(() -> gb.fit(binaryFeatures, binaryLabels));
        int[] predictions = gb.predict(binaryFeatures);
        assertEquals(binaryFeatures.length, predictions.length);
    }
    
    @Test
    @DisplayName("Works with learning rate = 1")
    void testMaxLearningRate() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(10)
            .learningRate(1.0)
            .randomState(42)
            .build();
        
        assertDoesNotThrow(() -> gb.fit(binaryFeatures, binaryLabels));
        int[] predictions = gb.predict(binaryFeatures);
        assertEquals(binaryFeatures.length, predictions.length);
    }
    
    // ==================== Integration Tests ====================
    
    @Test
    @DisplayName("Multiclass classification with more classes")
    void testFourClasses() {
        // 4-class classification
        double[][] features = new double[][] {
            {0, 0}, {0.1, 0.1},  // Class 0
            {1, 0}, {0.9, 0.1},  // Class 1
            {0, 1}, {0.1, 0.9},  // Class 2
            {1, 1}, {0.9, 0.9}   // Class 3
        };
        int[] labels = new int[] {0, 0, 1, 1, 2, 2, 3, 3};
        
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(50)
            .maxDepth(3)
            .randomState(42)
            .build();
        
        assertDoesNotThrow(() -> gb.fit(features, labels));
        
        int[] classes = gb.getClasses();
        assertEquals(4, classes.length);
        
        double[][] probas = gb.predictProba(features);
        assertEquals(4, probas[0].length);
        
        for (double[] proba : probas) {
            assertEquals(1.0, Arrays.stream(proba).sum(), 1e-6);
        }
    }
    
    @Test
    @DisplayName("Getters return correct values after training")
    void testGettersAfterTraining() {
        GradientBoostingClassifier gb = new GradientBoostingClassifier.Builder()
            .nEstimators(50)
            .learningRate(0.15)
            .maxDepth(4)
            .subsample(0.8)
            .randomState(42)
            .build();
        
        gb.fit(multiclassFeatures, multiclassLabels);
        
        assertEquals(50, gb.getNEstimators());
        assertEquals(0.15, gb.getLearningRate(), 1e-10);
        assertEquals(4, gb.getMaxDepth());
        assertEquals(0.8, gb.getSubsample(), 1e-10);
        assertTrue(gb.isTrained());
        assertEquals(150, gb.getNumTrees()); // 3 classes * 50 estimators
        
        int[] classes = gb.getClasses();
        assertNotNull(classes);
        assertArrayEquals(new int[]{0, 1, 2}, classes);
    }
}
