package io.github.yasmramos.mindforge.classification;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for RandomForestClassifier.
 */
public class RandomForestClassifierTest {
    
    private double[][] simpleData;
    private int[] simpleLabels;
    
    @BeforeEach
    public void setUp() {
        // Simple linearly separable dataset
        simpleData = new double[][] {
            {1.0, 1.0},
            {1.5, 1.5},
            {2.0, 2.0},
            {8.0, 8.0},
            {8.5, 8.5},
            {9.0, 9.0}
        };
        simpleLabels = new int[] {0, 0, 0, 1, 1, 1};
    }
    
    @Test
    public void testBasicFitAndPredict() {
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .randomState(42)
            .build();
        
        rf.fit(simpleData, simpleLabels);
        
        int[] predictions = rf.predict(simpleData);
        
        assertNotNull(predictions, "Predictions should not be null");
        assertEquals(simpleData.length, predictions.length, "Predictions length should match input");
        
        // Should perfectly classify training data
        assertArrayEquals(simpleLabels, predictions, "Should perfectly classify simple linearly separable data");
    }
    
    @Test
    public void testBinaryClassification() {
        // XOR-like problem that benefits from ensemble
        double[][] X = {
            {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},
            {0.1, 0.1}, {0.1, 0.9}, {0.9, 0.1}, {0.9, 0.9},
            {0.2, 0.2}, {0.2, 0.8}, {0.8, 0.2}, {0.8, 0.8}
        };
        int[] y = {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0};
        
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(50)
            .maxDepth(5)
            .randomState(42)
            .build();
        
        rf.fit(X, y);
        int[] predictions = rf.predict(X);
        
        // Calculate accuracy
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) correct++;
        }
        double accuracy = (double) correct / y.length;
        
        assertTrue(accuracy >= 0.75, "Should achieve at least 75% accuracy on XOR-like problem");
    }
    
    @Test
    public void testMulticlassClassification() {
        // Three-class problem
        double[][] X = {
            {1.0, 1.0}, {1.5, 1.5}, {2.0, 2.0},  // Class 0
            {5.0, 5.0}, {5.5, 5.5}, {6.0, 6.0},  // Class 1
            {9.0, 1.0}, {9.5, 1.5}, {10.0, 2.0}  // Class 2
        };
        int[] y = {0, 0, 0, 1, 1, 1, 2, 2, 2};
        
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(20)
            .randomState(42)
            .build();
        
        rf.fit(X, y);
        int[] predictions = rf.predict(X);
        
        assertArrayEquals(y, predictions, "Should perfectly classify three well-separated classes");
        
        int[] classes = rf.getClasses();
        assertEquals(3, classes.length, "Should have 3 classes");
        assertArrayEquals(new int[]{0, 1, 2}, classes, "Classes should be [0, 1, 2]");
    }
    
    @Test
    public void testPredictProba() {
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .randomState(42)
            .build();
        
        rf.fit(simpleData, simpleLabels);
        double[][] probabilities = rf.predictProba(simpleData);
        
        assertNotNull(probabilities, "Probabilities should not be null");
        assertEquals(simpleData.length, probabilities.length, "Should have probabilities for all samples");
        
        for (int i = 0; i < probabilities.length; i++) {
            assertEquals(2, probabilities[i].length, "Should have probabilities for 2 classes");
            
            // Probabilities should sum to approximately 1.0
            double sum = probabilities[i][0] + probabilities[i][1];
            assertEquals(1.0, sum, 0.0001, "Probabilities should sum to 1.0");
            
            // All probabilities should be in [0, 1]
            assertTrue(probabilities[i][0] >= 0.0 && probabilities[i][0] <= 1.0);
            assertTrue(probabilities[i][1] >= 0.0 && probabilities[i][1] <= 1.0);
            
            // Check that predicted class matches highest probability
            int predictedClass = probabilities[i][0] > probabilities[i][1] ? 0 : 1;
            assertEquals(simpleLabels[i], predictedClass, "Predicted class should match highest probability");
        }
    }
    
    @Test
    public void testOOBScore() {
        // Larger dataset for meaningful OOB score
        double[][] X = new double[100][2];
        int[] y = new int[100];
        
        for (int i = 0; i < 50; i++) {
            X[i][0] = 1.0 + Math.random() * 2.0;
            X[i][1] = 1.0 + Math.random() * 2.0;
            y[i] = 0;
        }
        for (int i = 50; i < 100; i++) {
            X[i][0] = 7.0 + Math.random() * 2.0;
            X[i][1] = 7.0 + Math.random() * 2.0;
            y[i] = 1;
        }
        
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(50)
            .bootstrap(true)
            .randomState(42)
            .build();
        
        rf.fit(X, y);
        double oobScore = rf.getOOBScore();
        
        assertFalse(Double.isNaN(oobScore), "OOB score should not be NaN with bootstrap=true");
        assertTrue(oobScore >= 0.7, "OOB score should be at least 0.7 for well-separated classes");
        assertTrue(oobScore <= 1.0, "OOB score should not exceed 1.0");
    }
    
    @Test
    public void testOOBScoreWithoutBootstrap() {
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .bootstrap(false)
            .randomState(42)
            .build();
        
        rf.fit(simpleData, simpleLabels);
        double oobScore = rf.getOOBScore();
        
        assertTrue(Double.isNaN(oobScore), "OOB score should be NaN when bootstrap=false");
    }
    
    @Test
    public void testFeatureImportance() {
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(20)
            .randomState(42)
            .build();
        
        rf.fit(simpleData, simpleLabels);
        double[] importance = rf.getFeatureImportance();
        
        assertNotNull(importance, "Feature importance should not be null");
        assertEquals(2, importance.length, "Should have importance for 2 features");
        
        // Importance values should be non-negative
        for (double imp : importance) {
            assertTrue(imp >= 0.0, "Feature importance should be non-negative");
        }
        
        // Sum should be approximately 1.0 (normalized)
        double sum = importance[0] + importance[1];
        assertEquals(1.0, sum, 0.1, "Feature importances should sum to approximately 1.0");
    }
    
    @Test
    public void testDifferentNEstimators() {
        int[] estimatorCounts = {1, 5, 10, 50};
        
        for (int nEst : estimatorCounts) {
            RandomForestClassifier rf = new RandomForestClassifier.Builder()
                .nEstimators(nEst)
                .randomState(42)
                .build();
            
            rf.fit(simpleData, simpleLabels);
            
            assertEquals(nEst, rf.getNEstimators(), "Should have " + nEst + " estimators");
            
            int[] predictions = rf.predict(simpleData);
            assertNotNull(predictions, "Should make predictions with " + nEst + " estimators");
        }
    }
    
    @Test
    public void testMaxFeaturesSqrt() {
        double[][] X = new double[50][10];  // 10 features
        int[] y = new int[50];
        
        for (int i = 0; i < 25; i++) {
            for (int j = 0; j < 10; j++) {
                X[i][j] = Math.random();
            }
            y[i] = 0;
        }
        for (int i = 25; i < 50; i++) {
            for (int j = 0; j < 10; j++) {
                X[i][j] = Math.random() + 5.0;
            }
            y[i] = 1;
        }
        
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .maxFeatures("sqrt")
            .randomState(42)
            .build();
        
        rf.fit(X, y);
        int[] predictions = rf.predict(X);
        
        assertNotNull(predictions, "Should make predictions with maxFeatures='sqrt'");
        assertEquals(50, predictions.length, "Should predict all samples");
    }
    
    @Test
    public void testMaxFeaturesLog2() {
        double[][] X = new double[50][10];  // 10 features
        int[] y = new int[50];
        
        for (int i = 0; i < 25; i++) {
            for (int j = 0; j < 10; j++) {
                X[i][j] = Math.random();
            }
            y[i] = 0;
        }
        for (int i = 25; i < 50; i++) {
            for (int j = 0; j < 10; j++) {
                X[i][j] = Math.random() + 5.0;
            }
            y[i] = 1;
        }
        
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .maxFeatures("log2")
            .randomState(42)
            .build();
        
        rf.fit(X, y);
        int[] predictions = rf.predict(X);
        
        assertNotNull(predictions, "Should make predictions with maxFeatures='log2'");
    }
    
    @Test
    public void testMaxFeaturesInteger() {
        double[][] X = new double[50][10];  // 10 features
        int[] y = new int[50];
        
        for (int i = 0; i < 25; i++) {
            for (int j = 0; j < 10; j++) {
                X[i][j] = Math.random();
            }
            y[i] = 0;
        }
        for (int i = 25; i < 50; i++) {
            for (int j = 0; j < 10; j++) {
                X[i][j] = Math.random() + 5.0;
            }
            y[i] = 1;
        }
        
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .maxFeatures(5)
            .randomState(42)
            .build();
        
        rf.fit(X, y);
        int[] predictions = rf.predict(X);
        
        assertNotNull(predictions, "Should make predictions with maxFeatures=5");
    }
    
    @Test
    public void testDifferentCriteria() {
        // Test with GINI
        RandomForestClassifier rfGini = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .criterion(DecisionTreeClassifier.Criterion.GINI)
            .randomState(42)
            .build();
        
        rfGini.fit(simpleData, simpleLabels);
        int[] predictionsGini = rfGini.predict(simpleData);
        
        // Test with ENTROPY
        RandomForestClassifier rfEntropy = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .criterion(DecisionTreeClassifier.Criterion.ENTROPY)
            .randomState(42)
            .build();
        
        rfEntropy.fit(simpleData, simpleLabels);
        int[] predictionsEntropy = rfEntropy.predict(simpleData);
        
        // Both should work
        assertNotNull(predictionsGini, "GINI criterion should work");
        assertNotNull(predictionsEntropy, "ENTROPY criterion should work");
        
        // Both should correctly classify simple data
        assertArrayEquals(simpleLabels, predictionsGini, "GINI should perfectly classify");
        assertArrayEquals(simpleLabels, predictionsEntropy, "ENTROPY should perfectly classify");
    }
    
    @Test
    public void testMaxDepthConstraint() {
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .maxDepth(3)
            .randomState(42)
            .build();
        
        rf.fit(simpleData, simpleLabels);
        int[] predictions = rf.predict(simpleData);
        
        assertNotNull(predictions, "Should work with maxDepth constraint");
    }
    
    @Test
    public void testMinSamplesSplit() {
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .minSamplesSplit(3)
            .randomState(42)
            .build();
        
        rf.fit(simpleData, simpleLabels);
        int[] predictions = rf.predict(simpleData);
        
        assertNotNull(predictions, "Should work with minSamplesSplit constraint");
    }
    
    @Test
    public void testMinSamplesLeaf() {
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .minSamplesLeaf(2)
            .randomState(42)
            .build();
        
        rf.fit(simpleData, simpleLabels);
        int[] predictions = rf.predict(simpleData);
        
        assertNotNull(predictions, "Should work with minSamplesLeaf constraint");
    }
    
    @Test
    public void testReproducibility() {
        // Two models with same random state should give same results
        RandomForestClassifier rf1 = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .randomState(42)
            .build();
        
        RandomForestClassifier rf2 = new RandomForestClassifier.Builder()
            .nEstimators(10)
            .randomState(42)
            .build();
        
        rf1.fit(simpleData, simpleLabels);
        rf2.fit(simpleData, simpleLabels);
        
        int[] pred1 = rf1.predict(simpleData);
        int[] pred2 = rf2.predict(simpleData);
        
        assertArrayEquals(pred1, pred2, "Same random state should give same predictions");
    }
    
    @Test
    public void testDifferentRandomStates() {
        // Different random states may give different results
        RandomForestClassifier rf1 = new RandomForestClassifier.Builder()
            .nEstimators(5)
            .randomState(42)
            .build();
        
        RandomForestClassifier rf2 = new RandomForestClassifier.Builder()
            .nEstimators(5)
            .randomState(123)
            .build();
        
        rf1.fit(simpleData, simpleLabels);
        rf2.fit(simpleData, simpleLabels);
        
        // Both should still correctly classify (different paths, same result)
        int[] pred1 = rf1.predict(simpleData);
        int[] pred2 = rf2.predict(simpleData);
        
        assertArrayEquals(simpleLabels, pred1, "RF with seed 42 should classify correctly");
        assertArrayEquals(simpleLabels, pred2, "RF with seed 123 should classify correctly");
    }
    
    @Test
    public void testPredictBeforeFit() {
        RandomForestClassifier rf = new RandomForestClassifier.Builder().build();
        
        assertThrows(IllegalStateException.class, () -> {
            rf.predict(simpleData);
        }, "Should throw exception when predicting before fitting");
    }
    
    @Test
    public void testNullTrainingData() {
        RandomForestClassifier rf = new RandomForestClassifier.Builder().build();
        
        assertThrows(IllegalArgumentException.class, () -> {
            rf.fit(null, simpleLabels);
        }, "Should throw exception for null X");
        
        assertThrows(IllegalArgumentException.class, () -> {
            rf.fit(simpleData, null);
        }, "Should throw exception for null y");
    }
    
    @Test
    public void testEmptyTrainingData() {
        RandomForestClassifier rf = new RandomForestClassifier.Builder().build();
        
        assertThrows(IllegalArgumentException.class, () -> {
            rf.fit(new double[0][0], new int[0]);
        }, "Should throw exception for empty data");
    }
    
    @Test
    public void testMismatchedDataLength() {
        RandomForestClassifier rf = new RandomForestClassifier.Builder().build();
        
        assertThrows(IllegalArgumentException.class, () -> {
            rf.fit(simpleData, new int[]{0, 1});  // Wrong length
        }, "Should throw exception for mismatched X and y lengths");
    }
    
    @Test
    public void testInvalidNEstimators() {
        assertThrows(IllegalArgumentException.class, () -> {
            new RandomForestClassifier.Builder().nEstimators(0).build();
        }, "Should throw exception for nEstimators = 0");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new RandomForestClassifier.Builder().nEstimators(-5).build();
        }, "Should throw exception for negative nEstimators");
    }
    
    @Test
    public void testInvalidMaxFeatures() {
        assertThrows(IllegalArgumentException.class, () -> {
            new RandomForestClassifier.Builder().maxFeatures("invalid").build();
        }, "Should throw exception for invalid maxFeatures string");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new RandomForestClassifier.Builder().maxFeatures(0).build();
        }, "Should throw exception for maxFeatures = 0");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new RandomForestClassifier.Builder().maxFeatures(-1).build();
        }, "Should throw exception for negative maxFeatures");
    }
    
    @Test
    public void testInvalidHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> {
            new RandomForestClassifier.Builder().maxDepth(0).build();
        }, "Should throw exception for maxDepth = 0");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new RandomForestClassifier.Builder().minSamplesSplit(1).build();
        }, "Should throw exception for minSamplesSplit < 2");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new RandomForestClassifier.Builder().minSamplesLeaf(0).build();
        }, "Should throw exception for minSamplesLeaf < 1");
    }
    
    @Test
    public void testSingleTree() {
        // Random Forest with 1 tree should work like a single Decision Tree
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(1)
            .randomState(42)
            .build();
        
        rf.fit(simpleData, simpleLabels);
        int[] predictions = rf.predict(simpleData);
        
        assertArrayEquals(simpleLabels, predictions, "Single tree should classify correctly");
    }
    
    @Test
    public void testLargeEnsemble() {
        // Test with many trees
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(100)
            .randomState(42)
            .build();
        
        rf.fit(simpleData, simpleLabels);
        int[] predictions = rf.predict(simpleData);
        
        assertEquals(100, rf.getNEstimators(), "Should have 100 trees");
        assertArrayEquals(simpleLabels, predictions, "Large ensemble should classify correctly");
    }
}
