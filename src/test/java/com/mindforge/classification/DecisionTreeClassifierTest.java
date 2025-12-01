package com.mindforge.classification;

import com.mindforge.validation.Metrics;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DecisionTreeClassifierTest {
    
    @Test
    void testSimpleClassification() {
        // Simple 2D dataset with two clear clusters
        double[][] X = {
            {1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0},  // Class 0
            {6.0, 5.0}, {7.0, 8.0}, {8.0, 7.0}   // Class 1
        };
        int[] y = {0, 0, 0, 1, 1, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        tree.train(X, y);
        
        // Test prediction
        int prediction = tree.predict(new double[]{2.5, 2.5});
        assertEquals(0, prediction, "Point close to class 0 should be classified as 0");
        
        prediction = tree.predict(new double[]{7.0, 7.0});
        assertEquals(1, prediction, "Point close to class 1 should be classified as 1");
    }
    
    @Test
    void testPerfectAccuracy() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}};
        int[] y = {0, 0, 0, 1, 1, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        tree.train(X, y);
        
        int[] predictions = tree.predict(X);
        double accuracy = Metrics.accuracy(y, predictions);
        assertEquals(1.0, accuracy, 0.001, "Should have perfect accuracy on linearly separable data");
    }
    
    @Test
    void testNumClasses() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}};
        int[] y = {0, 1, 2, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        tree.train(X, y);
        
        assertEquals(3, tree.getNumClasses(), "Should detect 3 classes");
    }
    
    @Test
    void testGiniCriterion() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}};
        int[] y = {0, 0, 0, 1, 1, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
            .criterion(DecisionTreeClassifier.Criterion.GINI)
            .build();
        tree.train(X, y);
        
        int[] predictions = tree.predict(X);
        double accuracy = Metrics.accuracy(y, predictions);
        assertEquals(1.0, accuracy, 0.001, "Gini criterion should achieve perfect accuracy");
    }
    
    @Test
    void testEntropyCriterion() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}};
        int[] y = {0, 0, 0, 1, 1, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
            .criterion(DecisionTreeClassifier.Criterion.ENTROPY)
            .build();
        tree.train(X, y);
        
        int[] predictions = tree.predict(X);
        double accuracy = Metrics.accuracy(y, predictions);
        assertEquals(1.0, accuracy, 0.001, "Entropy criterion should achieve perfect accuracy");
    }
    
    @Test
    void testMaxDepthConstraint() {
        // Create a dataset that would need depth > 2 to perfectly separate
        double[][] X = {
            {1.0, 1.0}, {1.0, 2.0}, {2.0, 1.0}, {2.0, 2.0},
            {5.0, 5.0}, {5.0, 6.0}, {6.0, 5.0}, {6.0, 6.0}
        };
        int[] y = {0, 0, 0, 0, 1, 1, 1, 1};
        
        DecisionTreeClassifier shallowTree = new DecisionTreeClassifier.Builder()
            .maxDepth(1)
            .build();
        shallowTree.train(X, y);
        
        assertTrue(shallowTree.getTreeDepth() <= 1, "Tree depth should respect maxDepth constraint");
        assertEquals(1, shallowTree.getTreeDepth(), "Tree should have depth 1 with maxDepth=1");
    }
    
    @Test
    void testMinSamplesSplit() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}};
        int[] y = {0, 0, 0, 1, 1, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
            .minSamplesSplit(10)  // More than total samples, should create only root
            .build();
        tree.train(X, y);
        
        assertEquals(1, tree.getNumLeaves(), "Should have only 1 leaf when minSamplesSplit is too large");
    }
    
    @Test
    void testMinSamplesLeaf() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}};
        int[] y = {0, 0, 0, 1, 1, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
            .minSamplesLeaf(3)
            .build();
        tree.train(X, y);
        
        // With minSamplesLeaf=3 and 6 samples split evenly, should work
        assertTrue(tree.isFitted(), "Tree should be fitted successfully");
    }
    
    @Test
    void testPredictProba() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}};
        int[] y = {0, 0, 0, 1, 1, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        tree.train(X, y);
        
        double[] proba = tree.predictProba(new double[]{1.5});
        
        assertEquals(2, proba.length, "Should return probabilities for 2 classes");
        assertTrue(proba[0] >= 0 && proba[0] <= 1, "Probabilities should be in [0, 1]");
        assertTrue(proba[1] >= 0 && proba[1] <= 1, "Probabilities should be in [0, 1]");
        assertEquals(1.0, proba[0] + proba[1], 0.001, "Probabilities should sum to 1");
    }
    
    @Test
    void testPredictProbaBatch() {
        double[][] X = {{1.0}, {2.0}, {5.0}, {6.0}};
        int[] y = {0, 0, 1, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        tree.train(X, y);
        
        double[][] probas = tree.predictProba(X);
        
        assertEquals(X.length, probas.length, "Should return probabilities for all samples");
        for (double[] proba : probas) {
            assertEquals(2, proba.length, "Each sample should have probabilities for 2 classes");
            assertEquals(1.0, proba[0] + proba[1], 0.001, "Probabilities should sum to 1");
        }
    }
    
    @Test
    void testMulticlassClassification() {
        double[][] X = {
            {1.0, 1.0}, {1.5, 1.5}, {2.0, 2.0},  // Class 0
            {5.0, 5.0}, {5.5, 5.5}, {6.0, 6.0},  // Class 1
            {10.0, 1.0}, {10.5, 1.5}, {11.0, 2.0}  // Class 2
        };
        int[] y = {0, 0, 0, 1, 1, 1, 2, 2, 2};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        tree.train(X, y);
        
        assertEquals(3, tree.getNumClasses(), "Should detect 3 classes");
        
        int[] predictions = tree.predict(X);
        double accuracy = Metrics.accuracy(y, predictions);
        assertEquals(1.0, accuracy, 0.001, "Should perfectly classify 3-class problem");
    }
    
    @Test
    void testXORProblem() {
        // XOR is a classic non-linearly separable problem
        double[][] X = {
            {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
        };
        int[] y = {0, 1, 1, 0};  // XOR pattern
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
            .maxDepth(3)
            .build();
        tree.train(X, y);
        
        int[] predictions = tree.predict(X);
        double accuracy = Metrics.accuracy(y, predictions);
        assertEquals(1.0, accuracy, 0.001, "Decision tree should solve XOR problem");
    }
    
    @Test
    void testTreeDepth() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}};
        int[] y = {0, 0, 0, 1, 1, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
            .maxDepth(3)
            .build();
        tree.train(X, y);
        
        int actualDepth = tree.getTreeDepth();
        assertTrue(actualDepth >= 1, "Tree depth should be at least 1");
        assertTrue(actualDepth <= 3, "Tree depth should not exceed maxDepth");
    }
    
    @Test
    void testNumLeaves() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}};
        int[] y = {0, 0, 1, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        tree.train(X, y);
        
        int numLeaves = tree.getNumLeaves();
        assertTrue(numLeaves >= 2, "Should have at least 2 leaves for binary classification");
    }
    
    @Test
    void testNotFittedError() {
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        assertFalse(tree.isFitted(), "Tree should not be fitted initially");
        
        assertThrows(IllegalStateException.class, () -> {
            tree.predict(new double[]{1.0});
        }, "Should throw exception when predicting before training");
        
        assertThrows(IllegalStateException.class, () -> {
            tree.predictProba(new double[]{1.0});
        }, "Should throw exception when getting probabilities before training");
    }
    
    @Test
    void testInvalidInput() {
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        
        assertThrows(IllegalArgumentException.class, () -> {
            tree.train(null, new int[]{0, 1});
        }, "Should throw exception for null X");
        
        assertThrows(IllegalArgumentException.class, () -> {
            tree.train(new double[][]{{1.0}}, null);
        }, "Should throw exception for null y");
        
        assertThrows(IllegalArgumentException.class, () -> {
            tree.train(new double[][]{{1.0}, {2.0}}, new int[]{0});
        }, "Should throw exception when X and y have different lengths");
    }
    
    @Test
    void testInvalidHyperparameters() {
        assertThrows(IllegalArgumentException.class, () -> {
            new DecisionTreeClassifier.Builder().maxDepth(0).build();
        }, "Should throw exception for maxDepth < 1");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new DecisionTreeClassifier.Builder().minSamplesSplit(1).build();
        }, "Should throw exception for minSamplesSplit < 2");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new DecisionTreeClassifier.Builder().minSamplesLeaf(0).build();
        }, "Should throw exception for minSamplesLeaf < 1");
    }
    
    @Test
    void testWrongFeatureCount() {
        double[][] X = {{1.0, 2.0}, {3.0, 4.0}};
        int[] y = {0, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        tree.train(X, y);
        
        assertThrows(IllegalArgumentException.class, () -> {
            tree.predict(new double[]{1.0});  // Only 1 feature instead of 2
        }, "Should throw exception for wrong number of features");
    }
    
    @Test
    void testSingleClassDataset() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}};
        int[] y = {0, 0, 0, 0};  // All same class
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        tree.train(X, y);
        
        int prediction = tree.predict(new double[]{5.0});
        assertEquals(0, prediction, "Should predict the only class");
        
        assertEquals(1, tree.getNumLeaves(), "Should have only 1 leaf for single-class data");
    }
    
    @Test
    void testBuilder() {
        DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
            .maxDepth(5)
            .minSamplesSplit(4)
            .minSamplesLeaf(2)
            .criterion(DecisionTreeClassifier.Criterion.ENTROPY)
            .build();
        
        assertNotNull(tree, "Builder should create valid tree");
        assertEquals(5, tree.getMaxDepth(), "MaxDepth should be set correctly");
    }
    
    @Test
    void testToString() {
        DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
            .maxDepth(10)
            .minSamplesSplit(5)
            .criterion(DecisionTreeClassifier.Criterion.GINI)
            .build();
        
        String str = tree.toString();
        assertTrue(str.contains("maxDepth=10"), "toString should include maxDepth");
        assertTrue(str.contains("minSamplesSplit=5"), "toString should include minSamplesSplit");
        assertTrue(str.contains("GINI"), "toString should include criterion");
    }
    
    @Test
    void testTreeInfo() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}};
        int[] y = {0, 0, 1, 1};
        
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        
        String infoBefore = tree.getTreeInfo();
        assertTrue(infoBefore.contains("not fitted"), "Info should indicate tree is not fitted");
        
        tree.train(X, y);
        
        String infoAfter = tree.getTreeInfo();
        assertTrue(infoAfter.contains("depth="), "Info should include depth");
        assertTrue(infoAfter.contains("leaves="), "Info should include number of leaves");
        assertTrue(infoAfter.contains("samples="), "Info should include number of samples");
    }
    
    @Test
    void testDeterministicBehavior() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}};
        int[] y = {0, 0, 0, 1, 1, 1};
        
        DecisionTreeClassifier tree1 = new DecisionTreeClassifier();
        tree1.train(X, y);
        int[] predictions1 = tree1.predict(X);
        
        DecisionTreeClassifier tree2 = new DecisionTreeClassifier();
        tree2.train(X, y);
        int[] predictions2 = tree2.predict(X);
        
        assertArrayEquals(predictions1, predictions2, "Same data should produce same predictions");
    }
}
