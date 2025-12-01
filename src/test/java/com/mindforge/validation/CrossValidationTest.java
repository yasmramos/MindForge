package com.mindforge.validation;

import com.mindforge.classification.KNearestNeighbors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.HashSet;
import java.util.Set;

/**
 * Tests for Cross-Validation utilities.
 */
public class CrossValidationTest {
    
    private double[][] X;
    private int[] y;
    private double[][] XSmall;
    private int[] ySmall;
    
    @BeforeEach
    public void setUp() {
        // Create a simple linearly separable dataset
        // Class 0: points with x1 < 0
        // Class 1: points with x1 >= 0
        X = new double[][] {
            {-2.0, 1.0}, {-1.5, 0.5}, {-1.0, 1.5}, {-0.5, 0.8}, {-0.3, 1.2},
            {2.0, 1.0}, {1.5, 0.5}, {1.0, 1.5}, {0.5, 0.8}, {0.3, 1.2},
            {-2.5, 0.3}, {-1.8, 1.0}, {-1.2, 0.7}, {-0.7, 1.3}, {-0.4, 0.9},
            {2.5, 0.3}, {1.8, 1.0}, {1.2, 0.7}, {0.7, 1.3}, {0.4, 0.9},
            {-2.2, 0.6}, {-1.6, 1.1}, {-1.1, 0.4}, {-0.6, 1.4}, {-0.2, 0.5},
            {2.2, 0.6}, {1.6, 1.1}, {1.1, 0.4}, {0.6, 1.4}, {0.2, 0.5}
        };
        
        y = new int[] {
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1
        };
        
        // Small dataset for LOOCV
        XSmall = new double[][] {
            {-1.0, 0.5}, {-0.8, 0.3}, {-0.6, 0.7},
            {1.0, 0.5}, {0.8, 0.3}, {0.6, 0.7}
        };
        
        ySmall = new int[] {0, 0, 0, 1, 1, 1};
    }
    
    @Test
    public void testKFoldBasic() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        CrossValidationResult result = CrossValidation.kFold(
            trainer, predictor, X, y, 5, 42
        );
        
        assertEquals(5, result.getNumFolds());
        assertEquals("accuracy", result.getMetricName());
        assertTrue(result.getMean() > 0.5); // Should be better than random
        assertTrue(result.getMean() <= 1.0);
        assertTrue(result.getStdDev() >= 0.0);
    }
    
    @Test
    public void testKFoldWithoutShuffle() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        CrossValidationResult result = CrossValidation.kFold(
            trainer, predictor, X, y, 3
        );
        
        assertEquals(3, result.getNumFolds());
        assertTrue(result.getMean() >= 0.0);
        assertTrue(result.getMean() <= 1.0);
    }
    
    @Test
    public void testKFoldReproducibility() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        CrossValidationResult result1 = CrossValidation.kFold(
            trainer, predictor, X, y, 5, 42
        );
        
        CrossValidationResult result2 = CrossValidation.kFold(
            trainer, predictor, X, y, 5, 42
        );
        
        assertArrayEquals(result1.getScores(), result2.getScores(), 1e-10);
    }
    
    @Test
    public void testKFoldDifferentK() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        CrossValidationResult result3 = CrossValidation.kFold(
            trainer, predictor, X, y, 3, 42
        );
        CrossValidationResult result5 = CrossValidation.kFold(
            trainer, predictor, X, y, 5, 42
        );
        CrossValidationResult result10 = CrossValidation.kFold(
            trainer, predictor, X, y, 10, 42
        );
        
        assertEquals(3, result3.getNumFolds());
        assertEquals(5, result5.getNumFolds());
        assertEquals(10, result10.getNumFolds());
        
        // All should have reasonable accuracy
        assertTrue(result3.getMean() > 0.5);
        assertTrue(result5.getMean() > 0.5);
        assertTrue(result10.getMean() > 0.5);
    }
    
    @Test
    public void testStratifiedKFold() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        CrossValidationResult result = CrossValidation.stratifiedKFold(
            trainer, predictor, X, y, 5, 42
        );
        
        assertEquals(5, result.getNumFolds());
        assertEquals("accuracy", result.getMetricName());
        assertTrue(result.getMean() > 0.5);
        assertTrue(result.getMean() <= 1.0);
    }
    
    @Test
    public void testStratifiedKFoldPreservesClassDistribution() {
        // Create imbalanced dataset
        double[][] XImbalanced = new double[40][];
        int[] yImbalanced = new int[40];
        
        // 30 samples of class 0, 10 samples of class 1
        for (int i = 0; i < 30; i++) {
            XImbalanced[i] = new double[]{-1.0 - i * 0.1, 0.5};
            yImbalanced[i] = 0;
        }
        for (int i = 30; i < 40; i++) {
            XImbalanced[i] = new double[]{1.0 + (i - 30) * 0.1, 0.5};
            yImbalanced[i] = 1;
        }
        
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        CrossValidationResult result = CrossValidation.stratifiedKFold(
            trainer, predictor, XImbalanced, yImbalanced, 5, 42
        );
        
        assertEquals(5, result.getNumFolds());
        assertTrue(result.getMean() > 0.5);
    }
    
    @Test
    public void testLeaveOneOut() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(1);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        CrossValidationResult result = CrossValidation.leaveOneOut(
            trainer, predictor, XSmall, ySmall
        );
        
        assertEquals(6, result.getNumFolds()); // One fold per sample
        assertEquals("accuracy", result.getMetricName());
        assertTrue(result.getMean() >= 0.0);
        assertTrue(result.getMean() <= 1.0);
    }
    
    @Test
    public void testLeaveOneOutPerfectClassifier() {
        // With k=1 and simple linearly separable data, should get high accuracy
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(1);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        CrossValidationResult result = CrossValidation.leaveOneOut(
            trainer, predictor, XSmall, ySmall
        );
        
        assertTrue(result.getMean() > 0.8);
    }
    
    @Test
    public void testShuffleSplit() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        CrossValidationResult result = CrossValidation.shuffleSplit(
            trainer, predictor, X, y, 10, 0.2, 42
        );
        
        assertEquals(10, result.getNumFolds());
        assertEquals("accuracy", result.getMetricName());
        assertTrue(result.getMean() > 0.5);
        assertTrue(result.getMean() <= 1.0);
    }
    
    @Test
    public void testShuffleSplitDifferentTestSizes() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        CrossValidationResult result1 = CrossValidation.shuffleSplit(
            trainer, predictor, X, y, 5, 0.2, 42
        );
        CrossValidationResult result2 = CrossValidation.shuffleSplit(
            trainer, predictor, X, y, 5, 0.3, 42
        );
        
        assertEquals(5, result1.getNumFolds());
        assertEquals(5, result2.getNumFolds());
        assertTrue(result1.getMean() > 0.5);
        assertTrue(result2.getMean() > 0.5);
    }
    
    @Test
    public void testTrainTestSplit() {
        CrossValidation.SplitData split = CrossValidation.trainTestSplit(
            X, y, 0.2, 42
        );
        
        int totalSamples = split.XTrain.length + split.XTest.length;
        assertEquals(X.length, totalSamples);
        assertEquals(split.XTrain.length, split.yTrain.length);
        assertEquals(split.XTest.length, split.yTest.length);
        
        // Test size should be approximately 20%
        double actualTestRatio = (double) split.XTest.length / X.length;
        assertTrue(Math.abs(actualTestRatio - 0.2) < 0.05);
    }
    
    @Test
    public void testTrainTestSplitReproducibility() {
        CrossValidation.SplitData split1 = CrossValidation.trainTestSplit(
            X, y, 0.2, 42
        );
        CrossValidation.SplitData split2 = CrossValidation.trainTestSplit(
            X, y, 0.2, 42
        );
        
        assertArrayEquals(split1.yTrain, split2.yTrain);
        assertArrayEquals(split1.yTest, split2.yTest);
    }
    
    @Test
    public void testTrainTestSplitNoOverlap() {
        CrossValidation.SplitData split = CrossValidation.trainTestSplit(
            X, y, 0.3, 42
        );
        
        // Verify no sample appears in both train and test
        Set<String> trainSamples = new HashSet<>();
        for (double[] sample : split.XTrain) {
            trainSamples.add(java.util.Arrays.toString(sample));
        }
        
        for (double[] sample : split.XTest) {
            String sampleStr = java.util.Arrays.toString(sample);
            assertFalse(trainSamples.contains(sampleStr), 
                "Sample should not appear in both train and test");
        }
    }
    
    @Test
    public void testCrossValidationResultStatistics() {
        double[] scores = {0.8, 0.85, 0.75, 0.9, 0.8};
        CrossValidationResult result = new CrossValidationResult(scores, "accuracy");
        
        assertEquals(5, result.getNumFolds());
        assertEquals(0.82, result.getMean(), 0.001);
        assertEquals(0.75, result.getMin(), 0.001);
        assertEquals(0.9, result.getMax(), 0.001);
        assertTrue(result.getStdDev() > 0.0);
    }
    
    @Test
    public void testCrossValidationResultToString() {
        double[] scores = {0.8, 0.85, 0.75};
        CrossValidationResult result = new CrossValidationResult(scores, "accuracy");
        
        String str = result.toString();
        assertTrue(str.contains("accuracy"));
        assertTrue(str.contains("folds=3"));
    }
    
    // Edge cases and error handling
    
    @Test
    public void testKFoldInvalidK() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.kFold(trainer, predictor, X, y, 1, 42);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.kFold(trainer, predictor, X, y, X.length + 1, 42);
        });
    }
    
    @Test
    public void testKFoldMismatchedArrays() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        int[] wrongY = new int[X.length - 1];
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.kFold(trainer, predictor, X, wrongY, 5, 42);
        });
    }
    
    @Test
    public void testStratifiedKFoldInvalidK() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.stratifiedKFold(trainer, predictor, X, y, 1, 42);
        });
    }
    
    @Test
    public void testShuffleSplitInvalidTestSize() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.shuffleSplit(trainer, predictor, X, y, 5, 0.0, 42);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.shuffleSplit(trainer, predictor, X, y, 5, 1.0, 42);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.shuffleSplit(trainer, predictor, X, y, 5, -0.1, 42);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.shuffleSplit(trainer, predictor, X, y, 5, 1.5, 42);
        });
    }
    
    @Test
    public void testShuffleSplitInvalidNSplits() {
        CrossValidation.ModelTrainer<KNearestNeighbors> trainer = (X, y) -> {
            KNearestNeighbors knn = new KNearestNeighbors(3);
            knn.train(X, y);
            return knn;
        };
        
        CrossValidation.ModelPredictor<KNearestNeighbors> predictor = 
            (model, X) -> model.predict(X);
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.shuffleSplit(trainer, predictor, X, y, 0, 0.2, 42);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.shuffleSplit(trainer, predictor, X, y, -1, 0.2, 42);
        });
    }
    
    @Test
    public void testTrainTestSplitInvalidTestSize() {
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.trainTestSplit(X, y, 0.0, 42);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.trainTestSplit(X, y, 1.0, 42);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.trainTestSplit(X, y, -0.1, 42);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.trainTestSplit(X, y, 1.5, 42);
        });
    }
}
