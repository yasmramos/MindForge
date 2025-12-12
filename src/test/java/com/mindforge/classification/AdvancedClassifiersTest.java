package com.mindforge.classification;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for advanced classifiers: Stacking, ExtraTrees, QDA.
 */
class AdvancedClassifiersTest {
    
    private double[][] X;
    private int[] y;
    
    @BeforeEach
    void setUp() {
        // Simple separable data
        X = new double[][]{
            {0, 0}, {0, 1}, {1, 0}, {1, 1},
            {5, 5}, {5, 6}, {6, 5}, {6, 6},
            {0, 5}, {1, 5}, {0, 6}, {1, 6}
        };
        y = new int[]{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    }
    
    @Nested
    @DisplayName("StackingClassifier Tests")
    class StackingClassifierTests {
        
        @Test
        @DisplayName("Default constructor works")
        void testDefaultConstructor() {
            StackingClassifier stacking = new StackingClassifier();
            assertNotNull(stacking);
            assertFalse(stacking.isTrained());
        }
        
        @Test
        @DisplayName("Add classifiers and train")
        void testAddClassifiersAndTrain() {
            StackingClassifier stacking = new StackingClassifier()
                .addClassifier(new KNearestNeighbors(1))
                .addClassifier(new DecisionTreeClassifier());
            
            stacking.train(X, y);
            
            assertTrue(stacking.isTrained());
            assertEquals(2, stacking.getNumBaseClassifiers());
        }
        
        @Test
        @DisplayName("Predict after training")
        void testPredict() {
            StackingClassifier stacking = new StackingClassifier()
                .addClassifier(new KNearestNeighbors(1))
                .addClassifier(new GaussianNaiveBayes());
            
            stacking.train(X, y);
            
            int[] predictions = stacking.predict(X);
            assertEquals(X.length, predictions.length);
        }
        
        @Test
        @DisplayName("Builder pattern works")
        void testBuilder() {
            StackingClassifier stacking = new StackingClassifier.Builder()
                .addClassifier(new KNearestNeighbors(1))
                .addClassifier(new DecisionTreeClassifier())
                .cvFolds(3)
                .build();
            
            assertNotNull(stacking);
        }
        
        @Test
        @DisplayName("No classifiers throws exception")
        void testNoClassifiers() {
            StackingClassifier stacking = new StackingClassifier();
            assertThrows(IllegalStateException.class, () -> stacking.train(X, y));
        }
    }
    
    @Nested
    @DisplayName("ExtraTreesClassifier Tests")
    class ExtraTreesClassifierTests {
        
        @Test
        @DisplayName("Default constructor works")
        void testDefaultConstructor() {
            ExtraTreesClassifier et = new ExtraTreesClassifier();
            assertNotNull(et);
            assertEquals(100, et.getNEstimators());
        }
        
        @Test
        @DisplayName("Train and predict")
        void testTrainAndPredict() {
            ExtraTreesClassifier et = new ExtraTreesClassifier(10);
            et.train(X, y);
            
            assertTrue(et.isTrained());
            int[] predictions = et.predict(X);
            assertEquals(X.length, predictions.length);
        }
        
        @Test
        @DisplayName("Predict probabilities")
        void testPredictProba() {
            ExtraTreesClassifier et = new ExtraTreesClassifier(10);
            et.train(X, y);
            
            double[] proba = et.predictProba(X[0]);
            assertEquals(3, proba.length);
            
            double sum = 0;
            for (double p : proba) sum += p;
            assertEquals(1.0, sum, 0.01);
        }
        
        @Test
        @DisplayName("Feature importances")
        void testFeatureImportances() {
            ExtraTreesClassifier et = new ExtraTreesClassifier(10);
            et.train(X, y);
            
            double[] importances = et.getFeatureImportances();
            assertEquals(2, importances.length);
        }
        
        @Test
        @DisplayName("Builder pattern")
        void testBuilder() {
            ExtraTreesClassifier et = new ExtraTreesClassifier.Builder()
                .nEstimators(20)
                .maxDepth(5)
                .randomState(42)
                .build();
            
            et.train(X, y);
            assertTrue(et.isTrained());
        }
    }
    
    @Nested
    @DisplayName("QuadraticDiscriminantAnalysis Tests")
    class QDATests {
        
        @Test
        @DisplayName("Default constructor works")
        void testDefaultConstructor() {
            QuadraticDiscriminantAnalysis qda = new QuadraticDiscriminantAnalysis();
            assertNotNull(qda);
            assertEquals(0.0, qda.getRegParam());
        }
        
        @Test
        @DisplayName("Train and predict")
        void testTrainAndPredict() {
            QuadraticDiscriminantAnalysis qda = new QuadraticDiscriminantAnalysis(0.1);
            qda.train(X, y);
            
            assertTrue(qda.isTrained());
            int[] predictions = qda.predict(X);
            assertEquals(X.length, predictions.length);
        }
        
        @Test
        @DisplayName("Predict probabilities")
        void testPredictProba() {
            QuadraticDiscriminantAnalysis qda = new QuadraticDiscriminantAnalysis(0.1);
            qda.train(X, y);
            
            double[] proba = qda.predictProba(X[0]);
            assertEquals(3, proba.length);
            
            double sum = 0;
            for (double p : proba) sum += p;
            assertEquals(1.0, sum, 0.01);
        }
        
        @Test
        @DisplayName("Gets priors and means")
        void testGetPriorsAndMeans() {
            QuadraticDiscriminantAnalysis qda = new QuadraticDiscriminantAnalysis();
            qda.train(X, y);
            
            double[] priors = qda.getPriors();
            double[][] means = qda.getMeans();
            
            assertNotNull(priors);
            assertNotNull(means);
            assertEquals(3, priors.length);
            assertEquals(3, means.length);
        }
        
        @Test
        @DisplayName("Score computes accuracy")
        void testScore() {
            QuadraticDiscriminantAnalysis qda = new QuadraticDiscriminantAnalysis(0.1);
            qda.train(X, y);
            
            double score = qda.score(X, y);
            assertTrue(score > 0.5);
        }
    }
    
    @Nested
    @DisplayName("LinearDiscriminantAnalysis Tests")
    class LDATests {
        
        @Test
        @DisplayName("Builder works")
        void testBuilder() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder().build();
            assertNotNull(lda);
        }
        
        @Test
        @DisplayName("Train and predict")
        void testTrainAndPredict() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder().build();
            lda.train(X, y);
            
            assertTrue(lda.isTrained());
            int[] predictions = lda.predict(X);
            assertEquals(X.length, predictions.length);
        }
        
        @Test
        @DisplayName("Predict probabilities")
        void testPredictProba() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder().build();
            lda.train(X, y);
            
            double[] proba = lda.predictProba(X[0]);
            assertEquals(3, proba.length);
            
            double sum = 0;
            for (double p : proba) sum += p;
            assertEquals(1.0, sum, 0.01);
        }
        
        @Test
        @DisplayName("Score computes accuracy")
        void testScore() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder().build();
            lda.train(X, y);
            
            double score = lda.score(X, y);
            assertTrue(score > 0.5);
        }
    }
}
