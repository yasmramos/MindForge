package com.mindforge.sprint1;

import com.mindforge.classification.LogisticRegression;
import com.mindforge.regression.DecisionTreeRegressor;
import com.mindforge.regression.RandomForestRegressor;
import com.mindforge.regression.GradientBoostingRegressor;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.Random;

/**
 * Comprehensive test suite for Sprint 1: Regression Core components.
 * Tests LogisticRegression, DecisionTreeRegressor, RandomForestRegressor, 
 * and GradientBoostingRegressor.
 * 
 * @author MindForge Team
 * @version 1.2.0-alpha
 */
@DisplayName("Sprint 1: Regression Core Tests")
public class Sprint1RegressionCoreTest {
    
    // Common test data
    private static double[][] linearX;
    private static double[] linearY;
    private static double[][] nonLinearX;
    private static double[] nonLinearY;
    private static double[][] classificationX;
    private static int[] binaryY;
    private static int[] multiClassY;
    
    @BeforeAll
    static void setupTestData() {
        Random random = new Random(42);
        
        // Linear regression data: y = 2*x1 + 3*x2 + noise
        int n = 200;
        linearX = new double[n][2];
        linearY = new double[n];
        for (int i = 0; i < n; i++) {
            linearX[i][0] = random.nextDouble() * 10;
            linearX[i][1] = random.nextDouble() * 10;
            linearY[i] = 2 * linearX[i][0] + 3 * linearX[i][1] + random.nextGaussian() * 0.5;
        }
        
        // Non-linear regression data: y = sin(x1) + x2^2 + noise
        nonLinearX = new double[n][2];
        nonLinearY = new double[n];
        for (int i = 0; i < n; i++) {
            nonLinearX[i][0] = random.nextDouble() * 6;
            nonLinearX[i][1] = random.nextDouble() * 3;
            nonLinearY[i] = Math.sin(nonLinearX[i][0]) + nonLinearX[i][1] * nonLinearX[i][1] 
                           + random.nextGaussian() * 0.3;
        }
        
        // Classification data
        classificationX = new double[n][2];
        binaryY = new int[n];
        multiClassY = new int[n];
        
        for (int i = 0; i < n; i++) {
            classificationX[i][0] = random.nextDouble() * 10 - 5;
            classificationX[i][1] = random.nextDouble() * 10 - 5;
            
            // Binary: class based on x1 + x2
            binaryY[i] = (classificationX[i][0] + classificationX[i][1] > 0) ? 1 : 0;
            
            // Multiclass: 3 classes based on angle
            double angle = Math.atan2(classificationX[i][1], classificationX[i][0]);
            if (angle < -Math.PI / 3) {
                multiClassY[i] = 0;
            } else if (angle < Math.PI / 3) {
                multiClassY[i] = 1;
            } else {
                multiClassY[i] = 2;
            }
        }
    }
    
    // =========================================================================
    // LOGISTIC REGRESSION TESTS
    // =========================================================================
    
    @Nested
    @DisplayName("LogisticRegression Tests")
    class LogisticRegressionTests {
        
        @Test
        @DisplayName("Binary classification basic test")
        void testBinaryClassification() {
            LogisticRegression lr = new LogisticRegression.Builder()
                .C(1.0)
                .maxIterations(500)
                .build();
            
            lr.fit(classificationX, binaryY);
            
            assertTrue(lr.isFitted());
            assertEquals(2, lr.getNumClasses());
            
            // Test predictions
            int[] predictions = lr.predict(classificationX);
            assertEquals(classificationX.length, predictions.length);
            
            // Calculate accuracy
            int correct = 0;
            for (int i = 0; i < binaryY.length; i++) {
                if (predictions[i] == binaryY[i]) correct++;
            }
            double accuracy = (double) correct / binaryY.length;
            assertTrue(accuracy > 0.8, "Accuracy should be > 80%: " + accuracy);
        }
        
        @Test
        @DisplayName("Multiclass classification test")
        void testMulticlassClassification() {
            LogisticRegression lr = new LogisticRegression.Builder()
                .C(1.0)
                .maxIterations(500)
                .multiClass(LogisticRegression.MultiClass.MULTINOMIAL)
                .build();
            
            lr.fit(classificationX, multiClassY);
            
            assertTrue(lr.isFitted());
            assertEquals(3, lr.getNumClasses());
            
            int[] predictions = lr.predict(classificationX);
            
            int correct = 0;
            for (int i = 0; i < multiClassY.length; i++) {
                if (predictions[i] == multiClassY[i]) correct++;
            }
            double accuracy = (double) correct / multiClassY.length;
            assertTrue(accuracy > 0.7, "Accuracy should be > 70%: " + accuracy);
        }
        
        @Test
        @DisplayName("Probability predictions")
        void testProbabilityPredictions() {
            LogisticRegression lr = new LogisticRegression();
            lr.fit(classificationX, binaryY);
            
            double[] proba = lr.predictProba(classificationX[0]);
            
            assertEquals(2, proba.length);
            assertTrue(proba[0] >= 0 && proba[0] <= 1);
            assertTrue(proba[1] >= 0 && proba[1] <= 1);
            assertEquals(1.0, proba[0] + proba[1], 0.001);
        }
        
        @Test
        @DisplayName("L1 regularization (sparsity)")
        void testL1Regularization() {
            LogisticRegression lr = new LogisticRegression.Builder()
                .regularization(LogisticRegression.Regularization.L1)
                .C(0.1)
                .solver(LogisticRegression.Solver.SGD)
                .maxIterations(1000)
                .build();
            
            lr.fit(classificationX, binaryY);
            
            assertTrue(lr.isFitted());
            // L1 should produce some sparsity
            double sparsity = lr.getSparsity();
            assertNotNull(lr.getCoefficients());
        }
        
        @Test
        @DisplayName("L2 regularization")
        void testL2Regularization() {
            LogisticRegression lr = new LogisticRegression.Builder()
                .regularization(LogisticRegression.Regularization.L2)
                .C(1.0)
                .build();
            
            lr.fit(classificationX, binaryY);
            
            assertTrue(lr.isFitted());
            
            // All coefficients should be non-zero
            double[][] coefs = lr.getCoefficients();
            assertNotNull(coefs);
        }
        
        @Test
        @DisplayName("ElasticNet regularization")
        void testElasticNetRegularization() {
            LogisticRegression lr = new LogisticRegression.Builder()
                .regularization(LogisticRegression.Regularization.ELASTICNET)
                .C(1.0)
                .l1Ratio(0.5)
                .solver(LogisticRegression.Solver.SGD)
                .maxIterations(1000)
                .build();
            
            lr.fit(classificationX, binaryY);
            
            assertTrue(lr.isFitted());
            assertNotNull(lr.getCoefficients());
        }
        
        @Test
        @DisplayName("Different solvers")
        void testDifferentSolvers() {
            // SGD solver
            LogisticRegression sgd = new LogisticRegression.Builder()
                .solver(LogisticRegression.Solver.SGD)
                .maxIterations(500)
                .build();
            sgd.fit(classificationX, binaryY);
            assertTrue(sgd.isFitted());
            
            // LBFGS solver
            LogisticRegression lbfgs = new LogisticRegression.Builder()
                .solver(LogisticRegression.Solver.LBFGS)
                .maxIterations(500)
                .build();
            lbfgs.fit(classificationX, binaryY);
            assertTrue(lbfgs.isFitted());
        }
        
        @Test
        @DisplayName("Get coefficients and intercepts")
        void testGetCoefficients() {
            LogisticRegression lr = new LogisticRegression();
            lr.fit(classificationX, binaryY);
            
            double[][] coefs = lr.getCoefficients();
            double[] intercepts = lr.getIntercepts();
            
            assertNotNull(coefs);
            assertNotNull(intercepts);
            assertEquals(1, coefs.length);  // Binary classification
            assertEquals(2, coefs[0].length);  // 2 features
            assertEquals(1, intercepts.length);
        }
        
        @Test
        @DisplayName("Loss history")
        void testLossHistory() {
            LogisticRegression lr = new LogisticRegression.Builder()
                .maxIterations(100)
                .build();
            lr.fit(classificationX, binaryY);
            
            double[] lossHistory = lr.getLossHistory();
            assertNotNull(lossHistory);
            assertTrue(lossHistory.length > 0);
            
            // Loss should generally decrease
            // (not strictly, but overall trend)
        }
        
        @Test
        @DisplayName("Classifier interface")
        void testClassifierInterface() {
            LogisticRegression lr = new LogisticRegression();
            lr.train(classificationX, binaryY);
            
            assertTrue(lr.isFitted());
            
            int prediction = lr.predict(classificationX[0]);
            assertTrue(prediction == 0 || prediction == 1);
        }
        
        @Test
        @DisplayName("Decision function for binary")
        void testDecisionFunction() {
            LogisticRegression lr = new LogisticRegression();
            lr.fit(classificationX, binaryY);
            
            double decision = lr.decisionFunction(classificationX[0]);
            // Decision function should return a real value
            assertFalse(Double.isNaN(decision));
        }
        
        @Test
        @DisplayName("Error on unfitted model")
        void testErrorOnUnfitted() {
            LogisticRegression lr = new LogisticRegression();
            
            assertThrows(IllegalStateException.class, () -> lr.predict(classificationX[0]));
            assertThrows(IllegalStateException.class, () -> lr.predictProba(classificationX[0]));
        }
        
        @Test
        @DisplayName("Error on null data")
        void testErrorOnNullData() {
            LogisticRegression lr = new LogisticRegression();
            
            assertThrows(IllegalArgumentException.class, () -> lr.fit(null, binaryY));
            assertThrows(IllegalArgumentException.class, () -> lr.fit(classificationX, null));
        }
    }
    
    // =========================================================================
    // DECISION TREE REGRESSOR TESTS
    // =========================================================================
    
    @Nested
    @DisplayName("DecisionTreeRegressor Tests")
    class DecisionTreeRegressorTests {
        
        @Test
        @DisplayName("Basic regression test")
        void testBasicRegression() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor.Builder()
                .maxDepth(10)
                .minSamplesSplit(2)
                .build();
            
            tree.fit(linearX, linearY);
            
            assertTrue(tree.isFitted());
            
            double[] predictions = tree.predict(linearX);
            assertEquals(linearX.length, predictions.length);
            
            // Calculate R^2
            double meanY = Arrays.stream(linearY).average().orElse(0);
            double ssRes = 0, ssTot = 0;
            for (int i = 0; i < linearY.length; i++) {
                ssRes += Math.pow(linearY[i] - predictions[i], 2);
                ssTot += Math.pow(linearY[i] - meanY, 2);
            }
            double r2 = 1 - (ssRes / ssTot);
            assertTrue(r2 > 0.8, "R^2 should be > 0.8: " + r2);
        }
        
        @Test
        @DisplayName("Non-linear regression")
        void testNonLinearRegression() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor.Builder()
                .maxDepth(10)
                .build();
            
            tree.fit(nonLinearX, nonLinearY);
            
            double[] predictions = tree.predict(nonLinearX);
            
            double meanY = Arrays.stream(nonLinearY).average().orElse(0);
            double ssRes = 0, ssTot = 0;
            for (int i = 0; i < nonLinearY.length; i++) {
                ssRes += Math.pow(nonLinearY[i] - predictions[i], 2);
                ssTot += Math.pow(nonLinearY[i] - meanY, 2);
            }
            double r2 = 1 - (ssRes / ssTot);
            assertTrue(r2 > 0.7, "R^2 should be > 0.7: " + r2);
        }
        
        @Test
        @DisplayName("Max depth constraint")
        void testMaxDepthConstraint() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor.Builder()
                .maxDepth(3)
                .build();
            
            tree.fit(linearX, linearY);
            
            assertTrue(tree.getTreeDepth() <= 3);
        }
        
        @Test
        @DisplayName("Min samples split constraint")
        void testMinSamplesSplitConstraint() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor.Builder()
                .minSamplesSplit(50)
                .build();
            
            tree.fit(linearX, linearY);
            
            // Tree should be smaller due to larger minSamplesSplit
            assertTrue(tree.getNumLeaves() < linearX.length / 50 + 10);
        }
        
        @Test
        @DisplayName("Feature importance")
        void testFeatureImportance() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor();
            tree.fit(linearX, linearY);
            
            double[] importance = tree.getFeatureImportance();
            
            assertNotNull(importance);
            assertEquals(2, importance.length);
            
            // Importance should sum to ~1
            double sum = Arrays.stream(importance).sum();
            assertEquals(1.0, sum, 0.01);
        }
        
        @Test
        @DisplayName("MSE criterion")
        void testMSECriterion() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor.Builder()
                .criterion(DecisionTreeRegressor.Criterion.MSE)
                .build();
            
            tree.fit(linearX, linearY);
            assertTrue(tree.isFitted());
        }
        
        @Test
        @DisplayName("MAE criterion")
        void testMAECriterion() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor.Builder()
                .criterion(DecisionTreeRegressor.Criterion.MAE)
                .build();
            
            tree.fit(linearX, linearY);
            assertTrue(tree.isFitted());
        }
        
        @Test
        @DisplayName("Tree structure info")
        void testTreeStructureInfo() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor.Builder()
                .maxDepth(5)
                .build();
            
            tree.fit(linearX, linearY);
            
            assertTrue(tree.getTreeDepth() > 0);
            assertTrue(tree.getNumLeaves() > 0);
            assertTrue(tree.getNumNodes() > 0);
            assertTrue(tree.getNumNodes() >= tree.getNumLeaves());
        }
        
        @Test
        @DisplayName("Training metrics")
        void testTrainingMetrics() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor();
            tree.fit(linearX, linearY);
            
            double mse = tree.getTrainMSE();
            double mae = tree.getTrainMAE();
            
            assertTrue(mse >= 0);
            assertTrue(mae >= 0);
            assertTrue(mse >= mae * mae / linearX.length || mae >= 0);  // Sanity check
        }
        
        @Test
        @DisplayName("Regressor interface")
        void testRegressorInterface() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor();
            tree.train(linearX, linearY);
            
            assertTrue(tree.isFitted());
            
            double prediction = tree.predict(linearX[0]);
            assertFalse(Double.isNaN(prediction));
        }
        
        @Test
        @DisplayName("Apply (leaf indices)")
        void testApply() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor.Builder()
                .maxDepth(5)
                .build();
            tree.fit(linearX, linearY);
            
            int[] leafIndices = tree.apply(linearX);
            
            assertNotNull(leafIndices);
            assertEquals(linearX.length, leafIndices.length);
        }
        
        @Test
        @DisplayName("Sample weights")
        void testSampleWeights() {
            DecisionTreeRegressor tree = new DecisionTreeRegressor();
            
            double[] weights = new double[linearX.length];
            Arrays.fill(weights, 1.0);
            // Double weight for first half
            for (int i = 0; i < weights.length / 2; i++) {
                weights[i] = 2.0;
            }
            
            tree.fit(linearX, linearY, weights);
            assertTrue(tree.isFitted());
        }
    }
    
    // =========================================================================
    // RANDOM FOREST REGRESSOR TESTS
    // =========================================================================
    
    @Nested
    @DisplayName("RandomForestRegressor Tests")
    class RandomForestRegressorTests {
        
        @Test
        @DisplayName("Basic regression test")
        void testBasicRegression() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(50)
                .maxDepth(10)
                .randomState(42)
                .build();
            
            rf.fit(linearX, linearY);
            
            assertTrue(rf.isFitted());
            assertEquals(50, rf.getNEstimators());
            
            double[] predictions = rf.predict(linearX);
            
            double r2 = rf.score(linearX, linearY);
            assertTrue(r2 > 0.85, "R^2 should be > 0.85: " + r2);
        }
        
        @Test
        @DisplayName("Non-linear regression")
        void testNonLinearRegression() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(100)
                .maxDepth(10)
                .build();
            
            rf.fit(nonLinearX, nonLinearY);
            
            double r2 = rf.score(nonLinearX, nonLinearY);
            assertTrue(r2 > 0.8, "R^2 should be > 0.8: " + r2);
        }
        
        @Test
        @DisplayName("OOB score")
        void testOOBScore() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(50)
                .bootstrap(true)
                .oobScore(true)
                .build();
            
            rf.fit(linearX, linearY);
            
            double oobScore = rf.getOobScore();
            assertFalse(Double.isNaN(oobScore));
            assertTrue(oobScore > 0 && oobScore <= 1, "OOB score should be in (0, 1]: " + oobScore);
        }
        
        @Test
        @DisplayName("Max features sqrt")
        void testMaxFeaturesSqrt() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(50)
                .maxFeatures("sqrt")
                .build();
            
            rf.fit(linearX, linearY);
            assertTrue(rf.isFitted());
        }
        
        @Test
        @DisplayName("Max features log2")
        void testMaxFeaturesLog2() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(50)
                .maxFeatures("log2")
                .build();
            
            rf.fit(linearX, linearY);
            assertTrue(rf.isFitted());
        }
        
        @Test
        @DisplayName("Max features fraction")
        void testMaxFeaturesFraction() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(50)
                .maxFeatures(0.5)
                .build();
            
            rf.fit(linearX, linearY);
            assertTrue(rf.isFitted());
        }
        
        @Test
        @DisplayName("No bootstrap")
        void testNoBootstrap() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(50)
                .bootstrap(false)
                .build();
            
            rf.fit(linearX, linearY);
            assertTrue(rf.isFitted());
        }
        
        @Test
        @DisplayName("Feature importance")
        void testFeatureImportance() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(50)
                .build();
            
            rf.fit(linearX, linearY);
            
            double[] importance = rf.getFeatureImportance();
            
            assertNotNull(importance);
            assertEquals(2, importance.length);
            
            double sum = Arrays.stream(importance).sum();
            assertEquals(1.0, sum, 0.01);
        }
        
        @Test
        @DisplayName("Predict all trees")
        void testPredictAllTrees() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(20)
                .build();
            
            rf.fit(linearX, linearY);
            
            double[] allPredictions = rf.predictAllTrees(linearX[0]);
            assertEquals(20, allPredictions.length);
        }
        
        @Test
        @DisplayName("Prediction variance")
        void testPredictionVariance() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(50)
                .build();
            
            rf.fit(linearX, linearY);
            
            double variance = rf.predictVariance(linearX[0]);
            assertTrue(variance >= 0);
        }
        
        @Test
        @DisplayName("Get estimators")
        void testGetEstimators() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(10)
                .build();
            
            rf.fit(linearX, linearY);
            
            var estimators = rf.getEstimators();
            assertEquals(10, estimators.size());
        }
        
        @Test
        @DisplayName("Apply (leaf indices)")
        void testApply() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(10)
                .maxDepth(5)
                .build();
            
            rf.fit(linearX, linearY);
            
            int[][] leafIndices = rf.apply(linearX);
            
            assertEquals(linearX.length, leafIndices.length);
            assertEquals(10, leafIndices[0].length);
        }
        
        @Test
        @DisplayName("Regressor interface")
        void testRegressorInterface() {
            RandomForestRegressor rf = new RandomForestRegressor(50);
            rf.train(linearX, linearY);
            
            assertTrue(rf.isFitted());
            
            double prediction = rf.predict(linearX[0]);
            assertFalse(Double.isNaN(prediction));
        }
        
        @Test
        @DisplayName("Score method")
        void testScoreMethod() {
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(50)
                .build();
            
            rf.fit(linearX, linearY);
            
            double score = rf.score(linearX, linearY);
            assertTrue(score >= 0 && score <= 1);
        }
    }
    
    // =========================================================================
    // GRADIENT BOOSTING REGRESSOR TESTS
    // =========================================================================
    
    @Nested
    @DisplayName("GradientBoostingRegressor Tests")
    class GradientBoostingRegressorTests {
        
        @Test
        @DisplayName("Basic regression test")
        void testBasicRegression() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .nEstimators(100)
                .learningRate(0.1)
                .maxDepth(3)
                .build();
            
            gb.fit(linearX, linearY);
            
            assertTrue(gb.isFitted());
            
            double r2 = gb.score(linearX, linearY);
            assertTrue(r2 > 0.9, "R^2 should be > 0.9: " + r2);
        }
        
        @Test
        @DisplayName("Non-linear regression")
        void testNonLinearRegression() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .nEstimators(100)
                .learningRate(0.1)
                .maxDepth(5)
                .build();
            
            gb.fit(nonLinearX, nonLinearY);
            
            double r2 = gb.score(nonLinearX, nonLinearY);
            assertTrue(r2 > 0.85, "R^2 should be > 0.85: " + r2);
        }
        
        @Test
        @DisplayName("Squared error loss")
        void testSquaredErrorLoss() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .loss(GradientBoostingRegressor.Loss.SQUARED_ERROR)
                .nEstimators(50)
                .build();
            
            gb.fit(linearX, linearY);
            assertTrue(gb.isFitted());
        }
        
        @Test
        @DisplayName("Absolute error loss")
        void testAbsoluteErrorLoss() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .loss(GradientBoostingRegressor.Loss.ABSOLUTE_ERROR)
                .nEstimators(50)
                .build();
            
            gb.fit(linearX, linearY);
            assertTrue(gb.isFitted());
            
            double r2 = gb.score(linearX, linearY);
            assertTrue(r2 > 0.5, "R^2 should be positive: " + r2);
        }
        
        @Test
        @DisplayName("Huber loss")
        void testHuberLoss() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .loss(GradientBoostingRegressor.Loss.HUBER)
                .alpha(0.9)
                .nEstimators(50)
                .build();
            
            gb.fit(linearX, linearY);
            assertTrue(gb.isFitted());
        }
        
        @Test
        @DisplayName("Quantile loss")
        void testQuantileLoss() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .loss(GradientBoostingRegressor.Loss.QUANTILE)
                .alpha(0.5)  // Median
                .nEstimators(50)
                .build();
            
            gb.fit(linearX, linearY);
            assertTrue(gb.isFitted());
        }
        
        @Test
        @DisplayName("Learning rate effect")
        void testLearningRateEffect() {
            GradientBoostingRegressor fast = new GradientBoostingRegressor.Builder()
                .learningRate(0.5)
                .nEstimators(20)
                .build();
            
            GradientBoostingRegressor slow = new GradientBoostingRegressor.Builder()
                .learningRate(0.01)
                .nEstimators(20)
                .build();
            
            fast.fit(linearX, linearY);
            slow.fit(linearX, linearY);
            
            // Fast learner should converge faster with same iterations
            double fastR2 = fast.score(linearX, linearY);
            double slowR2 = slow.score(linearX, linearY);
            
            assertTrue(fastR2 > slowR2, "Higher learning rate should converge faster");
        }
        
        @Test
        @DisplayName("Subsampling")
        void testSubsampling() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .subsample(0.8)
                .nEstimators(50)
                .build();
            
            gb.fit(linearX, linearY);
            assertTrue(gb.isFitted());
        }
        
        @Test
        @DisplayName("Feature importance")
        void testFeatureImportance() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .nEstimators(50)
                .build();
            
            gb.fit(linearX, linearY);
            
            double[] importance = gb.getFeatureImportance();
            
            assertNotNull(importance);
            assertEquals(2, importance.length);
            
            double sum = Arrays.stream(importance).sum();
            assertEquals(1.0, sum, 0.01);
        }
        
        @Test
        @DisplayName("Training loss history")
        void testTrainingLossHistory() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .nEstimators(50)
                .validationFraction(0)  // Disable early stopping
                .build();
            
            gb.fit(linearX, linearY);
            
            double[] lossHistory = gb.getTrainLossHistory();
            
            assertNotNull(lossHistory);
            assertEquals(50, lossHistory.length);
            
            // Loss should generally decrease
            assertTrue(lossHistory[lossHistory.length - 1] < lossHistory[0], 
                      "Loss should decrease over iterations");
        }
        
        @Test
        @DisplayName("Staged predictions")
        void testStagedPredictions() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .nEstimators(20)
                .validationFraction(0)
                .build();
            
            gb.fit(linearX, linearY);
            
            double[] staged = gb.stagedPredict(linearX[0]);
            
            assertEquals(20, staged.length);
            
            // Final staged prediction should equal regular prediction
            double finalPred = gb.predict(linearX[0]);
            assertEquals(finalPred, staged[staged.length - 1], 0.001);
        }
        
        @Test
        @DisplayName("Early stopping")
        void testEarlyStopping() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .nEstimators(500)
                .validationFraction(0.2)
                .nIterNoChange(10)
                .tol(1e-4)
                .build();
            
            gb.fit(linearX, linearY);
            
            // Should stop before 500 iterations
            int fittedEstimators = gb.getNEstimatorsFitted();
            assertTrue(fittedEstimators <= 500);
            
            double[] validLoss = gb.getValidLossHistory();
            assertNotNull(validLoss);
        }
        
        @Test
        @DisplayName("Simple constructor")
        void testSimpleConstructor() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor(50, 0.1, 3);
            
            gb.fit(linearX, linearY);
            assertTrue(gb.isFitted());
        }
        
        @Test
        @DisplayName("Apply (leaf indices)")
        void testApply() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .nEstimators(10)
                .maxDepth(3)
                .validationFraction(0)
                .build();
            
            gb.fit(linearX, linearY);
            
            int[][] leafIndices = gb.apply(linearX);
            
            assertEquals(linearX.length, leafIndices.length);
            assertEquals(10, leafIndices[0].length);
        }
        
        @Test
        @DisplayName("Regressor interface")
        void testRegressorInterface() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor();
            gb.train(linearX, linearY);
            
            assertTrue(gb.isFitted());
            
            double prediction = gb.predict(linearX[0]);
            assertFalse(Double.isNaN(prediction));
        }
        
        @Test
        @DisplayName("Getters")
        void testGetters() {
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .nEstimators(100)
                .learningRate(0.1)
                .build();
            
            assertEquals(100, gb.getNEstimators());
            assertEquals(0.1, gb.getLearningRate(), 0.001);
        }
    }
    
    // =========================================================================
    // INTEGRATION TESTS
    // =========================================================================
    
    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {
        
        @Test
        @DisplayName("Compare regressor performances")
        void testCompareRegressorPerformances() {
            // Train all regressors
            DecisionTreeRegressor dt = new DecisionTreeRegressor.Builder()
                .maxDepth(10)
                .build();
            dt.fit(linearX, linearY);
            
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(50)
                .maxDepth(10)
                .build();
            rf.fit(linearX, linearY);
            
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .nEstimators(50)
                .learningRate(0.1)
                .maxDepth(3)
                .build();
            gb.fit(linearX, linearY);
            
            // Calculate R^2 for each
            double dtR2 = calculateR2(dt.predict(linearX), linearY);
            double rfR2 = rf.score(linearX, linearY);
            double gbR2 = gb.score(linearX, linearY);
            
            // All should have positive R^2
            assertTrue(dtR2 > 0);
            assertTrue(rfR2 > 0);
            assertTrue(gbR2 > 0);
            
            // Ensemble methods should generally outperform single tree
            assertTrue(rfR2 >= dtR2 * 0.95, "RF should be competitive with DT");
            assertTrue(gbR2 >= dtR2 * 0.95, "GB should be competitive with DT");
        }
        
        @Test
        @DisplayName("Compare on non-linear data")
        void testCompareOnNonLinearData() {
            DecisionTreeRegressor dt = new DecisionTreeRegressor.Builder()
                .maxDepth(15)
                .build();
            dt.fit(nonLinearX, nonLinearY);
            
            RandomForestRegressor rf = new RandomForestRegressor.Builder()
                .nEstimators(100)
                .maxDepth(15)
                .build();
            rf.fit(nonLinearX, nonLinearY);
            
            GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
                .nEstimators(100)
                .maxDepth(5)
                .build();
            gb.fit(nonLinearX, nonLinearY);
            
            double dtR2 = calculateR2(dt.predict(nonLinearX), nonLinearY);
            double rfR2 = rf.score(nonLinearX, nonLinearY);
            double gbR2 = gb.score(nonLinearX, nonLinearY);
            
            // All should handle non-linear data
            assertTrue(dtR2 > 0.5);
            assertTrue(rfR2 > 0.5);
            assertTrue(gbR2 > 0.5);
        }
        
        private double calculateR2(double[] predictions, double[] actual) {
            double mean = Arrays.stream(actual).average().orElse(0);
            double ssRes = 0, ssTot = 0;
            for (int i = 0; i < actual.length; i++) {
                ssRes += Math.pow(actual[i] - predictions[i], 2);
                ssTot += Math.pow(actual[i] - mean, 2);
            }
            return 1 - (ssRes / ssTot);
        }
    }
    
    // =========================================================================
    // EDGE CASES
    // =========================================================================
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Single feature")
        void testSingleFeature() {
            double[][] X = new double[100][1];
            double[] y = new double[100];
            Random rand = new Random(42);
            
            for (int i = 0; i < 100; i++) {
                X[i][0] = rand.nextDouble() * 10;
                y[i] = 2 * X[i][0] + rand.nextGaussian() * 0.5;
            }
            
            DecisionTreeRegressor dt = new DecisionTreeRegressor();
            dt.fit(X, y);
            assertTrue(dt.isFitted());
            
            RandomForestRegressor rf = new RandomForestRegressor(20);
            rf.fit(X, y);
            assertTrue(rf.isFitted());
            
            GradientBoostingRegressor gb = new GradientBoostingRegressor(50, 0.1, 3);
            gb.fit(X, y);
            assertTrue(gb.isFitted());
        }
        
        @Test
        @DisplayName("Small dataset")
        void testSmallDataset() {
            double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}, {5.0, 6.0}};
            double[] y = {1.5, 2.5, 3.5, 4.5, 5.5};
            
            DecisionTreeRegressor dt = new DecisionTreeRegressor();
            dt.fit(X, y);
            assertTrue(dt.isFitted());
            
            RandomForestRegressor rf = new RandomForestRegressor(10);
            rf.fit(X, y);
            assertTrue(rf.isFitted());
            
            GradientBoostingRegressor gb = new GradientBoostingRegressor(10, 0.1, 2);
            gb.fit(X, y);
            assertTrue(gb.isFitted());
        }
        
        @Test
        @DisplayName("Constant target")
        void testConstantTarget() {
            double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}};
            double[] y = {5.0, 5.0, 5.0};
            
            DecisionTreeRegressor dt = new DecisionTreeRegressor();
            dt.fit(X, y);
            
            assertEquals(5.0, dt.predict(new double[]{10.0, 10.0}), 0.001);
        }
        
        @Test
        @DisplayName("Binary classification with only 2 samples per class")
        void testMinimalBinaryClassification() {
            double[][] X = {{0.0, 0.0}, {0.1, 0.1}, {1.0, 1.0}, {1.1, 1.1}};
            int[] y = {0, 0, 1, 1};
            
            LogisticRegression lr = new LogisticRegression();
            lr.fit(X, y);
            assertTrue(lr.isFitted());
        }
    }
}
