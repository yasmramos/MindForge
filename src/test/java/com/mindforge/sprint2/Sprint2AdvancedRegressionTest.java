package com.mindforge.sprint2;

import com.mindforge.regression.SVR;
import com.mindforge.regression.GaussianProcessRegressor;
import com.mindforge.classification.LinearDiscriminantAnalysis;
import com.mindforge.classification.Kernel;
import com.mindforge.preprocessing.RobustScaler;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.*;

import java.util.*;
import java.util.stream.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Sprint 2 - Advanced Regression components.
 * 
 * Components tested:
 * 1. SVR (Support Vector Regression)
 * 2. LinearDiscriminantAnalysis
 * 3. GaussianProcessRegressor
 * 4. RobustScaler
 * 
 * @author MindForge Team
 * @since 2.0.0
 */
@DisplayName("Sprint 2: Advanced Regression & LDA Tests")
public class Sprint2AdvancedRegressionTest {
    
    private static final double TOLERANCE = 1e-6;
    private static final double PREDICTION_TOLERANCE = 0.3; // For regression tasks
    
    // Sample datasets
    private double[][] linearRegressionX;
    private double[] linearRegressionY;
    private double[][] classificationX;
    private int[] classificationY;
    private double[][] dataWithOutliers;
    
    @BeforeEach
    void setUp() {
        // Create linear regression dataset
        Random random = new Random(42);
        int n = 100;
        linearRegressionX = new double[n][2];
        linearRegressionY = new double[n];
        
        for (int i = 0; i < n; i++) {
            linearRegressionX[i][0] = random.nextDouble() * 10;
            linearRegressionX[i][1] = random.nextDouble() * 10;
            linearRegressionY[i] = 2 * linearRegressionX[i][0] + 
                                   3 * linearRegressionX[i][1] + 
                                   random.nextGaussian() * 0.5;
        }
        
        // Create classification dataset (3 classes)
        classificationX = new double[150][4];
        classificationY = new int[150];
        
        for (int i = 0; i < 50; i++) {
            // Class 0
            classificationX[i][0] = random.nextGaussian() + 0;
            classificationX[i][1] = random.nextGaussian() + 0;
            classificationX[i][2] = random.nextGaussian() + 0;
            classificationX[i][3] = random.nextGaussian() + 0;
            classificationY[i] = 0;
            
            // Class 1
            classificationX[i + 50][0] = random.nextGaussian() + 3;
            classificationX[i + 50][1] = random.nextGaussian() + 3;
            classificationX[i + 50][2] = random.nextGaussian() + 3;
            classificationX[i + 50][3] = random.nextGaussian() + 3;
            classificationY[i + 50] = 1;
            
            // Class 2
            classificationX[i + 100][0] = random.nextGaussian() + 6;
            classificationX[i + 100][1] = random.nextGaussian() + 6;
            classificationX[i + 100][2] = random.nextGaussian() + 6;
            classificationX[i + 100][3] = random.nextGaussian() + 6;
            classificationY[i + 100] = 2;
        }
        
        // Create data with outliers
        dataWithOutliers = new double[105][2];
        for (int i = 0; i < 100; i++) {
            dataWithOutliers[i][0] = random.nextGaussian() * 2 + 10;
            dataWithOutliers[i][1] = random.nextGaussian() * 3 + 20;
        }
        // Add outliers
        dataWithOutliers[100] = new double[] {100.0, 200.0};
        dataWithOutliers[101] = new double[] {-50.0, -100.0};
        dataWithOutliers[102] = new double[] {150.0, 300.0};
        dataWithOutliers[103] = new double[] {-80.0, -150.0};
        dataWithOutliers[104] = new double[] {200.0, 400.0};
    }
    
    // =========================================================================
    // SVR Tests
    // =========================================================================
    
    @Nested
    @DisplayName("SVR Tests")
    class SVRTests {
        
        @Test
        @DisplayName("Linear SVR - Basic Training and Prediction")
        void testLinearSVRBasic() {
            SVR svr = new SVR.Builder()
                .kernel(Kernel.Type.LINEAR)
                .C(1.0)
                .epsilon(0.1)
                .build();
            
            assertFalse(svr.isFitted());
            
            svr.train(linearRegressionX, linearRegressionY);
            
            assertTrue(svr.isFitted());
            assertNotNull(svr.getWeights());
            assertEquals(2, svr.getWeights().length);
        }
        
        @Test
        @DisplayName("Linear SVR - Predictions")
        void testLinearSVRPredictions() {
            SVR svr = new SVR.Builder()
                .kernel(Kernel.Type.LINEAR)
                .C(10.0)
                .epsilon(0.1)
                .maxIter(2000)
                .build();
            
            svr.train(linearRegressionX, linearRegressionY);
            
            // Test single prediction
            double[] testPoint = new double[] {5.0, 5.0};
            double prediction = svr.predict(testPoint);
            double expected = 2 * 5.0 + 3 * 5.0; // 25
            
            // Should be within reasonable range
            assertTrue(Math.abs(prediction - expected) < 5.0,
                "Prediction " + prediction + " should be close to " + expected);
            
            // Test batch prediction
            double[] predictions = svr.predict(linearRegressionX);
            assertEquals(linearRegressionX.length, predictions.length);
        }
        
        @Test
        @DisplayName("RBF Kernel SVR")
        void testRBFKernelSVR() {
            SVR svr = new SVR.Builder()
                .kernel(Kernel.Type.RBF)
                .gamma(0.5)
                .C(1.0)
                .epsilon(0.1)
                .build();
            
            svr.train(linearRegressionX, linearRegressionY);
            
            assertTrue(svr.isFitted());
            assertTrue(svr.usesKernelMethod());
            assertTrue(svr.getNumSupportVectors() > 0);
        }
        
        @Test
        @DisplayName("Polynomial Kernel SVR")
        void testPolynomialKernelSVR() {
            SVR svr = new SVR.Builder()
                .kernel(Kernel.Type.POLYNOMIAL)
                .degree(2)
                .gamma(1.0)
                .coef0(0.0)
                .C(1.0)
                .build();
            
            svr.train(linearRegressionX, linearRegressionY);
            
            assertTrue(svr.isFitted());
            assertTrue(svr.usesKernelMethod());
        }
        
        @Test
        @DisplayName("SVR - String Kernel Selection")
        void testSVRStringKernel() {
            SVR svrRbf = new SVR.Builder()
                .kernel("rbf")
                .build();
            
            SVR svrPoly = new SVR.Builder()
                .kernel("polynomial")
                .build();
            
            SVR svrLinear = new SVR.Builder()
                .kernel("linear")
                .build();
            
            SVR svrSigmoid = new SVR.Builder()
                .kernel("sigmoid")
                .build();
            
            assertNotNull(svrRbf);
            assertNotNull(svrPoly);
            assertNotNull(svrLinear);
            assertNotNull(svrSigmoid);
        }
        
        @Test
        @DisplayName("SVR - Score Method")
        void testSVRScore() {
            SVR svr = new SVR.Builder()
                .kernel(Kernel.Type.LINEAR)
                .C(10.0)
                .epsilon(0.1)
                .maxIter(2000)
                .build();
            
            svr.train(linearRegressionX, linearRegressionY);
            
            double score = svr.score(linearRegressionX, linearRegressionY);
            
            // R^2 should be positive for a reasonable fit
            assertTrue(score > 0, "R^2 score should be positive: " + score);
        }
        
        @Test
        @DisplayName("SVR - Parameter Getters")
        void testSVRParameters() {
            SVR svr = new SVR.Builder()
                .C(5.0)
                .epsilon(0.2)
                .kernel(Kernel.Type.RBF)
                .build();
            
            assertEquals(5.0, svr.getC(), TOLERANCE);
            assertEquals(0.2, svr.getEpsilon(), TOLERANCE);
            assertEquals(Kernel.Type.RBF, svr.getKernel().getType());
        }
        
        @Test
        @DisplayName("SVR - Builder Validation")
        void testSVRBuilderValidation() {
            assertThrows(IllegalArgumentException.class, () -> 
                new SVR.Builder().C(-1.0).build()
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                new SVR.Builder().epsilon(-0.1).build()
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                new SVR.Builder().maxIter(0).build()
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                new SVR.Builder().kernel("invalid").build()
            );
        }
        
        @Test
        @DisplayName("SVR - Input Validation")
        void testSVRInputValidation() {
            SVR svr = new SVR.Builder().build();
            
            assertThrows(IllegalArgumentException.class, () -> 
                svr.train(null, linearRegressionY)
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                svr.train(linearRegressionX, null)
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                svr.train(new double[0][0], new double[0])
            );
            
            // Predict before training
            assertThrows(IllegalStateException.class, () -> 
                svr.predict(new double[] {1.0, 2.0})
            );
        }
        
        @Test
        @DisplayName("SVR - Support Vectors for Kernel Method")
        void testSVRSupportVectors() {
            SVR svr = new SVR.Builder()
                .kernel(Kernel.Type.RBF)
                .C(1.0)
                .build();
            
            svr.train(linearRegressionX, linearRegressionY);
            
            double[][] supportVectors = svr.getSupportVectors();
            assertNotNull(supportVectors);
            assertTrue(supportVectors.length > 0);
            assertTrue(supportVectors.length <= linearRegressionX.length);
        }
        
        @Test
        @DisplayName("Linear SVR - Weights Not Available for Kernel Method")
        void testSVRWeightsForKernel() {
            SVR svr = new SVR.Builder()
                .kernel(Kernel.Type.RBF)
                .build();
            
            svr.train(linearRegressionX, linearRegressionY);
            
            assertThrows(IllegalStateException.class, svr::getWeights);
        }
    }
    
    // =========================================================================
    // LinearDiscriminantAnalysis Tests
    // =========================================================================
    
    @Nested
    @DisplayName("LinearDiscriminantAnalysis Tests")
    class LDATests {
        
        @Test
        @DisplayName("LDA - Basic Training and Prediction")
        void testLDABasic() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .solver(LinearDiscriminantAnalysis.Solver.SVD)
                .build();
            
            assertFalse(lda.isFitted());
            
            lda.train(classificationX, classificationY);
            
            assertTrue(lda.isFitted());
            assertEquals(3, lda.getNumClasses());
        }
        
        @Test
        @DisplayName("LDA - Classification Accuracy")
        void testLDAClassification() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            lda.train(classificationX, classificationY);
            
            int[] predictions = lda.predict(classificationX);
            
            // Count correct predictions
            int correct = 0;
            for (int i = 0; i < predictions.length; i++) {
                if (predictions[i] == classificationY[i]) {
                    correct++;
                }
            }
            
            double accuracy = (double) correct / predictions.length;
            assertTrue(accuracy > 0.7, "Accuracy should be above 70%: " + accuracy);
        }
        
        @Test
        @DisplayName("LDA - Score Method")
        void testLDAScore() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            lda.train(classificationX, classificationY);
            
            double score = lda.score(classificationX, classificationY);
            
            assertTrue(score >= 0 && score <= 1, "Score should be between 0 and 1");
            assertTrue(score > 0.7, "Score should be above 0.7: " + score);
        }
        
        @Test
        @DisplayName("LDA - Dimensionality Reduction (Transform)")
        void testLDATransform() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .nComponents(2)
                .build();
            
            lda.fit(classificationX, classificationY);
            
            double[][] transformed = lda.transform(classificationX);
            
            assertEquals(classificationX.length, transformed.length);
            assertEquals(2, transformed[0].length); // 2 components
        }
        
        @Test
        @DisplayName("LDA - Fit Transform")
        void testLDAFitTransform() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            double[][] transformed = lda.fitTransform(classificationX, classificationY);
            
            assertTrue(lda.isFitted());
            assertEquals(classificationX.length, transformed.length);
            // Default: min(n_classes - 1, n_features) = min(2, 4) = 2
            assertEquals(2, transformed[0].length);
        }
        
        @Test
        @DisplayName("LDA - Probability Predictions")
        void testLDAProbabilities() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            lda.train(classificationX, classificationY);
            
            double[] probs = lda.predictProba(classificationX[0]);
            
            assertEquals(3, probs.length);
            
            // Probabilities should sum to 1
            double sum = 0;
            for (double p : probs) {
                assertTrue(p >= 0 && p <= 1, "Probability should be in [0, 1]");
                sum += p;
            }
            assertEquals(1.0, sum, 0.01);
        }
        
        @Test
        @DisplayName("LDA - Class Means")
        void testLDAClassMeans() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            lda.fit(classificationX, classificationY);
            
            double[][] classMeans = lda.getClassMeans();
            
            assertEquals(3, classMeans.length);
            assertEquals(4, classMeans[0].length);
        }
        
        @Test
        @DisplayName("LDA - Explained Variance Ratio")
        void testLDAExplainedVariance() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            lda.fit(classificationX, classificationY);
            
            double[] evr = lda.getExplainedVarianceRatio();
            
            assertNotNull(evr);
            assertTrue(evr.length > 0);
            
            // First component should explain more variance
            if (evr.length > 1) {
                assertTrue(evr[0] >= evr[1], 
                    "First component should explain more variance");
            }
        }
        
        @Test
        @DisplayName("LDA - Custom Priors")
        void testLDACustomPriors() {
            double[] customPriors = {0.5, 0.3, 0.2};
            
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .priors(customPriors)
                .build();
            
            lda.fit(classificationX, classificationY);
            
            double[] priors = lda.getPriors();
            assertArrayEquals(customPriors, priors, TOLERANCE);
        }
        
        @Test
        @DisplayName("LDA - Shrinkage Regularization")
        void testLDAShrinkage() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .shrinkage(0.5)
                .build();
            
            lda.fit(classificationX, classificationY);
            
            assertTrue(lda.isFitted());
            assertTrue(lda.score(classificationX, classificationY) > 0);
        }
        
        @Test
        @DisplayName("LDA - EIGEN Solver")
        void testLDAEigenSolver() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .solver(LinearDiscriminantAnalysis.Solver.EIGEN)
                .build();
            
            lda.fit(classificationX, classificationY);
            
            assertTrue(lda.isFitted());
            
            int[] predictions = lda.predict(classificationX);
            assertEquals(classificationX.length, predictions.length);
        }
        
        @Test
        @DisplayName("LDA - Builder Validation")
        void testLDABuilderValidation() {
            assertThrows(IllegalArgumentException.class, () -> 
                new LinearDiscriminantAnalysis.Builder().nComponents(0).build()
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                new LinearDiscriminantAnalysis.Builder().shrinkage(-0.1).build()
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                new LinearDiscriminantAnalysis.Builder().shrinkage(1.5).build()
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                new LinearDiscriminantAnalysis.Builder()
                    .priors(new double[] {0.5, 0.4}) // Doesn't sum to 1
                    .build()
            );
        }
        
        @Test
        @DisplayName("LDA - Input Validation")
        void testLDAInputValidation() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            assertThrows(IllegalArgumentException.class, () -> 
                lda.train(null, classificationY)
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                lda.train(classificationX, null)
            );
            
            // Predict before training
            assertThrows(IllegalStateException.class, () -> 
                lda.predict(classificationX[0])
            );
        }
        
        @Test
        @DisplayName("LDA - Scalings Matrix")
        void testLDAScalings() {
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            lda.fit(classificationX, classificationY);
            
            double[][] scalings = lda.getScalings();
            
            assertNotNull(scalings);
            assertEquals(4, scalings.length); // n_features
            assertEquals(2, scalings[0].length); // n_components
        }
    }
    
    // =========================================================================
    // GaussianProcessRegressor Tests
    // =========================================================================
    
    @Nested
    @DisplayName("GaussianProcessRegressor Tests")
    class GPRTests {
        
        @Test
        @DisplayName("GPR - Basic Training and Prediction")
        void testGPRBasic() {
            GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
                .kernel(GaussianProcessRegressor.KernelType.RBF)
                .lengthScale(1.0)
                .build();
            
            assertFalse(gpr.isFitted());
            
            // Use smaller dataset for GPR (O(n^3) complexity)
            double[][] X = Arrays.copyOf(linearRegressionX, 30);
            double[] y = Arrays.copyOf(linearRegressionY, 30);
            
            gpr.train(X, y);
            
            assertTrue(gpr.isFitted());
            assertEquals(30, gpr.getNumTrainingSamples());
        }
        
        @Test
        @DisplayName("GPR - Prediction with Uncertainty")
        void testGPRPredictWithStd() {
            GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
                .kernel(GaussianProcessRegressor.KernelType.RBF)
                .lengthScale(1.0)
                .variance(1.0)
                .alpha(1e-5)
                .build();
            
            double[][] X = Arrays.copyOf(linearRegressionX, 30);
            double[] y = Arrays.copyOf(linearRegressionY, 30);
            
            gpr.train(X, y);
            
            double[][] testX = {{5.0, 5.0}, {2.0, 3.0}};
            double[][] result = gpr.predictWithStd(testX);
            
            assertEquals(2, result.length); // [means, stds]
            assertEquals(2, result[0].length); // 2 test points
            assertEquals(2, result[1].length);
            
            // Standard deviations should be non-negative
            for (double std : result[1]) {
                assertTrue(std >= 0, "Standard deviation should be non-negative");
            }
        }
        
        @Test
        @DisplayName("GPR - Different Kernels")
        void testGPRDifferentKernels() {
            double[][] X = Arrays.copyOf(linearRegressionX, 20);
            double[] y = Arrays.copyOf(linearRegressionY, 20);
            
            for (GaussianProcessRegressor.KernelType kernel : 
                    GaussianProcessRegressor.KernelType.values()) {
                
                GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
                    .kernel(kernel)
                    .lengthScale(1.0)
                    .build();
                
                gpr.train(X, y);
                
                assertTrue(gpr.isFitted(), 
                    "GPR should be fitted with " + kernel + " kernel");
                
                double prediction = gpr.predict(X[0]);
                assertFalse(Double.isNaN(prediction), 
                    "Prediction should not be NaN with " + kernel + " kernel");
            }
        }
        
        @Test
        @DisplayName("GPR - String Kernel Selection")
        void testGPRStringKernel() {
            GaussianProcessRegressor gprRbf = new GaussianProcessRegressor.Builder()
                .kernel("rbf")
                .build();
            
            GaussianProcessRegressor gprMatern = new GaussianProcessRegressor.Builder()
                .kernel("matern52")
                .build();
            
            GaussianProcessRegressor gprLinear = new GaussianProcessRegressor.Builder()
                .kernel("linear")
                .build();
            
            assertNotNull(gprRbf);
            assertNotNull(gprMatern);
            assertNotNull(gprLinear);
        }
        
        @Test
        @DisplayName("GPR - Log Marginal Likelihood")
        void testGPRLogMarginalLikelihood() {
            GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
                .kernel(GaussianProcessRegressor.KernelType.RBF)
                .build();
            
            double[][] X = Arrays.copyOf(linearRegressionX, 20);
            double[] y = Arrays.copyOf(linearRegressionY, 20);
            
            gpr.train(X, y);
            
            double logML = gpr.getLogMarginalLikelihood();
            
            assertFalse(Double.isNaN(logML), "Log marginal likelihood should not be NaN");
            assertFalse(Double.isInfinite(logML), 
                "Log marginal likelihood should not be infinite");
        }
        
        @Test
        @DisplayName("GPR - Sampling from Posterior")
        void testGPRSampling() {
            GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
                .kernel(GaussianProcessRegressor.KernelType.RBF)
                .lengthScale(1.0)
                .build();
            
            double[][] X = Arrays.copyOf(linearRegressionX, 20);
            double[] y = Arrays.copyOf(linearRegressionY, 20);
            
            gpr.train(X, y);
            
            double[][] testX = {{5.0, 5.0}, {2.0, 3.0}, {7.0, 8.0}};
            double[][] samples = gpr.sample(testX, 5);
            
            assertEquals(5, samples.length); // 5 samples
            assertEquals(3, samples[0].length); // 3 test points
        }
        
        @Test
        @DisplayName("GPR - Score Method")
        void testGPRScore() {
            GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
                .kernel(GaussianProcessRegressor.KernelType.RBF)
                .lengthScale(2.0)
                .build();
            
            double[][] X = Arrays.copyOf(linearRegressionX, 30);
            double[] y = Arrays.copyOf(linearRegressionY, 30);
            
            gpr.train(X, y);
            
            double score = gpr.score(X, y);
            
            // Training score should be high for GPR
            assertTrue(score > 0, "R^2 score should be positive");
        }
        
        @Test
        @DisplayName("GPR - Normalize Y Option")
        void testGPRNormalizeY() {
            GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
                .normalizeY(true)
                .build();
            
            double[][] X = Arrays.copyOf(linearRegressionX, 20);
            double[] y = Arrays.copyOf(linearRegressionY, 20);
            
            gpr.train(X, y);
            
            double prediction = gpr.predict(X[0]);
            
            // Prediction should be in similar range as original y
            assertTrue(prediction > -100 && prediction < 200,
                "Prediction should be in reasonable range");
        }
        
        @Test
        @DisplayName("GPR - Parameter Getters")
        void testGPRParameters() {
            GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
                .kernel(GaussianProcessRegressor.KernelType.MATERN_52)
                .lengthScale(2.5)
                .variance(1.5)
                .alpha(1e-8)
                .build();
            
            assertEquals(GaussianProcessRegressor.KernelType.MATERN_52, 
                gpr.getKernelType());
            assertEquals(2.5, gpr.getLengthScale(), TOLERANCE);
            assertEquals(1.5, gpr.getVariance(), TOLERANCE);
            assertEquals(1e-8, gpr.getAlpha(), TOLERANCE);
        }
        
        @Test
        @DisplayName("GPR - Builder Validation")
        void testGPRBuilderValidation() {
            assertThrows(IllegalArgumentException.class, () -> 
                new GaussianProcessRegressor.Builder().lengthScale(0).build()
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                new GaussianProcessRegressor.Builder().variance(-1).build()
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                new GaussianProcessRegressor.Builder().alpha(-0.1).build()
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                new GaussianProcessRegressor.Builder().kernel("invalid").build()
            );
        }
        
        @Test
        @DisplayName("GPR - Input Validation")
        void testGPRInputValidation() {
            GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
                .build();
            
            assertThrows(IllegalArgumentException.class, () -> 
                gpr.train(null, linearRegressionY)
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                gpr.train(linearRegressionX, null)
            );
            
            assertThrows(IllegalStateException.class, () -> 
                gpr.predict(new double[] {1.0, 2.0})
            );
        }
    }
    
    // =========================================================================
    // RobustScaler Tests
    // =========================================================================
    
    @Nested
    @DisplayName("RobustScaler Tests")
    class RobustScalerTests {
        
        @Test
        @DisplayName("RobustScaler - Basic Fit and Transform")
        void testRobustScalerBasic() {
            RobustScaler scaler = new RobustScaler.Builder().build();
            
            assertFalse(scaler.isFitted());
            
            double[][] scaled = scaler.fitTransform(dataWithOutliers);
            
            assertTrue(scaler.isFitted());
            assertEquals(dataWithOutliers.length, scaled.length);
            assertEquals(dataWithOutliers[0].length, scaled[0].length);
        }
        
        @Test
        @DisplayName("RobustScaler - Median Centering")
        void testRobustScalerMedian() {
            RobustScaler scaler = new RobustScaler.Builder()
                .withCentering(true)
                .withScaling(false)
                .build();
            
            // Simple dataset
            double[][] data = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {100.0}};
            
            scaler.fit(data);
            
            double[] center = scaler.getCenter();
            // Median of {1, 2, 3, 4, 5, 100} should be (3 + 4) / 2 = 3.5
            assertEquals(3.5, center[0], 0.1);
        }
        
        @Test
        @DisplayName("RobustScaler - IQR Scaling")
        void testRobustScalerIQR() {
            RobustScaler scaler = new RobustScaler.Builder()
                .withCentering(false)
                .withScaling(true)
                .build();
            
            // Create data where IQR is known
            double[][] data = new double[100][1];
            for (int i = 0; i < 100; i++) {
                data[i][0] = i;
            }
            
            scaler.fit(data);
            
            double[] scale = scaler.getScale();
            // IQR of 0-99: Q3 (75th percentile) - Q1 (25th percentile)
            // Q1 ≈ 24.25, Q3 ≈ 74.25, IQR ≈ 50
            assertTrue(Math.abs(scale[0] - 50) < 2, "IQR should be approximately 50");
        }
        
        @Test
        @DisplayName("RobustScaler - Robustness to Outliers")
        void testRobustScalerRobustness() {
            RobustScaler robustScaler = new RobustScaler.Builder().build();
            
            // Compare with data with and without outliers
            double[][] dataClean = Arrays.copyOf(dataWithOutliers, 100);
            
            RobustScaler scalerClean = new RobustScaler.Builder().build();
            scalerClean.fit(dataClean);
            
            robustScaler.fit(dataWithOutliers);
            
            // Centers should be similar (robust to outliers)
            double[] centerClean = scalerClean.getCenter();
            double[] centerWithOutliers = robustScaler.getCenter();
            
            // Should be relatively close despite outliers
            for (int i = 0; i < centerClean.length; i++) {
                assertTrue(Math.abs(centerClean[i] - centerWithOutliers[i]) < 5,
                    "Centers should be similar despite outliers");
            }
        }
        
        @Test
        @DisplayName("RobustScaler - Custom Quantile Range")
        void testRobustScalerCustomQuantiles() {
            RobustScaler scaler = new RobustScaler.Builder()
                .quantileRange(10.0, 90.0)
                .build();
            
            double[] range = scaler.getQuantileRange();
            assertEquals(10.0, range[0], TOLERANCE);
            assertEquals(90.0, range[1], TOLERANCE);
            
            double[][] scaled = scaler.fitTransform(dataWithOutliers);
            assertNotNull(scaled);
        }
        
        @Test
        @DisplayName("RobustScaler - Inverse Transform")
        void testRobustScalerInverseTransform() {
            RobustScaler scaler = new RobustScaler.Builder().build();
            
            double[][] scaled = scaler.fitTransform(dataWithOutliers);
            double[][] recovered = scaler.inverseTransform(scaled);
            
            // Check reconstruction
            for (int i = 0; i < dataWithOutliers.length; i++) {
                for (int j = 0; j < dataWithOutliers[0].length; j++) {
                    assertEquals(dataWithOutliers[i][j], recovered[i][j], 0.001,
                        "Inverse transform should recover original data");
                }
            }
        }
        
        @Test
        @DisplayName("RobustScaler - Single Sample Transform")
        void testRobustScalerSingleSample() {
            RobustScaler scaler = new RobustScaler.Builder().build();
            scaler.fit(dataWithOutliers);
            
            double[] sample = {10.0, 20.0};
            double[] scaled = scaler.transform(sample);
            
            assertEquals(2, scaled.length);
            
            double[] recovered = scaler.inverseTransform(scaled);
            assertArrayEquals(sample, recovered, 0.001);
        }
        
        @Test
        @DisplayName("RobustScaler - Only Centering")
        void testRobustScalerOnlyCentering() {
            RobustScaler scaler = new RobustScaler.Builder()
                .withCentering(true)
                .withScaling(false)
                .build();
            
            assertTrue(scaler.isWithCentering());
            assertFalse(scaler.isWithScaling());
            
            double[][] scaled = scaler.fitTransform(dataWithOutliers);
            
            // Scale should be all 1s
            double[] scale = scaler.getScale();
            for (double s : scale) {
                assertEquals(1.0, s, TOLERANCE);
            }
        }
        
        @Test
        @DisplayName("RobustScaler - Only Scaling")
        void testRobustScalerOnlyScaling() {
            RobustScaler scaler = new RobustScaler.Builder()
                .withCentering(false)
                .withScaling(true)
                .build();
            
            assertFalse(scaler.isWithCentering());
            assertTrue(scaler.isWithScaling());
            
            double[][] scaled = scaler.fitTransform(dataWithOutliers);
            
            // Center should be all 0s
            double[] center = scaler.getCenter();
            for (double c : center) {
                assertEquals(0.0, c, TOLERANCE);
            }
        }
        
        @Test
        @DisplayName("RobustScaler - Unit Variance")
        void testRobustScalerUnitVariance() {
            RobustScaler scaler = new RobustScaler.Builder()
                .unitVariance(true)
                .build();
            
            // Create normally distributed data
            Random random = new Random(42);
            double[][] normalData = new double[1000][1];
            for (int i = 0; i < 1000; i++) {
                normalData[i][0] = random.nextGaussian() * 5 + 10;
            }
            
            double[][] scaled = scaler.fitTransform(normalData);
            
            // Compute variance of scaled data
            double mean = 0;
            for (double[] row : scaled) {
                mean += row[0];
            }
            mean /= scaled.length;
            
            double variance = 0;
            for (double[] row : scaled) {
                variance += (row[0] - mean) * (row[0] - mean);
            }
            variance /= scaled.length;
            
            // Variance should be approximately 1 for normal data
            assertTrue(Math.abs(variance - 1.0) < 0.3,
                "Variance should be approximately 1: " + variance);
        }
        
        @Test
        @DisplayName("RobustScaler - Default Constructor")
        void testRobustScalerDefaultConstructor() {
            RobustScaler scaler = new RobustScaler();
            
            assertTrue(scaler.isWithCentering());
            assertTrue(scaler.isWithScaling());
            
            double[] range = scaler.getQuantileRange();
            assertEquals(25.0, range[0], TOLERANCE);
            assertEquals(75.0, range[1], TOLERANCE);
        }
        
        @Test
        @DisplayName("RobustScaler - Constructor with Options")
        void testRobustScalerConstructorWithOptions() {
            RobustScaler scaler = new RobustScaler(false, true);
            
            assertFalse(scaler.isWithCentering());
            assertTrue(scaler.isWithScaling());
        }
        
        @Test
        @DisplayName("RobustScaler - Builder Validation")
        void testRobustScalerBuilderValidation() {
            assertThrows(IllegalArgumentException.class, () -> 
                new RobustScaler.Builder().quantileRange(-10, 75).build()
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                new RobustScaler.Builder().quantileRange(25, 150).build()
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                new RobustScaler.Builder().quantileRange(75, 25).build()
            );
        }
        
        @Test
        @DisplayName("RobustScaler - Input Validation")
        void testRobustScalerInputValidation() {
            RobustScaler scaler = new RobustScaler();
            
            assertThrows(IllegalArgumentException.class, () -> 
                scaler.fit(null)
            );
            
            assertThrows(IllegalArgumentException.class, () -> 
                scaler.fit(new double[0][0])
            );
            
            assertThrows(IllegalStateException.class, () -> 
                scaler.transform(dataWithOutliers)
            );
            
            // After fitting, wrong number of features
            scaler.fit(dataWithOutliers);
            assertThrows(IllegalArgumentException.class, () -> 
                scaler.transform(new double[][] {{1.0, 2.0, 3.0}})
            );
        }
        
        @Test
        @DisplayName("RobustScaler - Constant Feature Handling")
        void testRobustScalerConstantFeature() {
            double[][] dataWithConstant = new double[10][2];
            for (int i = 0; i < 10; i++) {
                dataWithConstant[i][0] = 5.0; // Constant
                dataWithConstant[i][1] = i;   // Variable
            }
            
            RobustScaler scaler = new RobustScaler();
            double[][] scaled = scaler.fitTransform(dataWithConstant);
            
            // Constant feature should remain centered but scale = 1
            assertEquals(1.0, scaler.getScale()[0], TOLERANCE);
            
            // Scaled values for constant feature should be 0 (centered)
            for (int i = 0; i < scaled.length; i++) {
                assertEquals(0.0, scaled[i][0], TOLERANCE);
            }
        }
    }
    
    // =========================================================================
    // Integration Tests
    // =========================================================================
    
    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {
        
        @Test
        @DisplayName("RobustScaler + SVR Pipeline")
        void testRobustScalerSVRPipeline() {
            // Scale data first
            RobustScaler scaler = new RobustScaler.Builder().build();
            double[][] scaledX = scaler.fitTransform(linearRegressionX);
            
            // Train SVR on scaled data
            SVR svr = new SVR.Builder()
                .kernel(Kernel.Type.LINEAR)
                .C(1.0)
                .build();
            
            svr.train(scaledX, linearRegressionY);
            
            // Predict on scaled test data
            double[] testPoint = scaler.transform(new double[] {5.0, 5.0});
            double prediction = svr.predict(testPoint);
            
            assertFalse(Double.isNaN(prediction));
        }
        
        @Test
        @DisplayName("RobustScaler + LDA Pipeline")
        void testRobustScalerLDAPipeline() {
            // Scale classification data
            RobustScaler scaler = new RobustScaler.Builder().build();
            double[][] scaledX = scaler.fitTransform(classificationX);
            
            // Train LDA on scaled data
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            lda.train(scaledX, classificationY);
            
            // Score on scaled data
            double score = lda.score(scaledX, classificationY);
            
            assertTrue(score > 0.5, "LDA should achieve reasonable accuracy");
        }
        
        @Test
        @DisplayName("RobustScaler + GPR Pipeline")
        void testRobustScalerGPRPipeline() {
            // Use smaller dataset
            double[][] X = Arrays.copyOf(linearRegressionX, 30);
            double[] y = Arrays.copyOf(linearRegressionY, 30);
            
            // Scale data
            RobustScaler scaler = new RobustScaler.Builder().build();
            double[][] scaledX = scaler.fitTransform(X);
            
            // Train GPR on scaled data
            GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
                .kernel(GaussianProcessRegressor.KernelType.RBF)
                .lengthScale(1.0)
                .build();
            
            gpr.train(scaledX, y);
            
            // Predict with uncertainty
            double[][] result = gpr.predictWithStd(scaledX);
            
            assertEquals(2, result.length);
            assertEquals(scaledX.length, result[0].length);
        }
        
        @Test
        @DisplayName("Compare SVR Kernels")
        void testCompareSVRKernels() {
            Map<String, Double> scores = new HashMap<>();
            
            String[] kernels = {"linear", "rbf", "polynomial"};
            
            for (String kernel : kernels) {
                SVR svr = new SVR.Builder()
                    .kernel(kernel)
                    .C(1.0)
                    .epsilon(0.1)
                    .build();
                
                svr.train(linearRegressionX, linearRegressionY);
                double score = svr.score(linearRegressionX, linearRegressionY);
                scores.put(kernel, score);
            }
            
            // All kernels should produce valid scores
            for (Map.Entry<String, Double> entry : scores.entrySet()) {
                assertFalse(Double.isNaN(entry.getValue()),
                    entry.getKey() + " kernel produced NaN score");
            }
        }
        
        @Test
        @DisplayName("LDA vs GPR on Regression (Transformed)")
        void testLDATransformForGPR() {
            // Use LDA for dimensionality reduction
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .nComponents(2)
                .build();
            
            // Create classification labels for LDA (discretize regression target)
            int[] discreteY = new int[linearRegressionY.length];
            double yMin = Arrays.stream(linearRegressionY).min().orElse(0);
            double yMax = Arrays.stream(linearRegressionY).max().orElse(1);
            double range = (yMax - yMin) / 3;
            
            for (int i = 0; i < linearRegressionY.length; i++) {
                if (linearRegressionY[i] < yMin + range) {
                    discreteY[i] = 0;
                } else if (linearRegressionY[i] < yMin + 2 * range) {
                    discreteY[i] = 1;
                } else {
                    discreteY[i] = 2;
                }
            }
            
            // Transform using LDA
            double[][] transformedX = lda.fitTransform(linearRegressionX, discreteY);
            
            // Use smaller subset for GPR
            double[][] X = Arrays.copyOf(transformedX, 30);
            double[] y = Arrays.copyOf(linearRegressionY, 30);
            
            // Train GPR on reduced dimensions
            GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
                .kernel(GaussianProcessRegressor.KernelType.RBF)
                .build();
            
            gpr.train(X, y);
            
            double score = gpr.score(X, y);
            assertTrue(score > 0, "GPR on LDA-transformed data should have positive R^2");
        }
    }
    
    // =========================================================================
    // Edge Cases and Stress Tests
    // =========================================================================
    
    @Nested
    @DisplayName("Edge Cases and Stress Tests")
    class EdgeCaseTests {
        
        @Test
        @DisplayName("Small Dataset - SVR")
        void testSmallDatasetSVR() {
            double[][] X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
            double[] y = {1.0, 2.0, 3.0};
            
            SVR svr = new SVR.Builder().build();
            svr.train(X, y);
            
            assertTrue(svr.isFitted());
        }
        
        @Test
        @DisplayName("Small Dataset - LDA")
        void testSmallDatasetLDA() {
            double[][] X = {
                {1.0, 2.0}, {1.5, 2.5}, {2.0, 3.0},
                {5.0, 6.0}, {5.5, 6.5}, {6.0, 7.0}
            };
            int[] y = {0, 0, 0, 1, 1, 1};
            
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            lda.train(X, y);
            
            assertTrue(lda.isFitted());
            assertEquals(2, lda.getNumClasses());
        }
        
        @Test
        @DisplayName("High Dimensional Data - SVR")
        void testHighDimensionalSVR() {
            Random random = new Random(42);
            int n = 50;
            int d = 20;
            
            double[][] X = new double[n][d];
            double[] y = new double[n];
            
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < d; j++) {
                    X[i][j] = random.nextDouble();
                    sum += X[i][j];
                }
                y[i] = sum + random.nextGaussian() * 0.1;
            }
            
            SVR svr = new SVR.Builder()
                .kernel(Kernel.Type.LINEAR)
                .build();
            
            svr.train(X, y);
            
            double score = svr.score(X, y);
            assertTrue(score > 0, "SVR should fit high-dimensional data");
        }
        
        @Test
        @DisplayName("Binary Classification - LDA")
        void testBinaryLDA() {
            Random random = new Random(42);
            
            double[][] X = new double[100][3];
            int[] y = new int[100];
            
            for (int i = 0; i < 50; i++) {
                X[i][0] = random.nextGaussian();
                X[i][1] = random.nextGaussian();
                X[i][2] = random.nextGaussian();
                y[i] = 0;
                
                X[i + 50][0] = random.nextGaussian() + 3;
                X[i + 50][1] = random.nextGaussian() + 3;
                X[i + 50][2] = random.nextGaussian() + 3;
                y[i + 50] = 1;
            }
            
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            lda.train(X, y);
            
            assertEquals(2, lda.getNumClasses());
            // Binary LDA has only 1 component
            assertEquals(1, lda.getScalings()[0].length);
            
            double score = lda.score(X, y);
            assertTrue(score > 0.8, "Binary LDA should achieve high accuracy");
        }
        
        @Test
        @DisplayName("Identical Features - RobustScaler")
        void testIdenticalFeaturesRobustScaler() {
            double[][] data = new double[10][2];
            for (int i = 0; i < 10; i++) {
                data[i][0] = 5.0; // All same
                data[i][1] = i;   // Different
            }
            
            RobustScaler scaler = new RobustScaler();
            double[][] scaled = scaler.fitTransform(data);
            
            // Should handle constant feature gracefully
            assertNotNull(scaled);
            assertEquals(10, scaled.length);
        }
        
        @Test
        @DisplayName("Many Classes - LDA")
        void testManyClassesLDA() {
            Random random = new Random(42);
            int numClasses = 10;
            int samplesPerClass = 20;
            int n = numClasses * samplesPerClass;
            
            double[][] X = new double[n][5];
            int[] y = new int[n];
            
            for (int c = 0; c < numClasses; c++) {
                for (int i = 0; i < samplesPerClass; i++) {
                    int idx = c * samplesPerClass + i;
                    for (int j = 0; j < 5; j++) {
                        X[idx][j] = random.nextGaussian() + c * 2;
                    }
                    y[idx] = c;
                }
            }
            
            LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
                .build();
            
            lda.train(X, y);
            
            assertEquals(10, lda.getNumClasses());
            assertTrue(lda.score(X, y) > 0.5);
        }
    }
    
    // =========================================================================
    // Parameterized Tests
    // =========================================================================
    
    @ParameterizedTest
    @DisplayName("SVR with Different C Values")
    @ValueSource(doubles = {0.1, 1.0, 10.0, 100.0})
    void testSVRWithDifferentC(double C) {
        SVR svr = new SVR.Builder()
            .kernel(Kernel.Type.LINEAR)
            .C(C)
            .build();
        
        svr.train(linearRegressionX, linearRegressionY);
        
        assertTrue(svr.isFitted());
        assertEquals(C, svr.getC(), TOLERANCE);
    }
    
    @ParameterizedTest
    @DisplayName("GPR with Different Length Scales")
    @ValueSource(doubles = {0.1, 0.5, 1.0, 2.0, 5.0})
    void testGPRWithDifferentLengthScales(double lengthScale) {
        GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
            .kernel(GaussianProcessRegressor.KernelType.RBF)
            .lengthScale(lengthScale)
            .build();
        
        double[][] X = Arrays.copyOf(linearRegressionX, 20);
        double[] y = Arrays.copyOf(linearRegressionY, 20);
        
        gpr.train(X, y);
        
        assertTrue(gpr.isFitted());
        assertEquals(lengthScale, gpr.getLengthScale(), TOLERANCE);
    }
    
    @ParameterizedTest
    @DisplayName("RobustScaler with Different Quantile Ranges")
    @CsvSource({
        "5.0, 95.0",
        "10.0, 90.0",
        "25.0, 75.0",
        "1.0, 99.0"
    })
    void testRobustScalerWithDifferentQuantiles(double min, double max) {
        RobustScaler scaler = new RobustScaler.Builder()
            .quantileRange(min, max)
            .build();
        
        double[][] scaled = scaler.fitTransform(dataWithOutliers);
        
        assertTrue(scaler.isFitted());
        assertEquals(min, scaler.getQuantileRange()[0], TOLERANCE);
        assertEquals(max, scaler.getQuantileRange()[1], TOLERANCE);
    }
}
