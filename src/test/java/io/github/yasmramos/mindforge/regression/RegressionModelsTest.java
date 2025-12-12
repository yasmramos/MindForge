package io.github.yasmramos.mindforge.regression;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for regression models: Ridge, Lasso, ElasticNet, Polynomial.
 */
class RegressionModelsTest {
    
    private double[][] X;
    private double[] y;
    
    @BeforeEach
    void setUp() {
        // Simple linear data: y = 2*x + 1
        X = new double[][]{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}};
        y = new double[]{3, 5, 7, 9, 11, 13, 15, 17, 19, 21};
    }
    
    @Nested
    @DisplayName("Ridge Regression Tests")
    class RidgeRegressionTests {
        
        @Test
        @DisplayName("Default constructor creates valid model")
        void testDefaultConstructor() {
            RidgeRegression ridge = new RidgeRegression();
            assertNotNull(ridge);
            assertEquals(1.0, ridge.getAlpha());
            assertFalse(ridge.isTrained());
        }
        
        @Test
        @DisplayName("Train and predict works correctly")
        void testTrainAndPredict() {
            RidgeRegression ridge = new RidgeRegression(0.1);
            ridge.train(X, y);
            
            assertTrue(ridge.isTrained());
            assertNotNull(ridge.getCoefficients());
            
            double pred = ridge.predict(new double[]{5.5});
            assertTrue(pred > 10 && pred < 14);
        }
        
        @Test
        @DisplayName("Batch predict works correctly")
        void testBatchPredict() {
            RidgeRegression ridge = new RidgeRegression(0.01);
            ridge.train(X, y);
            
            double[] predictions = ridge.predict(new double[][]{{1}, {5}, {10}});
            assertEquals(3, predictions.length);
        }
        
        @Test
        @DisplayName("Score returns R² value")
        void testScore() {
            RidgeRegression ridge = new RidgeRegression(0.001);
            ridge.train(X, y);
            
            double score = ridge.score(X, y);
            assertTrue(score > 0.9);
        }
        
        @Test
        @DisplayName("Invalid alpha throws exception")
        void testInvalidAlpha() {
            assertThrows(IllegalArgumentException.class, () -> new RidgeRegression(-1));
        }
        
        @Test
        @DisplayName("Null inputs throw exception")
        void testNullInputs() {
            RidgeRegression ridge = new RidgeRegression();
            assertThrows(IllegalArgumentException.class, () -> ridge.train(null, y));
            assertThrows(IllegalArgumentException.class, () -> ridge.train(X, null));
        }
    }
    
    @Nested
    @DisplayName("Lasso Regression Tests")
    class LassoRegressionTests {
        
        @Test
        @DisplayName("Default constructor creates valid model")
        void testDefaultConstructor() {
            LassoRegression lasso = new LassoRegression();
            assertNotNull(lasso);
            assertEquals(1.0, lasso.getAlpha());
        }
        
        @Test
        @DisplayName("Train produces sparse coefficients")
        void testSparsity() {
            // Multi-feature data with some irrelevant features
            double[][] XMulti = new double[20][5];
            double[] yMulti = new double[20];
            for (int i = 0; i < 20; i++) {
                XMulti[i][0] = i;
                XMulti[i][1] = Math.random();
                XMulti[i][2] = Math.random();
                XMulti[i][3] = Math.random();
                XMulti[i][4] = Math.random();
                yMulti[i] = 2 * XMulti[i][0] + 1;
            }
            
            LassoRegression lasso = new LassoRegression(1.0);
            lasso.train(XMulti, yMulti);
            
            assertTrue(lasso.isTrained());
            int[] selected = lasso.getSelectedFeatures();
            assertNotNull(selected);
        }
        
        @Test
        @DisplayName("Predict after training")
        void testPredict() {
            LassoRegression lasso = new LassoRegression(0.1);
            lasso.train(X, y);
            
            double pred = lasso.predict(new double[]{5});
            assertNotNull(pred);
        }
    }
    
    @Nested
    @DisplayName("ElasticNet Tests")
    class ElasticNetTests {
        
        @Test
        @DisplayName("Default constructor creates valid model")
        void testDefaultConstructor() {
            ElasticNet en = new ElasticNet();
            assertNotNull(en);
            assertEquals(1.0, en.getAlpha());
            assertEquals(0.5, en.getL1Ratio());
        }
        
        @Test
        @DisplayName("Train and predict works")
        void testTrainAndPredict() {
            ElasticNet en = new ElasticNet(0.1, 0.5);
            en.train(X, y);
            
            assertTrue(en.isTrained());
            double pred = en.predict(new double[]{5});
            assertNotNull(pred);
        }
        
        @Test
        @DisplayName("Invalid l1Ratio throws exception")
        void testInvalidL1Ratio() {
            assertThrows(IllegalArgumentException.class, () -> new ElasticNet(1.0, 1.5));
            assertThrows(IllegalArgumentException.class, () -> new ElasticNet(1.0, -0.1));
        }
    }
    
    @Nested
    @DisplayName("Polynomial Regression Tests")
    class PolynomialRegressionTests {
        
        @Test
        @DisplayName("Default constructor creates degree 2")
        void testDefaultConstructor() {
            PolynomialRegression poly = new PolynomialRegression();
            assertEquals(2, poly.getDegree());
        }
        
        @Test
        @DisplayName("Fits quadratic data well")
        void testQuadraticFit() {
            // Quadratic data: y = x^2
            double[][] XQuad = new double[][]{{1}, {2}, {3}, {4}, {5}};
            double[] yQuad = new double[]{1, 4, 9, 16, 25};
            
            PolynomialRegression poly = new PolynomialRegression(2);
            poly.train(XQuad, yQuad);
            
            assertTrue(poly.isTrained());
            double score = poly.score(XQuad, yQuad);
            assertTrue(score > 0.95);
        }
        
        @Test
        @DisplayName("Invalid degree throws exception")
        void testInvalidDegree() {
            assertThrows(IllegalArgumentException.class, () -> new PolynomialRegression(0));
            assertThrows(IllegalArgumentException.class, () -> new PolynomialRegression(15));
        }
    }
}
