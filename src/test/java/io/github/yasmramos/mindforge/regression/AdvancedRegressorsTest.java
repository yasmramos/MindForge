package io.github.yasmramos.mindforge.regression;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class AdvancedRegressorsTest {
    
    private double[][] generateData(int n, Random random) {
        double[][] X = new double[n][2];
        for (int i = 0; i < n; i++) {
            X[i][0] = random.nextDouble() * 10;
            X[i][1] = random.nextDouble() * 10;
        }
        return X;
    }
    
    private double[] generateTargets(double[][] X) {
        double[] y = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            y[i] = 2 * X[i][0] + 3 * X[i][1] + 1;
        }
        return y;
    }
    
    @Nested
    class DecisionTreeRegressorTests {
        @Test
        void testBasicFit() {
            double[][] X = {{1}, {2}, {3}, {4}, {5}};
            double[] y = {1.0, 2.0, 3.0, 4.0, 5.0};
            
            DecisionTreeRegressor dt = new DecisionTreeRegressor();
            dt.train(X, y);
            
            double pred = dt.predict(new double[]{3});
            assertTrue(pred >= 1 && pred <= 5);
        }
        
        @Test
        void testWithMaxDepth() {
            double[][] X = {{1}, {2}, {3}, {4}};
            double[] y = {1.0, 2.0, 3.0, 4.0};
            
            DecisionTreeRegressor dt = new DecisionTreeRegressor();
            dt.train(X, y);
            assertNotNull(dt.predict(new double[]{2.5}));
        }
    }
    
    @Nested
    class RandomForestRegressorTests {
        @Test
        void testBasicFit() {
            Random random = new Random(42);
            double[][] X = generateData(50, random);
            double[] y = generateTargets(X);
            
            RandomForestRegressor rf = new RandomForestRegressor(10);
            rf.train(X, y);
            
            double pred = rf.predict(X[0]);
            assertNotNull(pred);
        }
        
        @Test
        void testBatchPredict() {
            Random random = new Random(42);
            double[][] X = generateData(30, random);
            double[] y = generateTargets(X);
            
            RandomForestRegressor rf = new RandomForestRegressor(5);
            rf.train(X, y);
            
            double[] preds = rf.predict(X);
            assertEquals(30, preds.length);
        }
    }
    
    @Nested
    class GradientBoostingRegressorTests {
        @Test
        void testBasicFit() {
            double[][] X = {{1}, {2}, {3}, {4}, {5}, {6}};
            double[] y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
            
            GradientBoostingRegressor gb = new GradientBoostingRegressor(10, 0.1, 3);
            gb.train(X, y);
            
            double pred = gb.predict(new double[]{3.5});
            assertTrue(pred >= 1 && pred <= 7);
        }
    }
    
    @Nested
    class SVRTests {
        @Test
        void testBasicFit() {
            double[][] X = {{1}, {2}, {3}, {4}, {5}};
            double[] y = {1.0, 2.0, 3.0, 4.0, 5.0};
            
            SVR svr = new SVR.Builder().build();
            svr.train(X, y);
            
            double pred = svr.predict(new double[]{3});
            assertNotNull(pred);
        }
        
        @Test
        void testWithKernel() {
            double[][] X = {{1}, {2}, {3}, {4}};
            double[] y = {1.0, 4.0, 9.0, 16.0};
            
            SVR svr = new SVR.Builder().C(1.0).epsilon(0.1).build();
            svr.train(X, y);
            
            double pred = svr.predict(new double[]{2.5});
            assertNotNull(pred);
        }
    }
    
    @Nested
    class GaussianProcessRegressorTests {
        @Test
        void testBasicFit() {
            double[][] X = {{1}, {2}, {3}, {4}};
            double[] y = {1.0, 2.0, 3.0, 4.0};
            
            GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder().build();
            gpr.train(X, y);
            
            double pred = gpr.predict(new double[]{2.5});
            assertNotNull(pred);
        }
    }
}
