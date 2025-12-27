package io.github.yasmramos.mindforge.model_selection;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.function.Function;

/**
 * Tests for Bayesian Optimization.
 */
public class BayesianOptimizationTest {
    
    @Test
    public void testOptimizeSimpleFunction() {
        // Optimize a simple quadratic: -(x-3)^2 + 10
        // Maximum at x=3 with value 10
        Function<Map<String, Double>, Double> objective = params -> {
            double x = params.get("x");
            return -(x - 3) * (x - 3) + 10;
        };
        
        BayesianOptimization bo = new BayesianOptimization.Builder()
            .nIterations(20)
            .nInitialPoints(5)
            .seed(42)
            .build();
        
        Map<String, double[]> bounds = new HashMap<>();
        bounds.put("x", new double[]{0, 6});
        bo.setParameterBounds(bounds);
        
        Map<String, Double> bestParams = bo.optimize(objective);
        
        assertNotNull(bestParams);
        assertTrue(bestParams.containsKey("x"));
        
        // Should find x close to 3
        assertEquals(3.0, bestParams.get("x"), 0.5);
        
        // Best value should be close to 10
        assertTrue(bo.getBestValue() > 9);
    }
    
    @Test
    public void testOptimizeTwoDimensional() {
        // Optimize: -(x-2)^2 - (y-3)^2 + 20
        // Maximum at (2, 3) with value 20
        Function<Map<String, Double>, Double> objective = params -> {
            double x = params.get("x");
            double y = params.get("y");
            return -(x - 2) * (x - 2) - (y - 3) * (y - 3) + 20;
        };
        
        BayesianOptimization bo = new BayesianOptimization.Builder()
            .nIterations(30)
            .nInitialPoints(10)
            .seed(123)
            .build();
        
        Map<String, double[]> bounds = new LinkedHashMap<>();
        bounds.put("x", new double[]{0, 5});
        bounds.put("y", new double[]{0, 5});
        bo.setParameterBounds(bounds);
        
        Map<String, Double> bestParams = bo.optimize(objective);
        
        assertNotNull(bestParams);
        assertEquals(2, bestParams.size());
        
        // Should be reasonably close to optimal
        assertTrue(bo.getBestValue() > 18);
    }
    
    @Test
    public void testGetBestParamsAsMap() {
        Function<Map<String, Double>, Double> objective = params -> -params.get("x") * params.get("x");
        
        BayesianOptimization bo = new BayesianOptimization.Builder()
            .nIterations(10)
            .nInitialPoints(3)
            .build();
        
        Map<String, double[]> bounds = new HashMap<>();
        bounds.put("x", new double[]{-5, 5});
        bo.setParameterBounds(bounds);
        
        bo.optimize(objective);
        
        Map<String, Double> bestMap = bo.getBestParamsAsMap();
        assertNotNull(bestMap);
        assertTrue(bestMap.containsKey("x"));
    }
    
    @Test
    public void testGetObservedValues() {
        Function<Map<String, Double>, Double> objective = params -> params.get("x");
        
        BayesianOptimization bo = new BayesianOptimization.Builder()
            .nIterations(5)
            .nInitialPoints(3)
            .build();
        
        Map<String, double[]> bounds = new HashMap<>();
        bounds.put("x", new double[]{0, 10});
        bo.setParameterBounds(bounds);
        
        bo.optimize(objective);
        
        List<Double> observed = bo.getObservedValues();
        assertNotNull(observed);
        assertEquals(8, observed.size()); // 3 initial + 5 iterations
    }
    
    @Test
    public void testWithoutBounds() {
        BayesianOptimization bo = new BayesianOptimization.Builder().build();
        
        assertThrows(IllegalStateException.class, () -> 
            bo.optimize(params -> 0.0));
    }
    
    @Test
    public void testExplorationWeight() {
        Function<Map<String, Double>, Double> objective = params -> {
            double x = params.get("x");
            return -x * x; // Maximum at x=0
        };
        
        // High exploration weight
        BayesianOptimization bo = new BayesianOptimization.Builder()
            .nIterations(10)
            .nInitialPoints(3)
            .explorationWeight(0.5)
            .seed(42)
            .build();
        
        Map<String, double[]> bounds = new HashMap<>();
        bounds.put("x", new double[]{-5, 5});
        bo.setParameterBounds(bounds);
        
        bo.optimize(objective);
        
        // Should still find the optimum
        assertTrue(bo.getBestValue() > -1);
    }
    
    @Test
    public void testMultipleParameters() {
        Function<Map<String, Double>, Double> objective = params -> {
            double a = params.get("a");
            double b = params.get("b");
            double c = params.get("c");
            return -(a - 1) * (a - 1) - (b - 2) * (b - 2) - (c - 3) * (c - 3);
        };
        
        BayesianOptimization bo = new BayesianOptimization.Builder()
            .nIterations(40)
            .nInitialPoints(10)
            .seed(42)
            .build();
        
        Map<String, double[]> bounds = new LinkedHashMap<>();
        bounds.put("a", new double[]{0, 5});
        bounds.put("b", new double[]{0, 5});
        bounds.put("c", new double[]{0, 5});
        bo.setParameterBounds(bounds);
        
        Map<String, Double> best = bo.optimize(objective);
        
        assertEquals(3, best.size());
        assertTrue(bo.getBestValue() > -3); // Reasonably close to 0
    }
    
    @Test
    public void testNoisyObjective() {
        Random random = new Random(42);
        
        Function<Map<String, Double>, Double> objective = params -> {
            double x = params.get("x");
            return -(x - 5) * (x - 5) + random.nextGaussian() * 0.1;
        };
        
        BayesianOptimization bo = new BayesianOptimization.Builder()
            .nIterations(20)
            .nInitialPoints(5)
            .seed(42)
            .build();
        
        Map<String, double[]> bounds = new HashMap<>();
        bounds.put("x", new double[]{0, 10});
        bo.setParameterBounds(bounds);
        
        bo.optimize(objective);
        
        // Should find approximately x=5 despite noise
        double bestX = bo.getBestParamsAsMap().get("x");
        assertEquals(5.0, bestX, 2.0);
    }
}
