package com.mindforge.model_selection;

import java.util.*;
import java.util.function.*;

/**
 * Exhaustive search over specified parameter values for an estimator.
 * 
 * @param <M> Model type
 */
public class GridSearchCV<M> {
    
    private final int cv;
    private final BiFunction<M, double[][], int[]> fitFunction;
    private final BiFunction<M, double[][], int[]> predictFunction;
    private final BiFunction<int[], int[], Double> scorer;
    
    private Map<String, Object> bestParams;
    private double bestScore;
    private List<Map<String, Object>> cvResults;
    
    public GridSearchCV(int cv,
                        BiFunction<M, double[][], int[]> fitFunction,
                        BiFunction<M, double[][], int[]> predictFunction,
                        BiFunction<int[], int[], Double> scorer) {
        this.cv = cv;
        this.fitFunction = fitFunction;
        this.predictFunction = predictFunction;
        this.scorer = scorer;
    }
    
    public void fit(double[][] X, int[] y, 
                    Map<String, List<Object>> paramGrid,
                    Supplier<M> modelSupplier,
                    BiConsumer<M, Map<String, Object>> paramSetter) {
        
        List<Map<String, Object>> paramCombinations = generateCombinations(paramGrid);
        cvResults = new ArrayList<>();
        bestScore = Double.NEGATIVE_INFINITY;
        
        KFold kfold = new KFold(cv, true, new Random(42));
        List<int[][]> splits = kfold.split(X.length);
        
        for (Map<String, Object> params : paramCombinations) {
            double[] scores = new double[cv];
            
            for (int fold = 0; fold < cv; fold++) {
                int[] trainIdx = splits.get(fold)[0];
                int[] testIdx = splits.get(fold)[1];
                
                double[][] XTrain = subset(X, trainIdx);
                double[][] XTest = subset(X, testIdx);
                int[] yTrain = subset(y, trainIdx);
                int[] yTest = subset(y, testIdx);
                
                M model = modelSupplier.get();
                paramSetter.accept(model, params);
                fitFunction.apply(model, XTrain);
                int[] yPred = predictFunction.apply(model, XTest);
                
                scores[fold] = scorer.apply(yTest, yPred);
            }
            
            double meanScore = Arrays.stream(scores).average().orElse(0);
            
            Map<String, Object> result = new HashMap<>(params);
            result.put("mean_score", meanScore);
            result.put("std_score", std(scores));
            cvResults.add(result);
            
            if (meanScore > bestScore) {
                bestScore = meanScore;
                bestParams = new HashMap<>(params);
            }
        }
    }
    
    private List<Map<String, Object>> generateCombinations(Map<String, List<Object>> paramGrid) {
        List<Map<String, Object>> result = new ArrayList<>();
        result.add(new HashMap<>());
        
        for (Map.Entry<String, List<Object>> entry : paramGrid.entrySet()) {
            List<Map<String, Object>> newResult = new ArrayList<>();
            for (Map<String, Object> existing : result) {
                for (Object value : entry.getValue()) {
                    Map<String, Object> newMap = new HashMap<>(existing);
                    newMap.put(entry.getKey(), value);
                    newResult.add(newMap);
                }
            }
            result = newResult;
        }
        
        return result;
    }
    
    private double[][] subset(double[][] X, int[] indices) {
        double[][] result = new double[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            result[i] = X[indices[i]].clone();
        }
        return result;
    }
    
    private int[] subset(int[] y, int[] indices) {
        int[] result = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            result[i] = y[indices[i]];
        }
        return result;
    }
    
    private double std(double[] values) {
        double mean = Arrays.stream(values).average().orElse(0);
        double variance = Arrays.stream(values).map(v -> (v - mean) * (v - mean)).average().orElse(0);
        return Math.sqrt(variance);
    }
    
    public Map<String, Object> getBestParams() { return bestParams; }
    public double getBestScore() { return bestScore; }
    public List<Map<String, Object>> getCvResults() { return cvResults; }
}
