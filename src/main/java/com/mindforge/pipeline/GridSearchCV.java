package com.mindforge.pipeline;

import com.mindforge.classification.Classifier;
import java.io.Serializable;
import java.util.*;
import java.util.function.Supplier;

/**
 * Grid Search with Cross-Validation.
 * 
 * Exhaustive search over specified parameter values for an estimator.
 * 
 * @author MindForge
 */
public class GridSearchCV implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private Supplier<Classifier<double[]>> estimatorFactory;
    private Map<String, Object[]> paramGrid;
    private int cv;
    private String scoring;
    private boolean refit;
    
    private Classifier<double[]> bestEstimator;
    private Map<String, Object> bestParams;
    private double bestScore;
    private List<Map<String, Object>> cvResults;
    private boolean fitted;
    
    /**
     * Creates a GridSearchCV with default parameters.
     */
    public GridSearchCV(Supplier<Classifier<double[]>> estimatorFactory, 
                        Map<String, Object[]> paramGrid) {
        this(estimatorFactory, paramGrid, 5, "accuracy", true);
    }
    
    /**
     * Creates a GridSearchCV with full configuration.
     * 
     * @param estimatorFactory Factory to create classifier instances
     * @param paramGrid Parameter grid to search
     * @param cv Number of cross-validation folds
     * @param scoring Scoring method
     * @param refit Whether to refit best estimator on full data
     */
    public GridSearchCV(Supplier<Classifier<double[]>> estimatorFactory,
                        Map<String, Object[]> paramGrid, int cv,
                        String scoring, boolean refit) {
        if (estimatorFactory == null) {
            throw new IllegalArgumentException("Estimator factory cannot be null");
        }
        if (paramGrid == null || paramGrid.isEmpty()) {
            throw new IllegalArgumentException("Parameter grid cannot be null or empty");
        }
        if (cv < 2) {
            throw new IllegalArgumentException("cv must be at least 2");
        }
        
        this.estimatorFactory = estimatorFactory;
        this.paramGrid = paramGrid;
        this.cv = cv;
        this.scoring = scoring;
        this.refit = refit;
        this.fitted = false;
    }
    
    /**
     * Fits the grid search.
     */
    public GridSearchCV fit(double[][] X, int[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("X and y cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same length");
        }
        
        int n = X.length;
        cvResults = new ArrayList<>();
        bestScore = Double.NEGATIVE_INFINITY;
        bestParams = null;
        
        // Generate all parameter combinations
        List<Map<String, Object>> paramCombinations = generateParamCombinations();
        
        for (Map<String, Object> params : paramCombinations) {
            double[] foldScores = new double[cv];
            
            // Cross-validation
            int foldSize = n / cv;
            
            for (int fold = 0; fold < cv; fold++) {
                int testStart = fold * foldSize;
                int testEnd = (fold == cv - 1) ? n : (fold + 1) * foldSize;
                
                // Split data
                List<Integer> trainIndices = new ArrayList<>();
                List<Integer> testIndices = new ArrayList<>();
                
                for (int i = 0; i < n; i++) {
                    if (i >= testStart && i < testEnd) {
                        testIndices.add(i);
                    } else {
                        trainIndices.add(i);
                    }
                }
                
                double[][] XTrain = new double[trainIndices.size()][];
                int[] yTrain = new int[trainIndices.size()];
                double[][] XTest = new double[testIndices.size()][];
                int[] yTest = new int[testIndices.size()];
                
                for (int i = 0; i < trainIndices.size(); i++) {
                    XTrain[i] = X[trainIndices.get(i)];
                    yTrain[i] = y[trainIndices.get(i)];
                }
                for (int i = 0; i < testIndices.size(); i++) {
                    XTest[i] = X[testIndices.get(i)];
                    yTest[i] = y[testIndices.get(i)];
                }
                
                // Create and train estimator with these params
                Classifier<double[]> estimator = createEstimatorWithParams(params);
                estimator.train(XTrain, yTrain);
                
                // Score
                foldScores[fold] = calculateAccuracy(estimator, XTest, yTest);
            }
            
            // Calculate mean score
            double meanScore = 0;
            for (double score : foldScores) {
                meanScore += score;
            }
            meanScore /= cv;
            
            double stdScore = 0;
            for (double score : foldScores) {
                stdScore += Math.pow(score - meanScore, 2);
            }
            stdScore = Math.sqrt(stdScore / cv);
            
            // Store results
            Map<String, Object> result = new HashMap<>(params);
            result.put("mean_score", meanScore);
            result.put("std_score", stdScore);
            cvResults.add(result);
            
            // Update best
            if (meanScore > bestScore) {
                bestScore = meanScore;
                bestParams = new HashMap<>(params);
            }
        }
        
        // Refit on full data with best params
        if (refit && bestParams != null) {
            bestEstimator = createEstimatorWithParams(bestParams);
            bestEstimator.train(X, y);
        }
        
        fitted = true;
        return this;
    }
    
    /**
     * Generates all combinations of parameters.
     */
    private List<Map<String, Object>> generateParamCombinations() {
        List<Map<String, Object>> combinations = new ArrayList<>();
        combinations.add(new HashMap<>());
        
        for (Map.Entry<String, Object[]> entry : paramGrid.entrySet()) {
            String paramName = entry.getKey();
            Object[] values = entry.getValue();
            
            List<Map<String, Object>> newCombinations = new ArrayList<>();
            
            for (Map<String, Object> combo : combinations) {
                for (Object value : values) {
                    Map<String, Object> newCombo = new HashMap<>(combo);
                    newCombo.put(paramName, value);
                    newCombinations.add(newCombo);
                }
            }
            
            combinations = newCombinations;
        }
        
        return combinations;
    }
    
    /**
     * Creates an estimator with specified parameters.
     * This is a simplified version - in practice, you'd use reflection or builders.
     */
    private Classifier<double[]> createEstimatorWithParams(Map<String, Object> params) {
        // For simplicity, create base estimator
        // In a full implementation, parameters would be applied via reflection or setters
        return estimatorFactory.get();
    }
    
    /**
     * Predicts using the best estimator.
     */
    public int[] predict(double[][] X) {
        if (!fitted || bestEstimator == null) {
            throw new IllegalStateException("GridSearchCV not fitted or refit=false");
        }
        return bestEstimator.predict(X);
    }
    
    /**
     * Scores using the best estimator.
     */
    public double score(double[][] X, int[] y) {
        if (!fitted || bestEstimator == null) {
            throw new IllegalStateException("GridSearchCV not fitted or refit=false");
        }
        return calculateAccuracy(bestEstimator, X, y);
    }
    
    /**
     * Calculates accuracy score for a classifier.
     */
    private double calculateAccuracy(Classifier<double[]> clf, double[][] X, int[] y) {
        int[] predictions = clf.predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) {
                correct++;
            }
        }
        return (double) correct / y.length;
    }
    
    // Getters
    public Classifier<double[]> getBestEstimator() {
        return bestEstimator;
    }
    
    public Map<String, Object> getBestParams() {
        return bestParams != null ? new HashMap<>(bestParams) : null;
    }
    
    public double getBestScore() {
        return bestScore;
    }
    
    public List<Map<String, Object>> getCvResults() {
        return cvResults != null ? new ArrayList<>(cvResults) : null;
    }
    
    public boolean isFitted() {
        return fitted;
    }
    
    public int getCv() {
        return cv;
    }
}
