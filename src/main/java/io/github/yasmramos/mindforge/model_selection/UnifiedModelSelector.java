package io.github.yasmramos.mindforge.model_selection;

import io.github.yasmramos.mindforge.classification.Classifier;
import io.github.yasmramos.mindforge.regression.Regressor;
import io.github.yasmramos.mindforge.clustering.Clusterer;

import java.io.Serializable;
import java.util.*;
import java.util.function.Supplier;

/**
 * Unified Model Selector that integrates Grid Search, Randomized Search, and Bayesian Optimization.
 * Supports classification, regression, and clustering with automatic strategy selection.
 * 
 * @author MindForge
 */
public class UnifiedModelSelector implements Serializable {
    private static final long serialVersionUID = 1L;
    
    public enum SearchStrategy {
        GRID_SEARCH,
        RANDOMIZED_SEARCH,
        BAYESIAN_OPTIMIZATION,
        AUTO
    }
    
    public enum ModelType {
        CLASSIFICATION,
        REGRESSION,
        CLUSTERING
    }
    
    private final SearchStrategy strategy;
    private final ModelType modelType;
    private final int cv;
    private final String scoring;
    private final int nIterations;
    private final int nInitialPoints;
    private final long seed;
    
    private Object bestModel;
    private Map<String, Object> bestParams;
    private double bestScore;
    private List<Map<String, Object>> cvResults;
    private boolean fitted;
    
    // For Bayesian Optimization
    private Map<String, double[]> parameterBounds;
    
    /**
     * Creates a UnifiedModelSelector with auto strategy selection.
     */
    public UnifiedModelSelector(ModelType modelType, int cv, String scoring) {
        this(SearchStrategy.AUTO, modelType, cv, scoring, 50, 5, 42);
    }
    
    /**
     * Creates a UnifiedModelSelector with full configuration.
     */
    public UnifiedModelSelector(SearchStrategy strategy, ModelType modelType, int cv, 
                                String scoring, int nIterations, int nInitialPoints, long seed) {
        this.strategy = strategy;
        this.modelType = modelType;
        this.cv = cv;
        this.scoring = scoring;
        this.nIterations = nIterations;
        this.nInitialPoints = nInitialPoints;
        this.seed = seed;
        this.fitted = false;
        this.parameterBounds = new LinkedHashMap<>();
    }
    
    /**
     * Sets parameter grid for Grid Search.
     */
    public UnifiedModelSelector setParamGrid(Map<String, Object[]> paramGrid) {
        // Convert discrete values to bounds for Bayesian Optimization
        for (Map.Entry<String, Object[]> entry : paramGrid.entrySet()) {
            Object[] values = entry.getValue();
            if (values.length > 0 && values[0] instanceof Number) {
                double min = Double.MAX_VALUE;
                double max = Double.MIN_VALUE;
                for (Object val : values) {
                    double d = ((Number) val).doubleValue();
                    if (d < min) min = d;
                    if (d > max) max = d;
                }
                parameterBounds.put(entry.getKey(), new double[]{min, max});
            }
        }
        return this;
    }
    
    /**
     * Sets continuous parameter bounds for Bayesian Optimization.
     */
    public UnifiedModelSelector setParameterBounds(Map<String, double[]> bounds) {
        this.parameterBounds = new LinkedHashMap<>(bounds);
        return this;
    }
    
    /**
     * Fits the model selector using the configured strategy.
     */
    public UnifiedModelSelector fit(Supplier<Object> modelFactory, double[][] X, int[] y) {
        SearchStrategy actualStrategy = selectStrategy(modelType, parameterBounds);
        
        switch (actualStrategy) {
            case GRID_SEARCH:
                runGridSearch(modelFactory, X, y);
                break;
            case RANDOMIZED_SEARCH:
                runRandomizedSearch(modelFactory, X, y);
                break;
            case BAYESIAN_OPTIMIZATION:
                runBayesianOptimization(modelFactory, X, y);
                break;
            default:
                throw new IllegalStateException("Unknown strategy: " + actualStrategy);
        }
        
        fitted = true;
        return this;
    }
    
    /**
     * Fits for clustering (unsupervised).
     */
    public UnifiedModelSelector fitClustering(Supplier<Clusterer<double[]>> clusterFactory, double[][] X) {
        SearchStrategy actualStrategy = selectStrategy(ModelType.CLUSTERING, parameterBounds);
        
        switch (actualStrategy) {
            case BAYESIAN_OPTIMIZATION:
                runBayesianOptimizationClustering(clusterFactory, X);
                break;
            case RANDOMIZED_SEARCH:
                runRandomizedSearchClustering(clusterFactory, X);
                break;
            default:
                runGridSearchClustering(clusterFactory, X);
                break;
        }
        
        fitted = true;
        return this;
    }
    
    private SearchStrategy selectStrategy(ModelType type, Map<String, double[]> bounds) {
        if (strategy != SearchStrategy.AUTO) {
            return strategy;
        }
        
        // Auto-select based on parameter space
        if (bounds == null || bounds.isEmpty()) {
            return SearchStrategy.GRID_SEARCH;
        }
        
        // If many parameters or continuous space, use Bayesian Optimization
        if (bounds.size() > 3) {
            return SearchStrategy.BAYESIAN_OPTIMIZATION;
        }
        
        return SearchStrategy.RANDOMIZED_SEARCH;
    }
    
    private void runGridSearch(Supplier<Object> modelFactory, double[][] X, int[] y) {
        // Simplified grid search implementation
        cvResults = new ArrayList<>();
        bestScore = Double.NEGATIVE_INFINITY;
        
        // Generate combinations from bounds (discretized)
        List<Map<String, Object>> combinations = discretizeBounds(5);
        
        for (Map<String, Object> params : combinations) {
            double score = crossValidate(modelFactory, params, X, y);
            
            Map<String, Object> result = new HashMap<>(params);
            result.put("mean_score", score);
            cvResults.add(result);
            
            if (score > bestScore) {
                bestScore = score;
                bestParams = new HashMap<>(params);
                bestModel = createModelWithParams(modelFactory, params);
                ((Classifier<double[]>)bestModel).train(X, y);
            }
        }
    }
    
    private void runRandomizedSearch(Supplier<Object> modelFactory, double[][] X, int[] y) {
        Random random = new Random(seed);
        cvResults = new ArrayList<>();
        bestScore = Double.NEGATIVE_INFINITY;
        
        for (int i = 0; i < nIterations; i++) {
            Map<String, Object> params = sampleRandomParams(random);
            double score = crossValidate(modelFactory, params, X, y);
            
            Map<String, Object> result = new HashMap<>(params);
            result.put("mean_score", score);
            cvResults.add(result);
            
            if (score > bestScore) {
                bestScore = score;
                bestParams = new HashMap<>(params);
                bestModel = createModelWithParams(modelFactory, params);
                ((Classifier<double[]>)bestModel).train(X, y);
            }
        }
    }
    
    private void runBayesianOptimization(Supplier<Object> modelFactory, double[][] X, int[] y) {
        BayesianOptimization bo = new BayesianOptimization.Builder()
            .nIterations(nIterations)
            .nInitialPoints(nInitialPoints)
            .seed(seed)
            .build();
        
        bo.setParameterBounds(parameterBounds);
        
        Map<String, Double> optimalParams = bo.optimize(params -> {
            Map<String, Object> objParams = new HashMap<>();
            params.forEach((k, v) -> objParams.put(k, v));
            return crossValidate(modelFactory, objParams, X, y);
        });
        
        bestParams = new HashMap<>(optimalParams);
        bestScore = bo.getBestValue();
        bestModel = createModelWithParams(modelFactory, bestParams);
        
        if (modelType == ModelType.CLASSIFICATION) {
            ((Classifier<double[]>)bestModel).train(X, y);
        } else if (modelType == ModelType.REGRESSION) {
            // Convert int[] y to double[] for regression
            double[] yDouble = new double[y.length];
            for (int i = 0; i < y.length; i++) {
                yDouble[i] = y[i];
            }
            ((Regressor<double[]>)bestModel).train(X, yDouble);
        }
        
        // Store BO results
        cvResults = new ArrayList<>();
        List<Double> observedValues = bo.getObservedValues();
        for (Double value : observedValues) {
            Map<String, Object> result = new HashMap<>();
            result.put("mean_score", value);
            cvResults.add(result);
        }
    }
    
    private void runBayesianOptimizationClustering(Supplier<Clusterer<double[]>> clusterFactory, double[][] X) {
        BayesianOptimization bo = new BayesianOptimization.Builder()
            .nIterations(nIterations)
            .nInitialPoints(nInitialPoints)
            .seed(seed)
            .build();
        
        bo.setParameterBounds(parameterBounds);
        
        Map<String, Double> optimalParams = bo.optimize(params -> {
            Map<String, Object> objParams = new HashMap<>();
            params.forEach((k, v) -> objParams.put(k, v));
            return evaluateClustering(clusterFactory, objParams, X);
        });
        
        bestParams = new HashMap<>(optimalParams);
        bestScore = bo.getBestValue();
        bestModel = createClusterWithParams(clusterFactory, bestParams);
        ((Clusterer<double[]>)bestModel).cluster(X);
    }
    
    private void runRandomizedSearchClustering(Supplier<Clusterer<double[]>> clusterFactory, double[][] X) {
        Random random = new Random(seed);
        cvResults = new ArrayList<>();
        bestScore = Double.NEGATIVE_INFINITY;
        
        for (int i = 0; i < nIterations; i++) {
            Map<String, Object> params = sampleRandomParams(random);
            double score = evaluateClustering(clusterFactory, params, X);
            
            Map<String, Object> result = new HashMap<>(params);
            result.put("score", score);
            cvResults.add(result);
            
            if (score > bestScore) {
                bestScore = score;
                bestParams = new HashMap<>(params);
                bestModel = createClusterWithParams(clusterFactory, params);
                ((Clusterer<double[]>)bestModel).cluster(X);
            }
        }
    }
    
    private void runGridSearchClustering(Supplier<Clusterer<double[]>> clusterFactory, double[][] X) {
        cvResults = new ArrayList<>();
        bestScore = Double.NEGATIVE_INFINITY;
        
        List<Map<String, Object>> combinations = discretizeBounds(5);
        
        for (Map<String, Object> params : combinations) {
            double score = evaluateClustering(clusterFactory, params, X);
            
            Map<String, Object> result = new HashMap<>(params);
            result.put("score", score);
            cvResults.add(result);
            
            if (score > bestScore) {
                bestScore = score;
                bestParams = new HashMap<>(params);
                bestModel = createClusterWithParams(clusterFactory, params);
                ((Clusterer<double[]>)bestModel).cluster(X);
            }
        }
    }
    
    private double crossValidate(Supplier<Object> modelFactory, Map<String, Object> params, 
                                  double[][] X, int[] y) {
        KFold kfold = new KFold(cv, false, new Random(seed));
        List<Double> scores = new ArrayList<>();
        
        for (int[][] fold : kfold.split(X.length)) {
            int[] trainIdx = fold[0];
            int[] testIdx = fold[1];
            
            double[][] XTrain = subset(X, trainIdx);
            int[] yTrain = subset(y, trainIdx);
            double[][] XTest = subset(X, testIdx);
            int[] yTest = subset(y, testIdx);
            
            Object model = createModelWithParams(modelFactory, params);
            
            if (modelType == ModelType.CLASSIFICATION) {
                Classifier<double[]> clf = (Classifier<double[]>) model;
                clf.train(XTrain, yTrain);
                scores.add(calculateAccuracy(clf, XTest, yTest));
            } else if (modelType == ModelType.REGRESSION) {
                // Convert int[] yTrain to double[] for regression
                double[] yTrainDouble = new double[yTrain.length];
                for (int i = 0; i < yTrain.length; i++) {
                    yTrainDouble[i] = yTrain[i];
                }
                Regressor<double[]> reg = (Regressor<double[]>) model;
                reg.train(XTrain, yTrainDouble);
                scores.add(calculateR2(reg, XTest, yTest));
            }
        }
        
        return scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    private double evaluateClustering(Supplier<Clusterer<double[]>> clusterFactory, 
                                      Map<String, Object> params, double[][] X) {
        Clusterer<double[]> clusterer = createClusterWithParams(clusterFactory, params);
        int[] labels = clusterer.cluster(X);
        
        // Silhouette score as default metric
        return calculateSilhouetteScore(X, labels);
    }
    
    private double calculateSilhouetteScore(double[][] X, int[] labels) {
        int n = X.length;
        if (n < 2) return 0;
        
        double sumSilhouette = 0;
        int count = 0;
        
        for (int i = 0; i < n; i++) {
            int label = labels[i];
            
            // Calculate a(i): mean distance to other points in same cluster
            double sumSame = 0;
            int countSame = 0;
            for (int j = 0; j < n; j++) {
                if (i != j && labels[j] == label) {
                    sumSame += euclideanDistance(X[i], X[j]);
                    countSame++;
                }
            }
            double a = countSame > 0 ? sumSame / countSame : 0;
            
            // Calculate b(i): min mean distance to points in other clusters
            double b = Double.MAX_VALUE;
            Set<Integer> otherLabels = new HashSet<>();
            for (int j = 0; j < n; j++) {
                if (labels[j] != label) {
                    otherLabels.add(labels[j]);
                }
            }
            
            for (int otherLabel : otherLabels) {
                double sumOther = 0;
                int countOther = 0;
                for (int j = 0; j < n; j++) {
                    if (labels[j] == otherLabel) {
                        sumOther += euclideanDistance(X[i], X[j]);
                        countOther++;
                    }
                }
                if (countOther > 0) {
                    double meanOther = sumOther / countOther;
                    if (meanOther < b) {
                        b = meanOther;
                    }
                }
            }
            
            if (b == Double.MAX_VALUE) b = 0;
            
            // Silhouette coefficient
            double s = (b - a) / Math.max(a, b);
            sumSilhouette += s;
            count++;
        }
        
        return count > 0 ? sumSilhouette / count : 0;
    }
    
    private double euclideanDistance(double[] x1, double[] x2) {
        double sum = 0;
        for (int i = 0; i < x1.length; i++) {
            double diff = x1[i] - x2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    private List<Map<String, Object>> discretizeBounds(int nPoints) {
        List<Map<String, Object>> combinations = new ArrayList<>();
        combinations.add(new HashMap<>());
        
        for (Map.Entry<String, double[]> entry : parameterBounds.entrySet()) {
            String paramName = entry.getKey();
            double[] bounds = entry.getValue();
            double step = (bounds[1] - bounds[0]) / (nPoints - 1);
            
            List<Map<String, Object>> newCombinations = new ArrayList<>();
            
            for (Map<String, Object> combo : combinations) {
                for (int i = 0; i < nPoints; i++) {
                    Map<String, Object> newCombo = new HashMap<>(combo);
                    newCombo.put(paramName, bounds[0] + i * step);
                    newCombinations.add(newCombo);
                }
            }
            
            combinations = newCombinations;
        }
        
        return combinations;
    }
    
    private Map<String, Object> sampleRandomParams(Random random) {
        Map<String, Object> params = new HashMap<>();
        for (Map.Entry<String, double[]> entry : parameterBounds.entrySet()) {
            double[] bounds = entry.getValue();
            params.put(entry.getKey(), bounds[0] + random.nextDouble() * (bounds[1] - bounds[0]));
        }
        return params;
    }
    
    private Object createModelWithParams(Supplier<Object> factory, Map<String, Object> params) {
        // In a full implementation, use reflection to set parameters
        return factory.get();
    }
    
    private Clusterer<double[]> createClusterWithParams(Supplier<Clusterer<double[]>> factory, 
                                                         Map<String, Object> params) {
        return factory.get();
    }
    
    private double calculateAccuracy(Classifier<double[]> clf, double[][] X, int[] y) {
        int[] predictions = clf.predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
    }
    
    private double calculateR2(Regressor<double[]> reg, double[][] X, int[] y) {
        double[] predictions = reg.predict(X);
        double mean = 0;
        for (double val : y) mean += val;
        mean /= y.length;
        
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < y.length; i++) {
            ssTot += Math.pow(y[i] - mean, 2);
            ssRes += Math.pow(y[i] - predictions[i], 2);
        }
        
        return 1 - (ssRes / ssTot);
    }
    
    private double[][] subset(double[][] X, int[] indices) {
        double[][] result = new double[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            result[i] = X[indices[i]];
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
    
    // Getters
    public Object getBestModel() { return bestModel; }
    public Map<String, Object> getBestParams() { return bestParams != null ? new HashMap<>(bestParams) : null; }
    public double getBestScore() { return bestScore; }
    public List<Map<String, Object>> getCvResults() { return cvResults != null ? new ArrayList<>(cvResults) : null; }
    public boolean isFitted() { return fitted; }
    
    @SuppressWarnings("unchecked")
    public <T> T getBestModelAs() {
        return (T) bestModel;
    }
}
