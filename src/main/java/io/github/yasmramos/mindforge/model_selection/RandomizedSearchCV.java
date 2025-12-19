package io.github.yasmramos.mindforge.model_selection;

import io.github.yasmramos.mindforge.classification.Classifier;
import io.github.yasmramos.mindforge.validation.Metrics;

import java.io.Serializable;
import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Supplier;

/**
 * Randomized search over hyperparameters.
 * More efficient than GridSearchCV for large parameter spaces.
 */
public class RandomizedSearchCV implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int nIter;
    private final int cv;
    private final long randomSeed;
    private Map<String, Object> bestParams;
    private double bestScore;
    private List<Map<String, Object>> cvResults;
    
    public RandomizedSearchCV(int nIter, int cv, long randomSeed) {
        this.nIter = nIter;
        this.cv = cv;
        this.randomSeed = randomSeed;
        this.cvResults = new ArrayList<>();
    }
    
    public RandomizedSearchCV(int nIter, int cv) {
        this(nIter, cv, 42);
    }
    
    public RandomizedSearchCV() {
        this(10, 5, 42);
    }
    
    /**
     * Perform randomized search.
     * @param modelSupplier Function that creates a model given parameters
     * @param paramDistributions Map of parameter name to distribution (List or ParameterSampler)
     * @param X Training features
     * @param y Training labels
     */
    public void fit(BiFunction<Map<String, Object>, double[][], Object> modelSupplier,
                   Map<String, Object> paramDistributions,
                   double[][] X, int[] y) {
        Random random = new Random(randomSeed);
        bestScore = Double.NEGATIVE_INFINITY;
        cvResults.clear();
        
        for (int iter = 0; iter < nIter; iter++) {
            Map<String, Object> params = sampleParams(paramDistributions, random);
            double score = crossValidate(modelSupplier, params, X, y);
            
            Map<String, Object> result = new HashMap<>(params);
            result.put("mean_test_score", score);
            cvResults.add(result);
            
            if (score > bestScore) {
                bestScore = score;
                bestParams = new HashMap<>(params);
            }
        }
    }
    
    private Map<String, Object> sampleParams(Map<String, Object> distributions, Random random) {
        Map<String, Object> params = new HashMap<>();
        for (Map.Entry<String, Object> entry : distributions.entrySet()) {
            String paramName = entry.getKey();
            Object distribution = entry.getValue();
            
            if (distribution instanceof List) {
                List<?> options = (List<?>) distribution;
                params.put(paramName, options.get(random.nextInt(options.size())));
            } else if (distribution instanceof ParameterSampler) {
                params.put(paramName, ((ParameterSampler) distribution).sample(random));
            } else if (distribution instanceof double[]) {
                double[] range = (double[]) distribution;
                params.put(paramName, range[0] + random.nextDouble() * (range[1] - range[0]));
            } else if (distribution instanceof int[]) {
                int[] range = (int[]) distribution;
                params.put(paramName, range[0] + random.nextInt(range[1] - range[0] + 1));
            }
        }
        return params;
    }
    
    private double crossValidate(BiFunction<Map<String, Object>, double[][], Object> modelSupplier,
                                Map<String, Object> params, double[][] X, int[] y) {
        KFold kfold = new KFold(cv, true, new Random(randomSeed));
        List<int[][]> splits = kfold.split(X);
        
        double totalScore = 0;
        for (int[][] split : splits) {
            int[] trainIndices = split[0];
            int[] testIndices = split[1];
            
            double[][] XTrain = new double[trainIndices.length][];
            double[][] XTest = new double[testIndices.length][];
            int[] yTrain = new int[trainIndices.length];
            int[] yTest = new int[testIndices.length];
            
            for (int i = 0; i < trainIndices.length; i++) {
                XTrain[i] = X[trainIndices[i]];
                yTrain[i] = y[trainIndices[i]];
            }
            for (int i = 0; i < testIndices.length; i++) {
                XTest[i] = X[testIndices[i]];
                yTest[i] = y[testIndices[i]];
            }
            
            try {
                Object model = modelSupplier.apply(params, XTrain);
                // Use reflection to call fit and predict
                model.getClass().getMethod("fit", double[][].class, int[].class).invoke(model, XTrain, yTrain);
                int[] predictions = (int[]) model.getClass().getMethod("predict", double[][].class).invoke(model, XTest);
                totalScore += Metrics.accuracy(yTest, predictions);
            } catch (Exception e) {
                totalScore += 0; // Failed configuration
            }
        }
        
        return totalScore / cv;
    }
    
    public Map<String, Object> getBestParams() { return bestParams; }
    public double getBestScore() { return bestScore; }
    public List<Map<String, Object>> getCvResults() { return cvResults; }
    
    /**
     * Interface for custom parameter sampling.
     */
    public interface ParameterSampler {
        Object sample(Random random);
    }
    
    /**
     * Log-uniform distribution sampler.
     */
    public static class LogUniform implements ParameterSampler {
        private final double low;
        private final double high;
        
        public LogUniform(double low, double high) {
            this.low = Math.log(low);
            this.high = Math.log(high);
        }
        
        @Override
        public Object sample(Random random) {
            return Math.exp(low + random.nextDouble() * (high - low));
        }
    }
    
    public static class Builder {
        private int nIter = 10;
        private int cv = 5;
        private long randomSeed = 42;
        
        public Builder nIter(int n) { this.nIter = n; return this; }
        public Builder cv(int c) { this.cv = c; return this; }
        public Builder randomSeed(long s) { this.randomSeed = s; return this; }
        public RandomizedSearchCV build() { return new RandomizedSearchCV(nIter, cv, randomSeed); }
    }
}
