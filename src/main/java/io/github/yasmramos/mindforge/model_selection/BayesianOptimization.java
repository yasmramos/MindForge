package io.github.yasmramos.mindforge.model_selection;

import java.io.Serializable;
import java.util.*;
import java.util.function.Function;

/**
 * Bayesian Optimization for hyperparameter tuning.
 * Uses Gaussian Process surrogate model with Expected Improvement acquisition function.
 */
public class BayesianOptimization implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int nIterations;
    private final int nInitialPoints;
    private final double explorationWeight;
    private final long seed;
    
    private Map<String, double[]> parameterBounds;
    private List<double[]> observedPoints;
    private List<Double> observedValues;
    private double[] bestParams;
    private double bestValue;
    private List<String> paramNames;
    
    private BayesianOptimization(Builder builder) {
        this.nIterations = builder.nIterations;
        this.nInitialPoints = builder.nInitialPoints;
        this.explorationWeight = builder.explorationWeight;
        this.seed = builder.seed;
    }
    
    /**
     * Set parameter search space.
     * @param bounds Map of parameter name to [min, max] bounds
     */
    public void setParameterBounds(Map<String, double[]> bounds) {
        this.parameterBounds = new LinkedHashMap<>(bounds);
        this.paramNames = new ArrayList<>(bounds.keySet());
    }
    
    /**
     * Run optimization.
     * @param objectiveFunction Function that takes parameters and returns score (higher is better)
     * @return Best parameters found
     */
    public Map<String, Double> optimize(Function<Map<String, Double>, Double> objectiveFunction) {
        if (parameterBounds == null || parameterBounds.isEmpty()) {
            throw new IllegalStateException("Parameter bounds must be set before optimization");
        }
        
        Random random = new Random(seed);
        int nDims = parameterBounds.size();
        
        observedPoints = new ArrayList<>();
        observedValues = new ArrayList<>();
        bestValue = Double.NEGATIVE_INFINITY;
        
        // Initial random sampling
        for (int i = 0; i < nInitialPoints; i++) {
            double[] point = sampleRandomPoint(random);
            double value = evaluatePoint(point, objectiveFunction);
            
            observedPoints.add(point);
            observedValues.add(value);
            
            if (value > bestValue) {
                bestValue = value;
                bestParams = point.clone();
            }
        }
        
        // Bayesian optimization loop
        for (int iter = 0; iter < nIterations; iter++) {
            double[] nextPoint = findNextPoint(random);
            double value = evaluatePoint(nextPoint, objectiveFunction);
            
            observedPoints.add(nextPoint);
            observedValues.add(value);
            
            if (value > bestValue) {
                bestValue = value;
                bestParams = nextPoint.clone();
            }
        }
        
        return arrayToMap(bestParams);
    }
    
    private double[] sampleRandomPoint(Random random) {
        double[] point = new double[paramNames.size()];
        for (int i = 0; i < paramNames.size(); i++) {
            double[] bounds = parameterBounds.get(paramNames.get(i));
            point[i] = bounds[0] + random.nextDouble() * (bounds[1] - bounds[0]);
        }
        return point;
    }
    
    private double evaluatePoint(double[] point, Function<Map<String, Double>, Double> objective) {
        return objective.apply(arrayToMap(point));
    }
    
    private Map<String, Double> arrayToMap(double[] point) {
        Map<String, Double> params = new LinkedHashMap<>();
        for (int i = 0; i < paramNames.size(); i++) {
            params.put(paramNames.get(i), point[i]);
        }
        return params;
    }
    
    private double[] findNextPoint(Random random) {
        int nCandidates = 1000;
        double[] bestCandidate = null;
        double bestAcquisition = Double.NEGATIVE_INFINITY;
        
        for (int i = 0; i < nCandidates; i++) {
            double[] candidate = sampleRandomPoint(random);
            double acquisition = computeExpectedImprovement(candidate);
            
            if (acquisition > bestAcquisition) {
                bestAcquisition = acquisition;
                bestCandidate = candidate;
            }
        }
        
        return bestCandidate;
    }
    
    private double computeExpectedImprovement(double[] point) {
        double[] prediction = predictGP(point);
        double mean = prediction[0];
        double std = prediction[1];
        
        if (std < 1e-10) return 0;
        
        double improvement = mean - bestValue - explorationWeight;
        double z = improvement / std;
        
        // Expected improvement: std * (z * Phi(z) + phi(z))
        double phi = normalPDF(z);
        double bigPhi = normalCDF(z);
        
        return std * (z * bigPhi + phi);
    }
    
    private double[] predictGP(double[] point) {
        int n = observedPoints.size();
        if (n == 0) return new double[]{0, 1};
        
        // Compute kernel matrix K and kernel vector k*
        double[][] K = new double[n][n];
        double[] kStar = new double[n];
        
        double lengthScale = estimateLengthScale();
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                K[i][j] = rbfKernel(observedPoints.get(i), observedPoints.get(j), lengthScale);
            }
            K[i][i] += 1e-6; // Noise term for numerical stability
            kStar[i] = rbfKernel(observedPoints.get(i), point, lengthScale);
        }
        
        // Solve K * alpha = y
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = observedValues.get(i);
        }
        
        double[] alpha = solveLinearSystem(K, y);
        
        // Mean prediction: k* . alpha
        double mean = 0;
        for (int i = 0; i < n; i++) {
            mean += kStar[i] * alpha[i];
        }
        
        // Variance: k** - k* . K^-1 . k*
        double kStarStar = rbfKernel(point, point, lengthScale);
        double[] KInvKStar = solveLinearSystem(K, kStar);
        
        double variance = kStarStar;
        for (int i = 0; i < n; i++) {
            variance -= kStar[i] * KInvKStar[i];
        }
        variance = Math.max(variance, 1e-10);
        
        return new double[]{mean, Math.sqrt(variance)};
    }
    
    private double rbfKernel(double[] x1, double[] x2, double lengthScale) {
        double dist = 0;
        for (int i = 0; i < x1.length; i++) {
            double diff = (x1[i] - x2[i]) / lengthScale;
            dist += diff * diff;
        }
        return Math.exp(-0.5 * dist);
    }
    
    private double estimateLengthScale() {
        if (observedPoints.size() < 2) return 1.0;
        
        double sumDist = 0;
        int count = 0;
        for (int i = 0; i < observedPoints.size(); i++) {
            for (int j = i + 1; j < observedPoints.size(); j++) {
                double dist = 0;
                for (int k = 0; k < paramNames.size(); k++) {
                    double[] bounds = parameterBounds.get(paramNames.get(k));
                    double range = bounds[1] - bounds[0];
                    double diff = (observedPoints.get(i)[k] - observedPoints.get(j)[k]) / range;
                    dist += diff * diff;
                }
                sumDist += Math.sqrt(dist);
                count++;
            }
        }
        return sumDist / count;
    }
    
    private double[] solveLinearSystem(double[][] A, double[] b) {
        int n = A.length;
        double[][] augmented = new double[n][n + 1];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = A[i][j];
            }
            augmented[i][n] = b[i];
        }
        
        for (int i = 0; i < n; i++) {
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            double[] temp = augmented[i];
            augmented[i] = augmented[maxRow];
            augmented[maxRow] = temp;
            
            if (Math.abs(augmented[i][i]) < 1e-10) continue;
            
            for (int k = i + 1; k < n; k++) {
                double factor = augmented[k][i] / augmented[i][i];
                for (int j = i; j <= n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
        
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = augmented[i][n];
            for (int j = i + 1; j < n; j++) {
                x[i] -= augmented[i][j] * x[j];
            }
            if (Math.abs(augmented[i][i]) > 1e-10) {
                x[i] /= augmented[i][i];
            }
        }
        
        return x;
    }
    
    private double normalPDF(double x) {
        return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
    }
    
    private double normalCDF(double x) {
        return 0.5 * (1 + erf(x / Math.sqrt(2)));
    }
    
    private double erf(double x) {
        double t = 1.0 / (1.0 + 0.5 * Math.abs(x));
        double tau = t * Math.exp(-x * x - 1.26551223 +
            t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
            t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
            t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? 1 - tau : tau - 1;
    }
    
    // Getters
    public double[] getBestParams() { return bestParams != null ? bestParams.clone() : null; }
    public double getBestValue() { return bestValue; }
    public Map<String, Double> getBestParamsAsMap() { return bestParams != null ? arrayToMap(bestParams) : null; }
    public List<Double> getObservedValues() { return new ArrayList<>(observedValues); }
    
    public static class Builder {
        private int nIterations = 50;
        private int nInitialPoints = 5;
        private double explorationWeight = 0.01;
        private long seed = 42;
        
        public Builder nIterations(int n) { this.nIterations = n; return this; }
        public Builder nInitialPoints(int n) { this.nInitialPoints = n; return this; }
        public Builder explorationWeight(double w) { this.explorationWeight = w; return this; }
        public Builder seed(long seed) { this.seed = seed; return this; }
        
        public BayesianOptimization build() { return new BayesianOptimization(this); }
    }
}
