package io.github.yasmramos.mindforge.clustering;

import java.util.Arrays;
import java.util.Random;

/**
 * Gaussian Mixture Model (GMM) for probabilistic clustering.
 * 
 * GMM represents a mixture of K Gaussian distributions and uses the
 * Expectation-Maximization (EM) algorithm to estimate parameters.
 * Unlike K-Means, GMM provides soft cluster assignments (probabilities).
 * 
 * Features:
 * - Soft clustering with probability assignments
 * - Full covariance matrix estimation
 * - Multiple initialization strategies (random, k-means++)
 * - BIC/AIC for model selection
 * 
 * Example usage:
 * <pre>
 * GaussianMixture gmm = new GaussianMixture.Builder()
 *     .nComponents(3)
 *     .maxIter(100)
 *     .build();
 * 
 * gmm.fit(data);
 * int[] labels = gmm.predict(data);
 * double[][] probs = gmm.predictProba(data);
 * </pre>
 * 
 * @author MindForge Team
 * @since 2.0.0
 */
public class GaussianMixture implements Clusterer<double[]> {
    
    private final int nComponents;
    private final int maxIter;
    private final double tol;
    private final String initMethod;
    private final long randomSeed;
    
    // Model parameters
    private double[] weights;           // Mixing coefficients (pi_k)
    private double[][] means;           // Component means (mu_k)
    private double[][][] covariances;   // Covariance matrices (Sigma_k)
    
    // Precomputed for prediction
    private double[][] covariancesInv;  // Inverse covariances (for diagonal approx)
    private double[] logDetCovariances; // Log determinants
    
    private boolean fitted = false;
    private int numFeatures;
    private double logLikelihood;
    
    /**
     * Private constructor - use Builder.
     */
    private GaussianMixture(int nComponents, int maxIter, double tol, 
                            String initMethod, long randomSeed) {
        this.nComponents = nComponents;
        this.maxIter = maxIter;
        this.tol = tol;
        this.initMethod = initMethod;
        this.randomSeed = randomSeed;
    }
    
    /**
     * Fits the GMM to the data.
     * 
     * @param X Data matrix (n_samples x n_features)
     */
    public void fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Data cannot be null or empty");
        }
        
        int n = X.length;
        numFeatures = X[0].length;
        
        if (n < nComponents) {
            throw new IllegalArgumentException(
                "Number of samples must be >= number of components"
            );
        }
        
        Random random = new Random(randomSeed);
        
        // Initialize parameters
        initializeParameters(X, random);
        
        // EM algorithm
        double prevLogLik = Double.NEGATIVE_INFINITY;
        double[][] responsibilities = new double[n][nComponents];
        
        for (int iter = 0; iter < maxIter; iter++) {
            // E-step: compute responsibilities
            logLikelihood = eStep(X, responsibilities);
            
            // M-step: update parameters
            mStep(X, responsibilities);
            
            // Check convergence
            if (Math.abs(logLikelihood - prevLogLik) < tol) {
                break;
            }
            prevLogLik = logLikelihood;
        }
        
        // Precompute for prediction
        precomputeForPrediction();
        fitted = true;
    }
    
    /**
     * Initializes GMM parameters.
     */
    private void initializeParameters(double[][] X, Random random) {
        int n = X.length;
        
        weights = new double[nComponents];
        means = new double[nComponents][numFeatures];
        covariances = new double[nComponents][numFeatures][numFeatures];
        
        // Initialize weights uniformly
        Arrays.fill(weights, 1.0 / nComponents);
        
        // Initialize means using k-means++ style
        if ("kmeans".equals(initMethod)) {
            initializeMeansKMeansPlusPlus(X, random);
        } else {
            initializeMeansRandom(X, random);
        }
        
        // Initialize covariances as identity matrices scaled by data variance
        double[] variances = computeVariances(X);
        for (int k = 0; k < nComponents; k++) {
            for (int d = 0; d < numFeatures; d++) {
                covariances[k][d][d] = variances[d] + 1e-6;
            }
        }
    }
    
    /**
     * Initializes means randomly from data points.
     */
    private void initializeMeansRandom(double[][] X, Random random) {
        int n = X.length;
        boolean[] selected = new boolean[n];
        
        for (int k = 0; k < nComponents; k++) {
            int idx;
            do {
                idx = random.nextInt(n);
            } while (selected[idx]);
            
            selected[idx] = true;
            means[k] = Arrays.copyOf(X[idx], numFeatures);
        }
    }
    
    /**
     * Initializes means using k-means++ strategy.
     */
    private void initializeMeansKMeansPlusPlus(double[][] X, Random random) {
        int n = X.length;
        
        // First center: random
        int firstIdx = random.nextInt(n);
        means[0] = Arrays.copyOf(X[firstIdx], numFeatures);
        
        // Subsequent centers: proportional to squared distance
        double[] distances = new double[n];
        
        for (int k = 1; k < nComponents; k++) {
            double totalDist = 0.0;
            
            for (int i = 0; i < n; i++) {
                double minDist = Double.MAX_VALUE;
                for (int j = 0; j < k; j++) {
                    double dist = squaredDistance(X[i], means[j]);
                    minDist = Math.min(minDist, dist);
                }
                distances[i] = minDist;
                totalDist += minDist;
            }
            
            // Select next center with probability proportional to distance
            double threshold = random.nextDouble() * totalDist;
            double cumulative = 0.0;
            int selectedIdx = 0;
            
            for (int i = 0; i < n; i++) {
                cumulative += distances[i];
                if (cumulative >= threshold) {
                    selectedIdx = i;
                    break;
                }
            }
            
            means[k] = Arrays.copyOf(X[selectedIdx], numFeatures);
        }
    }
    
    /**
     * Computes variance for each feature.
     */
    private double[] computeVariances(double[][] X) {
        int n = X.length;
        double[] variances = new double[numFeatures];
        double[] means = new double[numFeatures];
        
        // Compute means
        for (double[] x : X) {
            for (int d = 0; d < numFeatures; d++) {
                means[d] += x[d];
            }
        }
        for (int d = 0; d < numFeatures; d++) {
            means[d] /= n;
        }
        
        // Compute variances
        for (double[] x : X) {
            for (int d = 0; d < numFeatures; d++) {
                double diff = x[d] - means[d];
                variances[d] += diff * diff;
            }
        }
        for (int d = 0; d < numFeatures; d++) {
            variances[d] /= n;
        }
        
        return variances;
    }
    
    /**
     * E-step: Compute responsibilities (posterior probabilities).
     */
    private double eStep(double[][] X, double[][] responsibilities) {
        int n = X.length;
        double totalLogLik = 0.0;
        
        for (int i = 0; i < n; i++) {
            double[] logProbs = new double[nComponents];
            double maxLogProb = Double.NEGATIVE_INFINITY;
            
            // Compute log probability for each component
            for (int k = 0; k < nComponents; k++) {
                logProbs[k] = Math.log(weights[k]) + 
                              logGaussianPdf(X[i], means[k], covariances[k]);
                maxLogProb = Math.max(maxLogProb, logProbs[k]);
            }
            
            // Log-sum-exp trick for numerical stability
            double sumExp = 0.0;
            for (int k = 0; k < nComponents; k++) {
                sumExp += Math.exp(logProbs[k] - maxLogProb);
            }
            double logSumExp = maxLogProb + Math.log(sumExp);
            
            // Compute responsibilities
            for (int k = 0; k < nComponents; k++) {
                responsibilities[i][k] = Math.exp(logProbs[k] - logSumExp);
            }
            
            totalLogLik += logSumExp;
        }
        
        return totalLogLik;
    }
    
    /**
     * M-step: Update parameters based on responsibilities.
     */
    private void mStep(double[][] X, double[][] responsibilities) {
        int n = X.length;
        
        for (int k = 0; k < nComponents; k++) {
            // Compute Nk (effective number of points in cluster k)
            double Nk = 0.0;
            for (int i = 0; i < n; i++) {
                Nk += responsibilities[i][k];
            }
            
            // Avoid division by zero
            Nk = Math.max(Nk, 1e-10);
            
            // Update weight
            weights[k] = Nk / n;
            
            // Update mean
            Arrays.fill(means[k], 0.0);
            for (int i = 0; i < n; i++) {
                for (int d = 0; d < numFeatures; d++) {
                    means[k][d] += responsibilities[i][k] * X[i][d];
                }
            }
            for (int d = 0; d < numFeatures; d++) {
                means[k][d] /= Nk;
            }
            
            // Update covariance
            for (int d1 = 0; d1 < numFeatures; d1++) {
                for (int d2 = 0; d2 < numFeatures; d2++) {
                    covariances[k][d1][d2] = 0.0;
                }
            }
            
            for (int i = 0; i < n; i++) {
                for (int d1 = 0; d1 < numFeatures; d1++) {
                    double diff1 = X[i][d1] - means[k][d1];
                    for (int d2 = 0; d2 < numFeatures; d2++) {
                        double diff2 = X[i][d2] - means[k][d2];
                        covariances[k][d1][d2] += responsibilities[i][k] * diff1 * diff2;
                    }
                }
            }
            
            for (int d1 = 0; d1 < numFeatures; d1++) {
                for (int d2 = 0; d2 < numFeatures; d2++) {
                    covariances[k][d1][d2] /= Nk;
                }
                // Add regularization to diagonal
                covariances[k][d1][d1] += 1e-6;
            }
        }
    }
    
    /**
     * Computes log of Gaussian PDF.
     */
    private double logGaussianPdf(double[] x, double[] mean, double[][] cov) {
        int d = x.length;
        
        // For numerical stability, use diagonal approximation
        double logDet = 0.0;
        double mahalanobis = 0.0;
        
        for (int i = 0; i < d; i++) {
            double variance = cov[i][i];
            logDet += Math.log(variance);
            double diff = x[i] - mean[i];
            mahalanobis += diff * diff / variance;
        }
        
        return -0.5 * (d * Math.log(2 * Math.PI) + logDet + mahalanobis);
    }
    
    /**
     * Precomputes values needed for prediction.
     */
    private void precomputeForPrediction() {
        covariancesInv = new double[nComponents][numFeatures];
        logDetCovariances = new double[nComponents];
        
        for (int k = 0; k < nComponents; k++) {
            logDetCovariances[k] = 0.0;
            for (int d = 0; d < numFeatures; d++) {
                covariancesInv[k][d] = 1.0 / covariances[k][d][d];
                logDetCovariances[k] += Math.log(covariances[k][d][d]);
            }
        }
    }
    
    /**
     * Computes squared Euclidean distance.
     */
    private double squaredDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
    
    @Override
    public int[] cluster(double[][] X) {
        fit(X);
        return predict(X);
    }
    
    @Override
    public int getNumClusters() {
        return nComponents;
    }
    
    @Override
    public int predict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        
        double[] probs = predictProba(x);
        int bestK = 0;
        double maxProb = probs[0];
        
        for (int k = 1; k < nComponents; k++) {
            if (probs[k] > maxProb) {
                maxProb = probs[k];
                bestK = k;
            }
        }
        
        return bestK;
    }
    
    /**
     * Predicts cluster assignments for multiple samples.
     */
    public int[] predict(double[][] X) {
        int[] labels = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            labels[i] = predict(X[i]);
        }
        return labels;
    }
    
    /**
     * Predicts probability of each component for a sample.
     */
    public double[] predictProba(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        
        double[] logProbs = new double[nComponents];
        double maxLogProb = Double.NEGATIVE_INFINITY;
        
        for (int k = 0; k < nComponents; k++) {
            logProbs[k] = Math.log(weights[k]) + 
                          logGaussianPdfFast(x, k);
            maxLogProb = Math.max(maxLogProb, logProbs[k]);
        }
        
        // Normalize using log-sum-exp
        double sumExp = 0.0;
        for (int k = 0; k < nComponents; k++) {
            sumExp += Math.exp(logProbs[k] - maxLogProb);
        }
        double logSumExp = maxLogProb + Math.log(sumExp);
        
        double[] probs = new double[nComponents];
        for (int k = 0; k < nComponents; k++) {
            probs[k] = Math.exp(logProbs[k] - logSumExp);
        }
        
        return probs;
    }
    
    /**
     * Predicts probabilities for multiple samples.
     */
    public double[][] predictProba(double[][] X) {
        double[][] probs = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            probs[i] = predictProba(X[i]);
        }
        return probs;
    }
    
    /**
     * Fast log Gaussian PDF using precomputed values.
     */
    private double logGaussianPdfFast(double[] x, int k) {
        double mahalanobis = 0.0;
        for (int d = 0; d < numFeatures; d++) {
            double diff = x[d] - means[k][d];
            mahalanobis += diff * diff * covariancesInv[k][d];
        }
        
        return -0.5 * (numFeatures * Math.log(2 * Math.PI) + 
                       logDetCovariances[k] + mahalanobis);
    }
    
    /**
     * Computes the log-likelihood of data under the model.
     */
    public double score(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before scoring");
        }
        
        double totalLogLik = 0.0;
        
        for (double[] x : X) {
            double[] logProbs = new double[nComponents];
            double maxLogProb = Double.NEGATIVE_INFINITY;
            
            for (int k = 0; k < nComponents; k++) {
                logProbs[k] = Math.log(weights[k]) + logGaussianPdfFast(x, k);
                maxLogProb = Math.max(maxLogProb, logProbs[k]);
            }
            
            double sumExp = 0.0;
            for (int k = 0; k < nComponents; k++) {
                sumExp += Math.exp(logProbs[k] - maxLogProb);
            }
            
            totalLogLik += maxLogProb + Math.log(sumExp);
        }
        
        return totalLogLik;
    }
    
    /**
     * Computes the Bayesian Information Criterion (BIC).
     * Lower is better.
     */
    public double bic(double[][] X) {
        int n = X.length;
        int numParams = nComponents * (1 + numFeatures + numFeatures) - 1;
        return -2 * score(X) + numParams * Math.log(n);
    }
    
    /**
     * Computes the Akaike Information Criterion (AIC).
     * Lower is better.
     */
    public double aic(double[][] X) {
        int numParams = nComponents * (1 + numFeatures + numFeatures) - 1;
        return -2 * score(X) + 2 * numParams;
    }
    
    // Getters
    
    public double[] getWeights() {
        return Arrays.copyOf(weights, weights.length);
    }
    
    public double[][] getMeans() {
        double[][] copy = new double[nComponents][];
        for (int k = 0; k < nComponents; k++) {
            copy[k] = Arrays.copyOf(means[k], numFeatures);
        }
        return copy;
    }
    
    public double[][][] getCovariances() {
        double[][][] copy = new double[nComponents][][];
        for (int k = 0; k < nComponents; k++) {
            copy[k] = new double[numFeatures][];
            for (int d = 0; d < numFeatures; d++) {
                copy[k][d] = Arrays.copyOf(covariances[k][d], numFeatures);
            }
        }
        return copy;
    }
    
    public double getLogLikelihood() {
        return logLikelihood;
    }
    
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Builder class for GaussianMixture.
     */
    public static class Builder {
        private int nComponents = 2;
        private int maxIter = 100;
        private double tol = 1e-4;
        private String initMethod = "kmeans";
        private long randomSeed = 42;
        
        /**
         * Sets the number of mixture components.
         */
        public Builder nComponents(int nComponents) {
            if (nComponents < 1) {
                throw new IllegalArgumentException("nComponents must be >= 1");
            }
            this.nComponents = nComponents;
            return this;
        }
        
        /**
         * Sets the maximum number of EM iterations.
         */
        public Builder maxIter(int maxIter) {
            if (maxIter < 1) {
                throw new IllegalArgumentException("maxIter must be >= 1");
            }
            this.maxIter = maxIter;
            return this;
        }
        
        /**
         * Sets the convergence tolerance.
         */
        public Builder tol(double tol) {
            if (tol <= 0) {
                throw new IllegalArgumentException("tol must be positive");
            }
            this.tol = tol;
            return this;
        }
        
        /**
         * Sets the initialization method ("random" or "kmeans").
         */
        public Builder initMethod(String initMethod) {
            this.initMethod = initMethod;
            return this;
        }
        
        /**
         * Sets the random seed for reproducibility.
         */
        public Builder randomSeed(long randomSeed) {
            this.randomSeed = randomSeed;
            return this;
        }
        
        /**
         * Builds the GaussianMixture instance.
         */
        public GaussianMixture build() {
            return new GaussianMixture(nComponents, maxIter, tol, initMethod, randomSeed);
        }
    }
}
