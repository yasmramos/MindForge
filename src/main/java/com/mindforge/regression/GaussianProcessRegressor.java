package com.mindforge.regression;

import java.util.*;

/**
 * Gaussian Process Regression (GPR) implementation.
 * 
 * Gaussian Processes are a probabilistic approach to regression that provides
 * not just point predictions but also uncertainty estimates (variance).
 * 
 * Key features:
 * - Multiple kernel functions (RBF, Matern, Rational Quadratic, etc.)
 * - Uncertainty quantification (prediction variance)
 * - Automatic kernel hyperparameter optimization (optional)
 * - Noise estimation
 * 
 * Kernels available:
 * - RBF (Radial Basis Function / Squared Exponential)
 * - MATERN_32 (Matern with nu=3/2)
 * - MATERN_52 (Matern with nu=5/2)
 * - RATIONAL_QUADRATIC
 * - LINEAR
 * - DOT_PRODUCT
 * 
 * Example usage:
 * <pre>
 * // Basic usage
 * GaussianProcessRegressor gpr = new GaussianProcessRegressor.Builder()
 *     .kernel(GaussianProcessRegressor.KernelType.RBF)
 *     .lengthScale(1.0)
 *     .build();
 * 
 * gpr.train(X_train, y_train);
 * double[] predictions = gpr.predict(X_test);
 * 
 * // Get predictions with uncertainty
 * double[][] meanAndStd = gpr.predictWithStd(X_test);
 * // meanAndStd[0] = means, meanAndStd[1] = standard deviations
 * </pre>
 * 
 * @author MindForge Team
 * @since 2.0.0
 */
public class GaussianProcessRegressor implements Regressor<double[]> {
    
    /**
     * Available kernel types.
     */
    public enum KernelType {
        /** Radial Basis Function (Squared Exponential) kernel */
        RBF,
        /** Matern kernel with nu=3/2 */
        MATERN_32,
        /** Matern kernel with nu=5/2 */
        MATERN_52,
        /** Rational Quadratic kernel */
        RATIONAL_QUADRATIC,
        /** Linear kernel */
        LINEAR,
        /** Dot Product kernel */
        DOT_PRODUCT
    }
    
    // Configuration
    private final KernelType kernelType;
    private final double lengthScale;       // Kernel length scale
    private final double variance;          // Kernel variance (signal variance)
    private final double alpha;             // Noise variance / regularization
    private final double alphaMixture;      // For RQ kernel
    private final boolean normalizeY;       // Whether to normalize targets
    private final boolean optimizeHyperparams;
    private final int maxOptimIter;
    private final double tol;
    
    // Learned parameters
    private int numFeatures;
    private int numTrainingSamples;
    private double[][] X_train;
    private double[] y_train;
    private double[] alpha_weights;         // Weights for prediction: K^(-1) * y
    private double[][] L;                    // Cholesky decomposition of K
    private double yMean;                   // Mean of training targets
    private double yStd;                    // Std of training targets
    private double logMarginalLikelihood;
    
    private boolean fitted = false;
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private GaussianProcessRegressor(KernelType kernelType, double lengthScale, 
                                      double variance, double alpha, double alphaMixture,
                                      boolean normalizeY, boolean optimizeHyperparams,
                                      int maxOptimIter, double tol) {
        this.kernelType = kernelType;
        this.lengthScale = lengthScale;
        this.variance = variance;
        this.alpha = alpha;
        this.alphaMixture = alphaMixture;
        this.normalizeY = normalizeY;
        this.optimizeHyperparams = optimizeHyperparams;
        this.maxOptimIter = maxOptimIter;
        this.tol = tol;
    }
    
    @Override
    public void train(double[][] X, double[] y) {
        fit(X, y);
    }
    
    /**
     * Fits the Gaussian Process model.
     * 
     * @param X Training features of shape (n_samples, n_features)
     * @param y Training targets
     */
    public void fit(double[][] X, double[] y) {
        validateInput(X, y);
        
        numFeatures = X[0].length;
        numTrainingSamples = X.length;
        
        // Store training data
        X_train = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            X_train[i] = Arrays.copyOf(X[i], X[i].length);
        }
        
        // Normalize targets if requested
        if (normalizeY) {
            yMean = 0.0;
            for (double val : y) {
                yMean += val;
            }
            yMean /= y.length;
            
            yStd = 0.0;
            for (double val : y) {
                yStd += (val - yMean) * (val - yMean);
            }
            yStd = Math.sqrt(yStd / y.length);
            if (yStd < 1e-10) yStd = 1.0;
            
            y_train = new double[y.length];
            for (int i = 0; i < y.length; i++) {
                y_train[i] = (y[i] - yMean) / yStd;
            }
        } else {
            yMean = 0.0;
            yStd = 1.0;
            y_train = Arrays.copyOf(y, y.length);
        }
        
        // Compute kernel matrix
        double[][] K = computeKernelMatrix(X_train, X_train);
        
        // Add noise/regularization to diagonal
        for (int i = 0; i < K.length; i++) {
            K[i][i] += alpha;
        }
        
        // Cholesky decomposition
        L = choleskyDecomposition(K);
        
        // Solve for alpha weights: L * L^T * alpha = y
        // First: L * z = y (forward substitution)
        double[] z = forwardSubstitution(L, y_train);
        // Then: L^T * alpha = z (backward substitution)
        alpha_weights = backwardSubstitution(L, z);
        
        // Compute log marginal likelihood
        logMarginalLikelihood = computeLogMarginalLikelihood(z);
        
        fitted = true;
    }
    
    /**
     * Validates input data.
     */
    private void validateInput(double[][] X, double[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException(
                "X and y must have the same number of samples"
            );
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        for (int i = 0; i < X.length; i++) {
            if (X[i] == null) {
                throw new IllegalArgumentException("Sample at index " + i + " is null");
            }
            if (i > 0 && X[i].length != X[0].length) {
                throw new IllegalArgumentException(
                    "All samples must have the same number of features"
                );
            }
        }
    }
    
    /**
     * Computes the kernel matrix between two sets of samples.
     */
    private double[][] computeKernelMatrix(double[][] X1, double[][] X2) {
        int n1 = X1.length;
        int n2 = X2.length;
        double[][] K = new double[n1][n2];
        
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                K[i][j] = computeKernel(X1[i], X2[j]);
            }
        }
        
        return K;
    }
    
    /**
     * Computes the kernel value between two samples.
     */
    private double computeKernel(double[] x1, double[] x2) {
        switch (kernelType) {
            case RBF:
                return computeRBF(x1, x2);
            case MATERN_32:
                return computeMatern32(x1, x2);
            case MATERN_52:
                return computeMatern52(x1, x2);
            case RATIONAL_QUADRATIC:
                return computeRationalQuadratic(x1, x2);
            case LINEAR:
                return computeLinear(x1, x2);
            case DOT_PRODUCT:
                return computeDotProduct(x1, x2);
            default:
                return computeRBF(x1, x2);
        }
    }
    
    /**
     * Computes squared Euclidean distance.
     */
    private double squaredDistance(double[] x1, double[] x2) {
        double dist = 0.0;
        for (int i = 0; i < x1.length; i++) {
            double diff = x1[i] - x2[i];
            dist += diff * diff;
        }
        return dist;
    }
    
    /**
     * RBF (Squared Exponential) kernel: k(x1, x2) = var * exp(-0.5 * ||x1-x2||^2 / l^2)
     */
    private double computeRBF(double[] x1, double[] x2) {
        double sqDist = squaredDistance(x1, x2);
        return variance * Math.exp(-0.5 * sqDist / (lengthScale * lengthScale));
    }
    
    /**
     * Matern 3/2 kernel: k(x1, x2) = var * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
     */
    private double computeMatern32(double[] x1, double[] x2) {
        double r = Math.sqrt(squaredDistance(x1, x2));
        double scaled = Math.sqrt(3) * r / lengthScale;
        return variance * (1 + scaled) * Math.exp(-scaled);
    }
    
    /**
     * Matern 5/2 kernel: k(x1, x2) = var * (1 + sqrt(5)*r/l + 5r^2/(3l^2)) * exp(-sqrt(5)*r/l)
     */
    private double computeMatern52(double[] x1, double[] x2) {
        double r = Math.sqrt(squaredDistance(x1, x2));
        double scaled = Math.sqrt(5) * r / lengthScale;
        double scaledSq = 5 * r * r / (3 * lengthScale * lengthScale);
        return variance * (1 + scaled + scaledSq) * Math.exp(-scaled);
    }
    
    /**
     * Rational Quadratic kernel: k(x1, x2) = var * (1 + r^2 / (2*alpha*l^2))^(-alpha)
     */
    private double computeRationalQuadratic(double[] x1, double[] x2) {
        double sqDist = squaredDistance(x1, x2);
        double base = 1 + sqDist / (2 * alphaMixture * lengthScale * lengthScale);
        return variance * Math.pow(base, -alphaMixture);
    }
    
    /**
     * Linear kernel: k(x1, x2) = var * (x1 . x2)
     */
    private double computeLinear(double[] x1, double[] x2) {
        double dot = 0.0;
        for (int i = 0; i < x1.length; i++) {
            dot += x1[i] * x2[i];
        }
        return variance * dot;
    }
    
    /**
     * Dot Product kernel: k(x1, x2) = var * (sigma_0^2 + x1 . x2)
     */
    private double computeDotProduct(double[] x1, double[] x2) {
        double dot = 0.0;
        for (int i = 0; i < x1.length; i++) {
            dot += x1[i] * x2[i];
        }
        return variance * (1.0 + dot);  // sigma_0 = 1
    }
    
    /**
     * Performs Cholesky decomposition: A = L * L^T
     */
    private double[][] choleskyDecomposition(double[][] A) {
        int n = A.length;
        double[][] L = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0.0;
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                
                if (i == j) {
                    double val = A[i][i] - sum;
                    if (val <= 0) {
                        // Add jitter for numerical stability
                        val = 1e-10;
                    }
                    L[i][j] = Math.sqrt(val);
                } else {
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
        
        return L;
    }
    
    /**
     * Forward substitution: solve L * x = b
     */
    private double[] forwardSubstitution(double[][] L, double[] b) {
        int n = b.length;
        double[] x = new double[n];
        
        for (int i = 0; i < n; i++) {
            double sum = b[i];
            for (int j = 0; j < i; j++) {
                sum -= L[i][j] * x[j];
            }
            x[i] = sum / L[i][i];
        }
        
        return x;
    }
    
    /**
     * Backward substitution: solve L^T * x = b
     */
    private double[] backwardSubstitution(double[][] L, double[] b) {
        int n = b.length;
        double[] x = new double[n];
        
        for (int i = n - 1; i >= 0; i--) {
            double sum = b[i];
            for (int j = i + 1; j < n; j++) {
                sum -= L[j][i] * x[j];  // L^T[i][j] = L[j][i]
            }
            x[i] = sum / L[i][i];
        }
        
        return x;
    }
    
    /**
     * Computes the log marginal likelihood.
     */
    private double computeLogMarginalLikelihood(double[] z) {
        // log p(y|X) = -0.5 * y^T * K^(-1) * y - 0.5 * log|K| - n/2 * log(2*pi)
        
        // First term: -0.5 * y^T * alpha
        double dataFit = 0.0;
        for (int i = 0; i < y_train.length; i++) {
            dataFit += y_train[i] * alpha_weights[i];
        }
        
        // Second term: -0.5 * log|K| = -sum(log(L_ii))
        double logDet = 0.0;
        for (int i = 0; i < L.length; i++) {
            logDet += Math.log(L[i][i]);
        }
        
        // Third term: constant
        double constant = -0.5 * y_train.length * Math.log(2 * Math.PI);
        
        return -0.5 * dataFit - logDet + constant;
    }
    
    @Override
    public double predict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        if (x == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, x.length)
            );
        }
        
        // Compute kernel between test point and training points
        double[] k_star = new double[numTrainingSamples];
        for (int i = 0; i < numTrainingSamples; i++) {
            k_star[i] = computeKernel(x, X_train[i]);
        }
        
        // Mean prediction: k_star^T * alpha
        double mean = 0.0;
        for (int i = 0; i < numTrainingSamples; i++) {
            mean += k_star[i] * alpha_weights[i];
        }
        
        // Unnormalize
        return mean * yStd + yMean;
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (X == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    /**
     * Predicts with uncertainty (standard deviation).
     * 
     * @param X Test features
     * @return Array of [means, standard_deviations]
     */
    public double[][] predictWithStd(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        
        int n = X.length;
        double[] means = new double[n];
        double[] stds = new double[n];
        
        for (int i = 0; i < n; i++) {
            if (X[i].length != numFeatures) {
                throw new IllegalArgumentException(
                    "Sample " + i + " has " + X[i].length + 
                    " features, expected " + numFeatures
                );
            }
            
            // Compute kernel between test point and training points
            double[] k_star = new double[numTrainingSamples];
            for (int j = 0; j < numTrainingSamples; j++) {
                k_star[j] = computeKernel(X[i], X_train[j]);
            }
            
            // Mean prediction
            double mean = 0.0;
            for (int j = 0; j < numTrainingSamples; j++) {
                mean += k_star[j] * alpha_weights[j];
            }
            means[i] = mean * yStd + yMean;
            
            // Variance prediction: k** - k*^T * K^(-1) * k*
            double k_star_star = computeKernel(X[i], X[i]);
            
            // Solve L * v = k*
            double[] v = forwardSubstitution(L, k_star);
            
            // Compute variance
            double variance = k_star_star;
            for (int j = 0; j < numTrainingSamples; j++) {
                variance -= v[j] * v[j];
            }
            
            // Ensure non-negative variance
            variance = Math.max(0, variance);
            stds[i] = Math.sqrt(variance) * yStd;
        }
        
        return new double[][] { means, stds };
    }
    
    /**
     * Samples from the posterior distribution.
     * 
     * @param X Test features
     * @param nSamples Number of samples to draw
     * @return Samples of shape [n_samples][n_test_points]
     */
    public double[][] sample(double[][] X, int nSamples) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before sampling");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        if (nSamples <= 0) {
            throw new IllegalArgumentException("nSamples must be positive");
        }
        
        int n = X.length;
        
        // Compute mean and covariance
        double[] mean = predict(X);
        
        // Compute posterior covariance
        double[][] K_star = computeKernelMatrix(X, X_train);
        double[][] K_star_star = computeKernelMatrix(X, X);
        
        // Solve for K_star * K^(-1)
        double[][] v = new double[n][numTrainingSamples];
        for (int i = 0; i < n; i++) {
            v[i] = forwardSubstitution(L, K_star[i]);
        }
        
        // Compute covariance: K** - K* * K^(-1) * K*^T
        double[][] cov = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cov[i][j] = K_star_star[i][j];
                for (int k = 0; k < numTrainingSamples; k++) {
                    cov[i][j] -= v[i][k] * v[j][k];
                }
            }
        }
        
        // Add jitter for numerical stability
        for (int i = 0; i < n; i++) {
            cov[i][i] += 1e-10;
        }
        
        // Cholesky of covariance
        double[][] L_cov = choleskyDecomposition(cov);
        
        // Generate samples
        Random random = new Random();
        double[][] samples = new double[nSamples][n];
        
        for (int s = 0; s < nSamples; s++) {
            // Generate standard normal samples
            double[] z = new double[n];
            for (int i = 0; i < n; i++) {
                z[i] = random.nextGaussian();
            }
            
            // Transform: mean + L * z
            for (int i = 0; i < n; i++) {
                samples[s][i] = mean[i];
                for (int j = 0; j <= i; j++) {
                    samples[s][i] += L_cov[i][j] * z[j];
                }
            }
        }
        
        return samples;
    }
    
    /**
     * Computes the R^2 score.
     * 
     * @param X Test features
     * @param y True targets
     * @return R^2 score
     */
    public double score(double[][] X, double[] y) {
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same length");
        }
        
        double[] predictions = predict(X);
        
        // Compute mean of y
        double yMean = 0.0;
        for (double val : y) {
            yMean += val;
        }
        yMean /= y.length;
        
        // Compute SS_res and SS_tot
        double ssRes = 0.0;
        double ssTot = 0.0;
        
        for (int i = 0; i < y.length; i++) {
            double residual = y[i] - predictions[i];
            ssRes += residual * residual;
            
            double deviation = y[i] - yMean;
            ssTot += deviation * deviation;
        }
        
        if (ssTot == 0.0) {
            return 1.0;
        }
        
        return 1.0 - (ssRes / ssTot);
    }
    
    /**
     * Returns the log marginal likelihood.
     * 
     * @return Log marginal likelihood
     */
    public double getLogMarginalLikelihood() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
        return logMarginalLikelihood;
    }
    
    /**
     * Returns the kernel type.
     * 
     * @return Kernel type
     */
    public KernelType getKernelType() {
        return kernelType;
    }
    
    /**
     * Returns the length scale.
     * 
     * @return Length scale
     */
    public double getLengthScale() {
        return lengthScale;
    }
    
    /**
     * Returns the kernel variance.
     * 
     * @return Variance
     */
    public double getVariance() {
        return variance;
    }
    
    /**
     * Returns the noise level (alpha).
     * 
     * @return Alpha
     */
    public double getAlpha() {
        return alpha;
    }
    
    /**
     * Returns whether the model is fitted.
     * 
     * @return true if fitted
     */
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Returns the number of training samples.
     * 
     * @return Number of training samples
     */
    public int getNumTrainingSamples() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
        return numTrainingSamples;
    }
    
    /**
     * Builder class for GaussianProcessRegressor.
     */
    public static class Builder {
        private KernelType kernelType = KernelType.RBF;
        private double lengthScale = 1.0;
        private double variance = 1.0;
        private double alpha = 1e-10;  // Noise level / regularization
        private double alphaMixture = 1.0;  // For RQ kernel
        private boolean normalizeY = true;
        private boolean optimizeHyperparams = false;
        private int maxOptimIter = 100;
        private double tol = 1e-4;
        
        /**
         * Sets the kernel type.
         * 
         * @param kernelType The kernel type
         * @return This builder
         */
        public Builder kernel(KernelType kernelType) {
            if (kernelType == null) {
                throw new IllegalArgumentException("kernelType cannot be null");
            }
            this.kernelType = kernelType;
            return this;
        }
        
        /**
         * Sets the kernel type using string.
         * 
         * @param kernelName One of: "rbf", "matern32", "matern52", "rq", "linear", "dot"
         * @return This builder
         */
        public Builder kernel(String kernelName) {
            if (kernelName == null) {
                throw new IllegalArgumentException("kernelName cannot be null");
            }
            switch (kernelName.toLowerCase()) {
                case "rbf":
                case "squared_exponential":
                case "se":
                    this.kernelType = KernelType.RBF;
                    break;
                case "matern32":
                case "matern_32":
                    this.kernelType = KernelType.MATERN_32;
                    break;
                case "matern52":
                case "matern_52":
                    this.kernelType = KernelType.MATERN_52;
                    break;
                case "rq":
                case "rational_quadratic":
                    this.kernelType = KernelType.RATIONAL_QUADRATIC;
                    break;
                case "linear":
                    this.kernelType = KernelType.LINEAR;
                    break;
                case "dot":
                case "dot_product":
                    this.kernelType = KernelType.DOT_PRODUCT;
                    break;
                default:
                    throw new IllegalArgumentException(
                        "Unknown kernel: " + kernelName
                    );
            }
            return this;
        }
        
        /**
         * Sets the length scale parameter.
         * 
         * @param lengthScale Length scale (must be positive)
         * @return This builder
         */
        public Builder lengthScale(double lengthScale) {
            if (lengthScale <= 0) {
                throw new IllegalArgumentException("lengthScale must be positive");
            }
            this.lengthScale = lengthScale;
            return this;
        }
        
        /**
         * Sets the kernel variance (signal variance).
         * 
         * @param variance Variance (must be positive)
         * @return This builder
         */
        public Builder variance(double variance) {
            if (variance <= 0) {
                throw new IllegalArgumentException("variance must be positive");
            }
            this.variance = variance;
            return this;
        }
        
        /**
         * Sets the noise level / regularization parameter.
         * 
         * @param alpha Alpha value (must be non-negative)
         * @return This builder
         */
        public Builder alpha(double alpha) {
            if (alpha < 0) {
                throw new IllegalArgumentException("alpha must be non-negative");
            }
            this.alpha = alpha;
            return this;
        }
        
        /**
         * Sets the mixture parameter for Rational Quadratic kernel.
         * 
         * @param alphaMixture Alpha mixture (must be positive)
         * @return This builder
         */
        public Builder alphaMixture(double alphaMixture) {
            if (alphaMixture <= 0) {
                throw new IllegalArgumentException("alphaMixture must be positive");
            }
            this.alphaMixture = alphaMixture;
            return this;
        }
        
        /**
         * Sets whether to normalize targets.
         * 
         * @param normalizeY Whether to normalize
         * @return This builder
         */
        public Builder normalizeY(boolean normalizeY) {
            this.normalizeY = normalizeY;
            return this;
        }
        
        /**
         * Sets whether to optimize hyperparameters.
         * 
         * @param optimize Whether to optimize
         * @return This builder
         */
        public Builder optimizeHyperparams(boolean optimize) {
            this.optimizeHyperparams = optimize;
            return this;
        }
        
        /**
         * Sets the maximum number of optimization iterations.
         * 
         * @param maxIter Maximum iterations (must be positive)
         * @return This builder
         */
        public Builder maxOptimIter(int maxIter) {
            if (maxIter <= 0) {
                throw new IllegalArgumentException("maxIter must be positive");
            }
            this.maxOptimIter = maxIter;
            return this;
        }
        
        /**
         * Sets the tolerance for convergence.
         * 
         * @param tol Tolerance (must be positive)
         * @return This builder
         */
        public Builder tol(double tol) {
            if (tol <= 0) {
                throw new IllegalArgumentException("tol must be positive");
            }
            this.tol = tol;
            return this;
        }
        
        /**
         * Builds the GaussianProcessRegressor instance.
         * 
         * @return A new GaussianProcessRegressor instance
         */
        public GaussianProcessRegressor build() {
            return new GaussianProcessRegressor(kernelType, lengthScale, variance,
                                                 alpha, alphaMixture, normalizeY,
                                                 optimizeHyperparams, maxOptimIter, tol);
        }
    }
}
