package com.mindforge.regression;

import com.mindforge.classification.Kernel;
import java.util.*;

/**
 * Support Vector Regression (SVR) implementation.
 * 
 * Implements epsilon-insensitive Support Vector Regression with multiple kernel options:
 * - LINEAR: Linear kernel (default)
 * - RBF: Radial Basis Function (Gaussian) kernel
 * - POLYNOMIAL: Polynomial kernel
 * - SIGMOID: Sigmoid/hyperbolic tangent kernel
 * 
 * Uses epsilon-insensitive loss function (epsilon-tube) where predictions
 * within epsilon of the true value incur no loss. The algorithm finds the
 * flattest function that has at most epsilon deviation from the targets.
 * 
 * Key parameters:
 * - C: Regularization parameter (trade-off between flatness and tolerance)
 * - epsilon: Epsilon in the epsilon-SVR model (width of the tube)
 * - kernel: Kernel function for non-linear regression
 * 
 * Example usage:
 * <pre>
 * // Linear SVR (default)
 * SVR svr = new SVR.Builder()
 *     .C(1.0)
 *     .epsilon(0.1)
 *     .build();
 * 
 * // RBF kernel SVR
 * SVR svr = new SVR.Builder()
 *     .kernel(Kernel.Type.RBF)
 *     .gamma(0.5)
 *     .C(1.0)
 *     .epsilon(0.1)
 *     .build();
 * 
 * svr.train(X_train, y_train);
 * double[] predictions = svr.predict(X_test);
 * </pre>
 * 
 * @author MindForge Team
 * @since 2.0.0
 */
public class SVR implements Regressor<double[]> {
    
    // Hyperparameters
    private final double C;          // Regularization parameter
    private final double epsilon;     // Epsilon in epsilon-SVR model
    private final int maxIter;
    private final double tol;         // Tolerance for stopping criterion
    private final double learningRate;
    
    // Kernel parameters
    private final Kernel kernel;
    
    // Model parameters
    private int numFeatures;
    
    // For linear kernel (primal form)
    private double[] weights;
    private double bias;
    
    // For non-linear kernels (dual form)
    private double[] alphaPositive;   // Alpha for samples above tube
    private double[] alphaNegative;   // Alpha* for samples below tube
    private double[][] supportVectors;
    private double[] supportVectorTargets;
    private double biasKernel;
    
    private boolean fitted = false;
    private final boolean useKernelMethod;
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private SVR(double C, double epsilon, int maxIter, double tol, 
                double learningRate, Kernel kernel) {
        this.C = C;
        this.epsilon = epsilon;
        this.maxIter = maxIter;
        this.tol = tol;
        this.learningRate = learningRate;
        this.kernel = kernel;
        this.useKernelMethod = (kernel.getType() != Kernel.Type.LINEAR);
    }
    
    @Override
    public void train(double[][] X, double[] y) {
        validateInput(X, y);
        
        numFeatures = X[0].length;
        
        if (useKernelMethod) {
            trainWithKernel(X, y);
        } else {
            trainLinear(X, y);
        }
        
        fitted = true;
    }
    
    /**
     * Validates input data for training.
     */
    private void validateInput(double[][] X, double[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException(
                "X and y must have the same number of samples: " + 
                X.length + " vs " + y.length
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
     * Trains using linear kernel with gradient descent (primal form).
     * Uses epsilon-insensitive loss with L2 regularization.
     */
    private void trainLinear(double[][] X, double[] y) {
        int n = X.length;
        
        // Normalize features for better convergence
        double[] featureMean = new double[numFeatures];
        double[] featureStd = new double[numFeatures];
        
        for (int f = 0; f < numFeatures; f++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += X[i][f];
            }
            featureMean[f] = sum / n;
            
            double sumSq = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = X[i][f] - featureMean[f];
                sumSq += diff * diff;
            }
            featureStd[f] = Math.sqrt(sumSq / n);
            if (featureStd[f] < 1e-10) {
                featureStd[f] = 1.0;
            }
        }
        
        // Normalize X
        double[][] Xnorm = new double[n][numFeatures];
        for (int i = 0; i < n; i++) {
            for (int f = 0; f < numFeatures; f++) {
                Xnorm[i][f] = (X[i][f] - featureMean[f]) / featureStd[f];
            }
        }
        
        // Normalize y
        double yMean = 0.0;
        for (double val : y) {
            yMean += val;
        }
        yMean /= n;
        
        double yStd = 0.0;
        for (double val : y) {
            yStd += (val - yMean) * (val - yMean);
        }
        yStd = Math.sqrt(yStd / n);
        if (yStd < 1e-10) {
            yStd = 1.0;
        }
        
        double[] yNorm = new double[n];
        for (int i = 0; i < n; i++) {
            yNorm[i] = (y[i] - yMean) / yStd;
        }
        
        // Initialize weights and bias
        weights = new double[numFeatures];
        bias = 0.0;
        
        // Adaptive learning rate based on problem size
        double adaptiveLr = learningRate * Math.min(1.0, 10.0 / numFeatures);
        
        // Subgradient descent optimization with full batch for stability
        double prevLoss = Double.MAX_VALUE;
        
        for (int iter = 0; iter < maxIter; iter++) {
            double[] gradW = new double[numFeatures];
            double gradB = 0.0;
            double loss = 0.0;
            
            // Full batch gradient for stability in high dimensions
            for (int i = 0; i < n; i++) {
                double prediction = bias;
                for (int f = 0; f < numFeatures; f++) {
                    prediction += weights[f] * Xnorm[i][f];
                }
                double error = prediction - yNorm[i];
                
                // Epsilon-insensitive loss and subgradient
                double scaledEpsilon = epsilon / yStd;
                if (Math.abs(error) > scaledEpsilon) {
                    double sign = error > 0 ? 1.0 : -1.0;
                    loss += Math.abs(error) - scaledEpsilon;
                    
                    for (int f = 0; f < numFeatures; f++) {
                        gradW[f] += sign * Xnorm[i][f];
                    }
                    gradB += sign;
                }
            }
            
            // Normalize gradients
            for (int f = 0; f < numFeatures; f++) {
                gradW[f] /= n;
            }
            gradB /= n;
            
            // Add L2 regularization gradient
            for (int f = 0; f < numFeatures; f++) {
                loss += 0.5 / C * weights[f] * weights[f];
                gradW[f] += (1.0 / C) * weights[f];
            }
            
            // Adaptive learning rate with decay
            double lr = adaptiveLr / (1.0 + 0.001 * iter);
            
            // Update weights and bias
            for (int f = 0; f < numFeatures; f++) {
                weights[f] -= lr * gradW[f];
            }
            bias -= lr * gradB;
            
            // Check convergence
            if (iter > 100 && Math.abs(prevLoss - loss) < tol) {
                break;
            }
            prevLoss = loss;
        }
        
        // Transform weights back to original scale
        for (int f = 0; f < numFeatures; f++) {
            weights[f] = weights[f] * yStd / featureStd[f];
        }
        
        // Adjust bias for original scale
        double biasAdjust = 0.0;
        for (int f = 0; f < numFeatures; f++) {
            biasAdjust += weights[f] * featureMean[f] / (yStd / featureStd[f]) * featureStd[f];
        }
        bias = bias * yStd + yMean;
        for (int f = 0; f < numFeatures; f++) {
            bias -= weights[f] * featureMean[f];
        }
    }
    
    /**
     * Shuffles an array in place.
     */
    private void shuffleArray(int[] array, Random random) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    
    /**
     * Trains using non-linear kernels with SMO-style algorithm (dual form).
     * Optimizes the dual formulation of epsilon-SVR.
     */
    private void trainWithKernel(double[][] X, double[] y) {
        int n = X.length;
        
        // Store training data for prediction
        supportVectors = new double[n][];
        supportVectorTargets = new double[n];
        for (int i = 0; i < n; i++) {
            supportVectors[i] = Arrays.copyOf(X[i], X[i].length);
            supportVectorTargets[i] = y[i];
        }
        
        // Precompute kernel matrix
        double[][] K = kernel.computeMatrix(X);
        
        // Add small regularization to diagonal for numerical stability
        double[] diagK = new double[n];
        for (int i = 0; i < n; i++) {
            diagK[i] = K[i][i];
            K[i][i] += 1e-6;
        }
        
        // Initialize alphas using a combined representation
        // alpha[i] = alphaPositive[i] - alphaNegative[i], range [-C, C]
        alphaPositive = new double[n];
        alphaNegative = new double[n];
        double[] alpha = new double[n];  // Combined alpha
        
        // Initialize gradient (g_i = -y_i initially since f(x_i) = 0)
        double[] gradient = new double[n];
        for (int i = 0; i < n; i++) {
            gradient[i] = -y[i];
        }
        
        biasKernel = 0.0;
        
        Random random = new Random(42);
        
        // Multiple passes for convergence
        for (int iter = 0; iter < maxIter; iter++) {
            double maxViolation = 0.0;
            
            // Shuffle indices for randomized coordinate descent
            int[] indices = new int[n];
            for (int i = 0; i < n; i++) indices[i] = i;
            shuffleArray(indices, random);
            
            for (int idx = 0; idx < n; idx++) {
                int i = indices[idx];
                
                // Current prediction error
                double G = gradient[i];
                
                // Compute optimal step based on epsilon-tube conditions
                // For epsilon-SVR, we need to consider the epsilon-insensitive loss
                
                double oldAlpha = alpha[i];
                double newAlpha = oldAlpha;
                
                // Compute optimal unconstrained update
                // The gradient of the dual objective is: sum_j(alpha_j * K_ij) - y_i + epsilon*sign(alpha_i)
                // For simplicity, we use gradient descent with the current gradient
                
                if (G + epsilon < 0) {
                    // Want to increase alpha (prediction too low)
                    // alpha_i should be positive (alpha+ contribution)
                    double step = -(G + epsilon) / K[i][i];
                    newAlpha = Math.min(C, oldAlpha + step);
                } else if (G - epsilon > 0) {
                    // Want to decrease alpha (prediction too high)
                    // alpha_i should be negative (alpha- contribution)
                    double step = (G - epsilon) / K[i][i];
                    newAlpha = Math.max(-C, oldAlpha - step);
                } else {
                    // Within epsilon tube - shrink alpha towards 0
                    if (oldAlpha > 0) {
                        newAlpha = Math.max(0, oldAlpha - 0.01 * K[i][i]);
                    } else if (oldAlpha < 0) {
                        newAlpha = Math.min(0, oldAlpha + 0.01 * K[i][i]);
                    }
                }
                
                // Apply update if significant
                double deltaAlpha = newAlpha - oldAlpha;
                if (Math.abs(deltaAlpha) > 1e-10) {
                    alpha[i] = newAlpha;
                    
                    // Update gradients for all samples
                    for (int j = 0; j < n; j++) {
                        gradient[j] += deltaAlpha * K[i][j];
                    }
                    
                    maxViolation = Math.max(maxViolation, Math.abs(deltaAlpha));
                }
            }
            
            // Check convergence
            if (maxViolation < tol && iter > 10) {
                break;
            }
        }
        
        // Convert combined alpha back to alpha+ and alpha-
        for (int i = 0; i < n; i++) {
            if (alpha[i] > 0) {
                alphaPositive[i] = alpha[i];
                alphaNegative[i] = 0;
            } else {
                alphaPositive[i] = 0;
                alphaNegative[i] = -alpha[i];
            }
        }
        
        // Compute bias using free support vectors
        computeFinalBias(K);
    }
    
    /**
     * Computes kernel prediction during training using kernel matrix.
     */
    private double computeKernelPredictionTraining(double[][] K, int idx) {
        double prediction = biasKernel;
        for (int i = 0; i < K.length; i++) {
            prediction += (alphaPositive[i] - alphaNegative[i]) * K[i][idx];
        }
        return prediction;
    }
    
    /**
     * Computes the final bias term.
     */
    private void computeFinalBias(double[][] K) {
        int n = supportVectors.length;
        int count = 0;
        double sumBias = 0.0;
        
        // First, try to compute bias from free support vectors
        for (int i = 0; i < n; i++) {
            double alpha = alphaPositive[i] - alphaNegative[i];
            if (Math.abs(alpha) > 1e-8) {
                boolean isFree = (alphaPositive[i] > 1e-8 && alphaPositive[i] < C - 1e-8) ||
                                 (alphaNegative[i] > 1e-8 && alphaNegative[i] < C - 1e-8);
                if (isFree) {
                    double fi = 0.0;
                    for (int j = 0; j < n; j++) {
                        fi += (alphaPositive[j] - alphaNegative[j]) * K[i][j];
                    }
                    
                    if (alphaPositive[i] > 1e-8) {
                        sumBias += supportVectorTargets[i] - fi - epsilon;
                    } else {
                        sumBias += supportVectorTargets[i] - fi + epsilon;
                    }
                    count++;
                }
            }
        }
        
        if (count > 0) {
            biasKernel = sumBias / count;
            return;
        }
        
        // Fallback: compute bias from all support vectors
        for (int i = 0; i < n; i++) {
            double alpha = alphaPositive[i] - alphaNegative[i];
            if (Math.abs(alpha) > 1e-8) {
                double fi = 0.0;
                for (int j = 0; j < n; j++) {
                    fi += (alphaPositive[j] - alphaNegative[j]) * K[i][j];
                }
                sumBias += supportVectorTargets[i] - fi;
                count++;
            }
        }
        
        if (count > 0) {
            biasKernel = sumBias / count;
            return;
        }
        
        // Last resort: compute bias from mean of targets
        double yMean = 0.0;
        for (int i = 0; i < n; i++) {
            yMean += supportVectorTargets[i];
        }
        biasKernel = yMean / n;
    }
    
    /**
     * Computes the linear prediction for a sample.
     */
    private double computeLinearPrediction(double[] x) {
        double prediction = bias;
        for (int f = 0; f < numFeatures; f++) {
            prediction += weights[f] * x[f];
        }
        return prediction;
    }
    
    @Override
    public double predict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, x.length)
            );
        }
        
        if (useKernelMethod) {
            return computeKernelPrediction(x);
        } else {
            return computeLinearPrediction(x);
        }
    }
    
    /**
     * Computes the kernel prediction for a new sample.
     */
    private double computeKernelPrediction(double[] x) {
        double prediction = biasKernel;
        
        for (int i = 0; i < supportVectors.length; i++) {
            double alpha = alphaPositive[i] - alphaNegative[i];
            if (Math.abs(alpha) > 1e-8) {
                prediction += alpha * kernel.compute(supportVectors[i], x);
            }
        }
        
        return prediction;
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
     * Computes the coefficient of determination R^2.
     * 
     * @param X Test features
     * @param y True target values
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
            return 1.0; // Perfect prediction when all y values are the same
        }
        
        return 1.0 - (ssRes / ssTot);
    }
    
    /**
     * Returns the number of support vectors.
     * Only applicable for kernel SVR.
     * 
     * @return Number of support vectors
     */
    public int getNumSupportVectors() {
        if (!useKernelMethod) {
            return 0;
        }
        
        int count = 0;
        for (int i = 0; i < supportVectors.length; i++) {
            if (Math.abs(alphaPositive[i] - alphaNegative[i]) > 1e-8) {
                count++;
            }
        }
        return count;
    }
    
    /**
     * Returns the support vectors.
     * Only applicable for kernel SVR.
     * 
     * @return Support vectors
     */
    public double[][] getSupportVectors() {
        if (!useKernelMethod) {
            throw new IllegalStateException("Support vectors not available for linear kernel");
        }
        
        List<double[]> svList = new ArrayList<>();
        for (int i = 0; i < supportVectors.length; i++) {
            if (Math.abs(alphaPositive[i] - alphaNegative[i]) > 1e-8) {
                svList.add(Arrays.copyOf(supportVectors[i], supportVectors[i].length));
            }
        }
        
        return svList.toArray(new double[0][]);
    }
    
    /**
     * Returns the weight vector (linear kernel only).
     * 
     * @return Weight vector
     */
    public double[] getWeights() {
        if (useKernelMethod) {
            throw new IllegalStateException("Weights not available for non-linear kernels");
        }
        return Arrays.copyOf(weights, weights.length);
    }
    
    /**
     * Returns the bias term.
     * 
     * @return Bias term
     */
    public double getBias() {
        return useKernelMethod ? biasKernel : bias;
    }
    
    /**
     * Returns the kernel used by this SVR.
     * 
     * @return Kernel
     */
    public Kernel getKernel() {
        return kernel;
    }
    
    /**
     * Returns the epsilon value.
     * 
     * @return Epsilon
     */
    public double getEpsilon() {
        return epsilon;
    }
    
    /**
     * Returns the regularization parameter C.
     * 
     * @return C
     */
    public double getC() {
        return C;
    }
    
    /**
     * Returns whether the model has been fitted.
     * 
     * @return true if fitted
     */
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Returns whether the model uses kernel method.
     * 
     * @return true if using kernel method
     */
    public boolean usesKernelMethod() {
        return useKernelMethod;
    }
    
    /**
     * Builder class for SVR.
     */
    public static class Builder {
        private double C = 1.0;
        private double epsilon = 0.1;
        private int maxIter = 1000;
        private double tol = 1e-3;
        private double learningRate = 0.01;
        
        // Kernel parameters
        private Kernel.Type kernelType = Kernel.Type.LINEAR;
        private double gamma = 1.0;
        private double coef0 = 0.0;
        private int degree = 3;
        
        /**
         * Sets the regularization parameter C.
         * Larger C = less regularization, stricter fitting.
         * Smaller C = more regularization, more tolerance to errors.
         * 
         * @param C Regularization parameter (must be positive)
         * @return This builder
         */
        public Builder C(double C) {
            if (C <= 0.0) {
                throw new IllegalArgumentException("C must be positive");
            }
            this.C = C;
            return this;
        }
        
        /**
         * Sets the epsilon parameter.
         * Epsilon specifies the epsilon-tube within which no penalty
         * is associated in the training loss function.
         * 
         * @param epsilon Epsilon value (must be non-negative)
         * @return This builder
         */
        public Builder epsilon(double epsilon) {
            if (epsilon < 0.0) {
                throw new IllegalArgumentException("epsilon must be non-negative");
            }
            this.epsilon = epsilon;
            return this;
        }
        
        /**
         * Sets the maximum number of iterations.
         * 
         * @param maxIter Maximum iterations (must be positive)
         * @return This builder
         */
        public Builder maxIter(int maxIter) {
            if (maxIter <= 0) {
                throw new IllegalArgumentException("maxIter must be positive");
            }
            this.maxIter = maxIter;
            return this;
        }
        
        /**
         * Sets the tolerance for stopping criterion.
         * 
         * @param tol Tolerance (must be positive)
         * @return This builder
         */
        public Builder tol(double tol) {
            if (tol <= 0.0) {
                throw new IllegalArgumentException("tol must be positive");
            }
            this.tol = tol;
            return this;
        }
        
        /**
         * Sets the learning rate for gradient descent (linear kernel only).
         * 
         * @param learningRate Learning rate (must be positive)
         * @return This builder
         */
        public Builder learningRate(double learningRate) {
            if (learningRate <= 0.0) {
                throw new IllegalArgumentException("learningRate must be positive");
            }
            this.learningRate = learningRate;
            return this;
        }
        
        /**
         * Sets the kernel type.
         * 
         * @param kernelType The kernel type (LINEAR, RBF, POLYNOMIAL, SIGMOID)
         * @return This builder
         */
        public Builder kernel(Kernel.Type kernelType) {
            if (kernelType == null) {
                throw new IllegalArgumentException("kernelType cannot be null");
            }
            this.kernelType = kernelType;
            return this;
        }
        
        /**
         * Sets the kernel type using string.
         * 
         * @param kernelName One of: "linear", "rbf", "poly", "polynomial", "sigmoid"
         * @return This builder
         */
        public Builder kernel(String kernelName) {
            if (kernelName == null) {
                throw new IllegalArgumentException("kernelName cannot be null");
            }
            switch (kernelName.toLowerCase()) {
                case "linear":
                    this.kernelType = Kernel.Type.LINEAR;
                    break;
                case "rbf":
                case "gaussian":
                    this.kernelType = Kernel.Type.RBF;
                    break;
                case "poly":
                case "polynomial":
                    this.kernelType = Kernel.Type.POLYNOMIAL;
                    break;
                case "sigmoid":
                    this.kernelType = Kernel.Type.SIGMOID;
                    break;
                default:
                    throw new IllegalArgumentException(
                        "Unknown kernel: " + kernelName + 
                        ". Use: linear, rbf, poly, or sigmoid"
                    );
            }
            return this;
        }
        
        /**
         * Sets the gamma parameter for RBF, polynomial, and sigmoid kernels.
         * 
         * @param gamma The gamma value (must be positive)
         * @return This builder
         */
        public Builder gamma(double gamma) {
            if (gamma <= 0.0) {
                throw new IllegalArgumentException("gamma must be positive");
            }
            this.gamma = gamma;
            return this;
        }
        
        /**
         * Sets the coef0 parameter for polynomial and sigmoid kernels.
         * 
         * @param coef0 The independent term
         * @return This builder
         */
        public Builder coef0(double coef0) {
            this.coef0 = coef0;
            return this;
        }
        
        /**
         * Sets the degree for polynomial kernel.
         * 
         * @param degree The polynomial degree (must be at least 1)
         * @return This builder
         */
        public Builder degree(int degree) {
            if (degree < 1) {
                throw new IllegalArgumentException("degree must be at least 1");
            }
            this.degree = degree;
            return this;
        }
        
        /**
         * Builds the SVR instance.
         * 
         * @return A new SVR instance
         */
        public SVR build() {
            Kernel kernelObj = new Kernel(kernelType, gamma, coef0, degree);
            return new SVR(C, epsilon, maxIter, tol, learningRate, kernelObj);
        }
    }
}
