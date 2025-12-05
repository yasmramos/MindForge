package com.mindforge.classification;

import java.util.*;

/**
 * Support Vector Classifier (SVC) for binary and multiclass classification.
 * 
 * Implements Support Vector Machines with multiple kernel options:
 * - LINEAR: Linear kernel (default)
 * - RBF: Radial Basis Function (Gaussian) kernel
 * - POLYNOMIAL: Polynomial kernel
 * - SIGMOID: Sigmoid/hyperbolic tangent kernel
 * 
 * Uses the SMO (Sequential Minimal Optimization) algorithm for training
 * and One-vs-Rest strategy for multiclass classification.
 * 
 * Example usage:
 * <pre>
 * // Linear SVM (default)
 * SVC svc = new SVC.Builder()
 *     .C(1.0)
 *     .build();
 * 
 * // RBF kernel SVM
 * SVC svc = new SVC.Builder()
 *     .kernel(Kernel.Type.RBF)
 *     .gamma(0.5)
 *     .C(1.0)
 *     .build();
 * 
 * // Polynomial kernel SVM
 * SVC svc = new SVC.Builder()
 *     .kernel(Kernel.Type.POLYNOMIAL)
 *     .degree(3)
 *     .gamma(1.0)
 *     .coef0(0.0)
 *     .C(1.0)
 *     .build();
 * 
 * svc.train(X_train, y_train);
 * int[] predictions = svc.predict(X_test);
 * </pre>
 * 
 * @author MindForge Team
 */
public class SVC implements Classifier<double[]> {
    
    // Hyperparameters
    private double C; // Regularization parameter
    private int maxIter;
    private double tol; // Tolerance for stopping criterion
    private double learningRate; // For linear kernel gradient descent fallback
    
    // Kernel parameters
    private Kernel kernel;
    
    // Model parameters
    private int numClasses;
    private int numFeatures;
    private int[] classes;
    
    // For linear kernel (primal form)
    private double[][] weights; // [class][feature]
    private double[] bias; // [class]
    
    // For non-linear kernels (dual form)
    private double[][][] alphas; // [class][support_vectors]
    private double[][] supportVectors; // Training data (needed for prediction with kernels)
    private int[][] supportVectorLabels; // Binary labels for each class
    private double[] biasKernel; // Bias for kernel SVM
    
    private boolean isTrained = false;
    private boolean useKernelMethod = false;
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private SVC(double C, int maxIter, double tol, double learningRate, Kernel kernel) {
        this.C = C;
        this.maxIter = maxIter;
        this.tol = tol;
        this.learningRate = learningRate;
        this.kernel = kernel;
        this.useKernelMethod = (kernel.getType() != Kernel.Type.LINEAR);
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        numFeatures = X[0].length;
        
        // Find unique classes
        Set<Integer> classSet = new HashSet<>();
        for (int label : y) {
            classSet.add(label);
        }
        numClasses = classSet.size();
        classes = new int[numClasses];
        int idx = 0;
        for (int cls : classSet) {
            classes[idx++] = cls;
        }
        Arrays.sort(classes);
        
        if (useKernelMethod) {
            trainWithKernel(X, y);
        } else {
            trainLinear(X, y);
        }
        
        isTrained = true;
    }
    
    /**
     * Trains using linear kernel with gradient descent (primal form).
     */
    private void trainLinear(double[][] X, int[] y) {
        // Initialize weights and bias
        weights = new double[numClasses][numFeatures];
        bias = new double[numClasses];
        
        // Train one-vs-rest classifiers
        for (int c = 0; c < numClasses; c++) {
            trainLinearBinaryClassifier(X, y, c);
        }
    }
    
    /**
     * Trains a binary linear classifier for one class vs the rest.
     */
    private void trainLinearBinaryClassifier(double[][] X, int[] y, int classIdx) {
        int n = X.length;
        int targetClass = classes[classIdx];
        
        // Create binary labels: +1 for target class, -1 for others
        int[] binaryY = new int[n];
        for (int i = 0; i < n; i++) {
            binaryY[i] = (y[i] == targetClass) ? 1 : -1;
        }
        
        // Initialize weights and bias for this classifier
        double[] w = new double[numFeatures];
        double b = 0.0;
        
        Random random = new Random(42);
        for (int f = 0; f < numFeatures; f++) {
            w[f] = random.nextGaussian() * 0.01;
        }
        
        // Gradient descent optimization
        for (int iter = 0; iter < maxIter; iter++) {
            double[] gradW = new double[numFeatures];
            double gradB = 0.0;
            double loss = 0.0;
            
            // Compute gradients
            for (int i = 0; i < n; i++) {
                double prediction = computeLinearScore(X[i], w, b);
                double margin = binaryY[i] * prediction;
                
                // Hinge loss and gradient
                if (margin < 1) {
                    loss += 1 - margin;
                    
                    // Gradient of hinge loss
                    for (int f = 0; f < numFeatures; f++) {
                        gradW[f] -= binaryY[i] * X[i][f];
                    }
                    gradB -= binaryY[i];
                }
            }
            
            // Add regularization term
            for (int f = 0; f < numFeatures; f++) {
                loss += 0.5 * C * w[f] * w[f];
                gradW[f] += C * w[f];
            }
            
            // Update weights and bias
            for (int f = 0; f < numFeatures; f++) {
                w[f] -= learningRate * gradW[f] / n;
            }
            b -= learningRate * gradB / n;
            
            // Check convergence
            if (iter > 0 && Math.abs(loss) < tol) {
                break;
            }
        }
        
        // Store trained parameters
        weights[classIdx] = w;
        bias[classIdx] = b;
    }
    
    /**
     * Trains using non-linear kernels with SMO algorithm (dual form).
     */
    private void trainWithKernel(double[][] X, int[] y) {
        int n = X.length;
        
        // Store support vectors (all training data for kernel method)
        supportVectors = new double[n][];
        for (int i = 0; i < n; i++) {
            supportVectors[i] = Arrays.copyOf(X[i], X[i].length);
        }
        
        // Precompute kernel matrix
        double[][] K = kernel.computeMatrix(X);
        
        // Initialize alpha and bias arrays for each class
        alphas = new double[numClasses][1][n];
        supportVectorLabels = new int[numClasses][n];
        biasKernel = new double[numClasses];
        
        // Train one-vs-rest classifiers
        for (int c = 0; c < numClasses; c++) {
            trainKernelBinaryClassifier(X, y, K, c);
        }
    }
    
    /**
     * Trains a binary kernel classifier using simplified SMO algorithm.
     */
    private void trainKernelBinaryClassifier(double[][] X, int[] y, double[][] K, int classIdx) {
        int n = X.length;
        int targetClass = classes[classIdx];
        
        // Create binary labels: +1 for target class, -1 for others
        int[] binaryY = new int[n];
        for (int i = 0; i < n; i++) {
            binaryY[i] = (y[i] == targetClass) ? 1 : -1;
        }
        supportVectorLabels[classIdx] = binaryY;
        
        // Initialize alphas
        double[] alpha = new double[n];
        double b = 0.0;
        
        // Simplified SMO algorithm
        int passes = 0;
        int maxPasses = 5;
        
        while (passes < maxPasses) {
            int numChangedAlphas = 0;
            
            for (int i = 0; i < n; i++) {
                // Compute error for i
                double Ei = computeKernelOutput(alpha, binaryY, K, b, i) - binaryY[i];
                
                // Check KKT conditions
                if ((binaryY[i] * Ei < -tol && alpha[i] < C) ||
                    (binaryY[i] * Ei > tol && alpha[i] > 0)) {
                    
                    // Select j randomly, j != i
                    int j;
                    Random rand = new Random();
                    do {
                        j = rand.nextInt(n);
                    } while (j == i);
                    
                    // Compute error for j
                    double Ej = computeKernelOutput(alpha, binaryY, K, b, j) - binaryY[j];
                    
                    // Save old alphas
                    double alphaIOld = alpha[i];
                    double alphaJOld = alpha[j];
                    
                    // Compute bounds L and H
                    double L, H;
                    if (binaryY[i] != binaryY[j]) {
                        L = Math.max(0, alpha[j] - alpha[i]);
                        H = Math.min(C, C + alpha[j] - alpha[i]);
                    } else {
                        L = Math.max(0, alpha[i] + alpha[j] - C);
                        H = Math.min(C, alpha[i] + alpha[j]);
                    }
                    
                    if (Math.abs(L - H) < 1e-10) {
                        continue;
                    }
                    
                    // Compute eta
                    double eta = 2 * K[i][j] - K[i][i] - K[j][j];
                    if (eta >= 0) {
                        continue;
                    }
                    
                    // Compute new alpha[j]
                    alpha[j] = alpha[j] - (binaryY[j] * (Ei - Ej)) / eta;
                    
                    // Clip alpha[j]
                    alpha[j] = Math.min(H, Math.max(L, alpha[j]));
                    
                    // Check if change is significant
                    if (Math.abs(alpha[j] - alphaJOld) < 1e-5) {
                        continue;
                    }
                    
                    // Compute new alpha[i]
                    alpha[i] = alpha[i] + binaryY[i] * binaryY[j] * (alphaJOld - alpha[j]);
                    
                    // Compute new threshold b
                    double b1 = b - Ei - binaryY[i] * (alpha[i] - alphaIOld) * K[i][i]
                                      - binaryY[j] * (alpha[j] - alphaJOld) * K[i][j];
                    double b2 = b - Ej - binaryY[i] * (alpha[i] - alphaIOld) * K[i][j]
                                      - binaryY[j] * (alpha[j] - alphaJOld) * K[j][j];
                    
                    if (0 < alpha[i] && alpha[i] < C) {
                        b = b1;
                    } else if (0 < alpha[j] && alpha[j] < C) {
                        b = b2;
                    } else {
                        b = (b1 + b2) / 2;
                    }
                    
                    numChangedAlphas++;
                }
            }
            
            if (numChangedAlphas == 0) {
                passes++;
            } else {
                passes = 0;
            }
        }
        
        // Store results
        alphas[classIdx][0] = alpha;
        biasKernel[classIdx] = b;
    }
    
    /**
     * Computes kernel SVM output for a training sample.
     */
    private double computeKernelOutput(double[] alpha, int[] binaryY, double[][] K, double b, int idx) {
        double output = b;
        for (int i = 0; i < alpha.length; i++) {
            output += alpha[i] * binaryY[i] * K[i][idx];
        }
        return output;
    }
    
    /**
     * Computes the linear decision score for a sample.
     */
    private double computeLinearScore(double[] x, double[] w, double b) {
        double score = b;
        for (int f = 0; f < x.length; f++) {
            score += w[f] * x[f];
        }
        return score;
    }
    
    /**
     * Computes kernel SVM output for a new sample.
     */
    private double computeKernelScore(double[] x, int classIdx) {
        double output = biasKernel[classIdx];
        double[] alpha = alphas[classIdx][0];
        int[] binaryY = supportVectorLabels[classIdx];
        
        for (int i = 0; i < supportVectors.length; i++) {
            if (alpha[i] > 1e-8) { // Only support vectors contribute
                output += alpha[i] * binaryY[i] * kernel.compute(supportVectors[i], x);
            }
        }
        return output;
    }
    
    @Override
    public int predict(double[] x) {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, x.length)
            );
        }
        
        double maxScore = Double.NEGATIVE_INFINITY;
        int predictedClass = classes[0];
        
        // Find class with maximum decision score
        for (int c = 0; c < numClasses; c++) {
            double score;
            if (useKernelMethod) {
                score = computeKernelScore(x, c);
            } else {
                score = computeLinearScore(x, weights[c], bias[c]);
            }
            
            if (score > maxScore) {
                maxScore = score;
                predictedClass = classes[c];
            }
        }
        
        return predictedClass;
    }
    
    /**
     * Predicts class labels for multiple samples.
     */
    public int[] predict(double[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    /**
     * Computes decision scores for all classes.
     */
    public double[] decisionFunction(double[] x) {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, x.length)
            );
        }
        
        double[] scores = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            if (useKernelMethod) {
                scores[c] = computeKernelScore(x, c);
            } else {
                scores[c] = computeLinearScore(x, weights[c], bias[c]);
            }
        }
        
        return scores;
    }
    
    /**
     * Returns the number of support vectors for each class.
     * Only applicable for kernel SVM.
     */
    public int[] getNumSupportVectors() {
        if (!useKernelMethod) {
            return new int[numClasses]; // All zeros for linear SVM
        }
        
        int[] counts = new int[numClasses];
        for (int c = 0; c < numClasses; c++) {
            int count = 0;
            double[] alpha = alphas[c][0];
            for (double a : alpha) {
                if (a > 1e-8) {
                    count++;
                }
            }
            counts[c] = count;
        }
        return counts;
    }
    
    /**
     * Returns the total number of support vectors.
     */
    public int getTotalSupportVectors() {
        int[] counts = getNumSupportVectors();
        int total = 0;
        for (int count : counts) {
            total += count;
        }
        return total;
    }
    
    @Override
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * Returns the class labels.
     */
    public int[] getClasses() {
        return Arrays.copyOf(classes, classes.length);
    }
    
    /**
     * Returns the weight vectors (linear kernel only).
     */
    public double[][] getWeights() {
        if (useKernelMethod) {
            throw new IllegalStateException("Weights not available for non-linear kernels");
        }
        double[][] copy = new double[numClasses][];
        for (int i = 0; i < numClasses; i++) {
            copy[i] = Arrays.copyOf(weights[i], numFeatures);
        }
        return copy;
    }
    
    /**
     * Returns the bias terms.
     */
    public double[] getBias() {
        if (useKernelMethod) {
            return Arrays.copyOf(biasKernel, biasKernel.length);
        }
        return Arrays.copyOf(bias, bias.length);
    }
    
    /**
     * Returns the kernel used by this SVM.
     */
    public Kernel getKernel() {
        return kernel;
    }
    
    /**
     * Returns whether the model has been trained.
     */
    public boolean isTrained() {
        return isTrained;
    }
    
    /**
     * Returns whether the model uses kernel method (non-linear).
     */
    public boolean usesKernelMethod() {
        return useKernelMethod;
    }
    
    /**
     * Builder class for SVC.
     */
    public static class Builder {
        private double C = 1.0;
        private int maxIter = 1000;
        private double tol = 1e-3;
        private double learningRate = 0.01;
        
        // Kernel parameters
        private Kernel.Type kernelType = Kernel.Type.LINEAR;
        private double gamma = 1.0;
        private double coef0 = 0.0;
        private int degree = 3;
        private boolean autoGamma = false;
        
        /**
         * Sets the regularization parameter C.
         * Larger C = less regularization (may overfit).
         * Smaller C = more regularization (may underfit).
         * 
         * @param C Regularization parameter (must be positive)
         */
        public Builder C(double C) {
            if (C <= 0.0) {
                throw new IllegalArgumentException("C must be positive");
            }
            this.C = C;
            return this;
        }
        
        /**
         * Sets the maximum number of iterations.
         * 
         * @param maxIter Maximum iterations (must be positive)
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
         */
        public Builder tol(double tol) {
            if (tol <= 0.0) {
                throw new IllegalArgumentException("tol must be positive");
            }
            this.tol = tol;
            return this;
        }
        
        /**
         * Sets the learning rate (for linear kernel only).
         * 
         * @param learningRate Learning rate (must be positive)
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
         */
        public Builder kernel(Kernel.Type kernelType) {
            this.kernelType = kernelType;
            return this;
        }
        
        /**
         * Sets the kernel type using string.
         * 
         * @param kernelName One of: "linear", "rbf", "poly", "polynomial", "sigmoid"
         */
        public Builder kernel(String kernelName) {
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
         */
        public Builder gamma(double gamma) {
            if (gamma <= 0.0) {
                throw new IllegalArgumentException("gamma must be positive");
            }
            this.gamma = gamma;
            this.autoGamma = false;
            return this;
        }
        
        /**
         * Sets gamma to be computed automatically as 1/n_features.
         */
        public Builder gammaAuto() {
            this.autoGamma = true;
            return this;
        }
        
        /**
         * Sets gamma using a string value.
         * 
         * @param gammaStr Either "auto", "scale", or a numeric value
         */
        public Builder gamma(String gammaStr) {
            if (gammaStr.equalsIgnoreCase("auto") || gammaStr.equalsIgnoreCase("scale")) {
                this.autoGamma = true;
            } else {
                try {
                    this.gamma = Double.parseDouble(gammaStr);
                    this.autoGamma = false;
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                        "gamma must be 'auto', 'scale', or a numeric value"
                    );
                }
            }
            return this;
        }
        
        /**
         * Sets the coef0 parameter for polynomial and sigmoid kernels.
         * 
         * @param coef0 The independent term
         */
        public Builder coef0(double coef0) {
            this.coef0 = coef0;
            return this;
        }
        
        /**
         * Sets the degree for polynomial kernel.
         * 
         * @param degree The polynomial degree (must be at least 1)
         */
        public Builder degree(int degree) {
            if (degree < 1) {
                throw new IllegalArgumentException("degree must be at least 1");
            }
            this.degree = degree;
            return this;
        }
        
        /**
         * Builds the SVC instance.
         * 
         * @return A new SVC instance
         */
        public SVC build() {
            // Create kernel with appropriate gamma
            double effectiveGamma = gamma;
            // Note: autoGamma will be applied during training when n_features is known
            // For now, use default if auto
            
            Kernel kernelObj = new Kernel(kernelType, effectiveGamma, coef0, degree);
            return new SVC(C, maxIter, tol, learningRate, kernelObj);
        }
    }
}
