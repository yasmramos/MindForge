package io.github.yasmramos.mindforge.classification;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Logistic Regression classifier for binary and multiclass classification.
 * 
 * Supports multiple solvers:
 * - Gradient Descent (GD): Full batch gradient descent
 * - Stochastic Gradient Descent (SGD): Online learning with mini-batches
 * - Newton-CG: Newton-Conjugate Gradient for faster convergence
 * 
 * Regularization options:
 * - L1 (Lasso): Promotes sparsity in coefficients
 * - L2 (Ridge): Prevents overfitting with weight decay
 * - Elastic Net: Combination of L1 and L2
 * 
 * For multiclass classification, uses One-vs-Rest (OvR) strategy.
 * 
 * Example usage:
 * <pre>
 * LogisticRegression lr = new LogisticRegression.Builder()
 *     .penalty("l2")
 *     .C(1.0)
 *     .solver("gradient_descent")
 *     .maxIter(1000)
 *     .build();
 * 
 * lr.fit(X_train, y_train);
 * int[] predictions = lr.predict(X_test);
 * double[][] probabilities = lr.predictProba(X_test);
 * </pre>
 */
public class LogisticRegression {
    
    private String penalty;           // "l1", "l2", "elasticnet", "none"
    private double C;                 // Inverse of regularization strength (smaller = stronger)
    private double l1Ratio;           // Elastic Net mixing parameter (0=L2, 1=L1)
    private String solver;            // "gradient_descent", "sgd", "newton_cg"
    private int maxIter;              // Maximum iterations
    private double tol;               // Convergence tolerance
    private double learningRate;      // Learning rate for GD/SGD
    private int batchSize;            // Batch size for SGD
    private boolean fitIntercept;     // Whether to fit intercept
    private int randomState;          // Random seed for reproducibility
    private int verbose;              // Verbosity level
    
    // Model parameters
    private double[][] weights;       // Coefficients for each class
    private double[] intercepts;      // Intercepts for each class
    private int[] classes;            // Unique class labels
    private int nFeatures;            // Number of features
    private boolean isBinary;         // Binary vs multiclass
    
    // Training history
    private List<Double> lossHistory; // Loss values during training
    
    private LogisticRegression(Builder builder) {
        this.penalty = builder.penalty;
        this.C = builder.C;
        this.l1Ratio = builder.l1Ratio;
        this.solver = builder.solver;
        this.maxIter = builder.maxIter;
        this.tol = builder.tol;
        this.learningRate = builder.learningRate;
        this.batchSize = builder.batchSize;
        this.fitIntercept = builder.fitIntercept;
        this.randomState = builder.randomState;
        this.verbose = builder.verbose;
        this.lossHistory = new ArrayList<>();
    }
    
    /**
     * Fit the logistic regression model.
     * 
     * @param X Training features, shape (n_samples, n_features)
     * @param y Training labels
     */
    public void fit(double[][] X, int[] y) {
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same number of samples");
        }
        
        // Extract unique classes
        Set<Integer> uniqueClasses = new HashSet<>();
        for (int label : y) {
            uniqueClasses.add(label);
        }
        classes = uniqueClasses.stream().sorted().mapToInt(Integer::intValue).toArray();
        
        nFeatures = X[0].length;
        isBinary = classes.length == 2;
        
        // Standardize features if needed
        double[][] XStd = X;
        
        if (isBinary) {
            // Binary classification: single model
            weights = new double[1][nFeatures];
            intercepts = new double[1];
            
            // Convert to binary labels (0 and 1)
            int[] yBinary = new int[y.length];
            for (int i = 0; i < y.length; i++) {
                yBinary[i] = y[i] == classes[1] ? 1 : 0;
            }
            
            fitBinaryModel(XStd, yBinary, 0);
        } else {
            // Multiclass: One-vs-Rest
            weights = new double[classes.length][nFeatures];
            intercepts = new double[classes.length];
            
            for (int c = 0; c < classes.length; c++) {
                // Create binary labels for this class
                int[] yBinary = new int[y.length];
                for (int i = 0; i < y.length; i++) {
                    yBinary[i] = y[i] == classes[c] ? 1 : 0;
                }
                
                fitBinaryModel(XStd, yBinary, c);
            }
        }
    }
    
    /**
     * Fit a binary logistic regression model for one class.
     */
    private void fitBinaryModel(double[][] X, int[] y, int classIdx) {
        Random rand = new Random(randomState + classIdx);
        
        // Initialize weights
        for (int j = 0; j < nFeatures; j++) {
            weights[classIdx][j] = rand.nextGaussian() * 0.01;
        }
        intercepts[classIdx] = 0.0;
        
        switch (solver) {
            case "gradient_descent":
                fitGradientDescent(X, y, classIdx);
                break;
            case "sgd":
                fitSGD(X, y, classIdx, rand);
                break;
            case "newton_cg":
                fitNewtonCG(X, y, classIdx);
                break;
            default:
                throw new IllegalArgumentException("Unknown solver: " + solver);
        }
    }
    
    /**
     * Fit using full batch Gradient Descent.
     */
    private void fitGradientDescent(double[][] X, int[] y, int classIdx) {
        int nSamples = X.length;
        double prevLoss = Double.POSITIVE_INFINITY;
        
        for (int iter = 0; iter < maxIter; iter++) {
            // Compute predictions and gradients
            double[] predictions = new double[nSamples];
            for (int i = 0; i < nSamples; i++) {
                predictions[i] = sigmoid(dotProduct(X[i], weights[classIdx]) + intercepts[classIdx]);
            }
            
            // Compute gradients
            double[] gradW = new double[nFeatures];
            double gradB = 0.0;
            
            for (int i = 0; i < nSamples; i++) {
                double error = predictions[i] - y[i];
                for (int j = 0; j < nFeatures; j++) {
                    gradW[j] += error * X[i][j];
                }
                gradB += error;
            }
            
            // Average gradients
            for (int j = 0; j < nFeatures; j++) {
                gradW[j] /= nSamples;
            }
            gradB /= nSamples;
            
            // Add regularization gradient
            addRegularizationGradient(gradW, weights[classIdx]);
            
            // Update weights
            for (int j = 0; j < nFeatures; j++) {
                weights[classIdx][j] -= learningRate * gradW[j];
            }
            intercepts[classIdx] -= learningRate * gradB;
            
            // Compute loss
            double loss = computeLoss(X, y, classIdx);
            if (classIdx == 0) {
                lossHistory.add(loss);
            }
            
            // Check convergence
            if (Math.abs(prevLoss - loss) < tol) {
                if (verbose > 0 && classIdx == 0) {
                    System.out.println("Converged at iteration " + iter);
                }
                break;
            }
            prevLoss = loss;
            
            if (verbose > 1 && classIdx == 0 && iter % 100 == 0) {
                System.out.println("Iteration " + iter + ", Loss: " + loss);
            }
        }
    }
    
    /**
     * Fit using Stochastic Gradient Descent.
     */
    private void fitSGD(double[][] X, int[] y, int classIdx, Random rand) {
        int nSamples = X.length;
        double prevLoss = Double.POSITIVE_INFINITY;
        
        for (int epoch = 0; epoch < maxIter; epoch++) {
            // Shuffle data
            int[] indices = IntStream.range(0, nSamples).toArray();
            shuffleArray(indices, rand);
            
            // Mini-batch updates
            for (int start = 0; start < nSamples; start += batchSize) {
                int end = Math.min(start + batchSize, nSamples);
                int batchLen = end - start;
                
                // Compute gradients for batch
                double[] gradW = new double[nFeatures];
                double gradB = 0.0;
                
                for (int idx = start; idx < end; idx++) {
                    int i = indices[idx];
                    double pred = sigmoid(dotProduct(X[i], weights[classIdx]) + intercepts[classIdx]);
                    double error = pred - y[i];
                    
                    for (int j = 0; j < nFeatures; j++) {
                        gradW[j] += error * X[i][j];
                    }
                    gradB += error;
                }
                
                // Average gradients
                for (int j = 0; j < nFeatures; j++) {
                    gradW[j] /= batchLen;
                }
                gradB /= batchLen;
                
                // Add regularization
                addRegularizationGradient(gradW, weights[classIdx]);
                
                // Update weights
                for (int j = 0; j < nFeatures; j++) {
                    weights[classIdx][j] -= learningRate * gradW[j];
                }
                intercepts[classIdx] -= learningRate * gradB;
            }
            
            // Compute loss at end of epoch
            double loss = computeLoss(X, y, classIdx);
            if (classIdx == 0) {
                lossHistory.add(loss);
            }
            
            // Check convergence
            if (Math.abs(prevLoss - loss) < tol) {
                if (verbose > 0 && classIdx == 0) {
                    System.out.println("Converged at epoch " + epoch);
                }
                break;
            }
            prevLoss = loss;
            
            if (verbose > 1 && classIdx == 0 && epoch % 10 == 0) {
                System.out.println("Epoch " + epoch + ", Loss: " + loss);
            }
        }
    }
    
    /**
     * Fit using Newton-Conjugate Gradient method.
     * Simplified implementation using approximate Hessian.
     */
    private void fitNewtonCG(double[][] X, int[] y, int classIdx) {
        int nSamples = X.length;
        double prevLoss = Double.POSITIVE_INFINITY;
        
        for (int iter = 0; iter < maxIter; iter++) {
            // Compute predictions
            double[] predictions = new double[nSamples];
            for (int i = 0; i < nSamples; i++) {
                predictions[i] = sigmoid(dotProduct(X[i], weights[classIdx]) + intercepts[classIdx]);
            }
            
            // Compute gradient
            double[] gradW = new double[nFeatures];
            double gradB = 0.0;
            
            for (int i = 0; i < nSamples; i++) {
                double error = predictions[i] - y[i];
                for (int j = 0; j < nFeatures; j++) {
                    gradW[j] += error * X[i][j];
                }
                gradB += error;
            }
            
            // Average
            for (int j = 0; j < nFeatures; j++) {
                gradW[j] /= nSamples;
            }
            gradB /= nSamples;
            
            // Add regularization
            addRegularizationGradient(gradW, weights[classIdx]);
            
            // Approximate Hessian diagonal (Fisher information)
            double[] hessianDiag = new double[nFeatures];
            for (int j = 0; j < nFeatures; j++) {
                for (int i = 0; i < nSamples; i++) {
                    double p = predictions[i];
                    hessianDiag[j] += p * (1 - p) * X[i][j] * X[i][j];
                }
                hessianDiag[j] /= nSamples;
                
                // Add regularization to Hessian
                if (penalty.equals("l2") || penalty.equals("elasticnet")) {
                    hessianDiag[j] += (1.0 / C) * (1.0 - l1Ratio);
                }
                
                // Avoid division by zero
                if (hessianDiag[j] < 1e-8) {
                    hessianDiag[j] = 1e-8;
                }
            }
            
            // Newton update: w = w - H^-1 * grad
            for (int j = 0; j < nFeatures; j++) {
                weights[classIdx][j] -= gradW[j] / hessianDiag[j];
            }
            
            // Simple update for intercept
            intercepts[classIdx] -= learningRate * gradB;
            
            // Compute loss
            double loss = computeLoss(X, y, classIdx);
            if (classIdx == 0) {
                lossHistory.add(loss);
            }
            
            // Check convergence
            if (Math.abs(prevLoss - loss) < tol) {
                if (verbose > 0 && classIdx == 0) {
                    System.out.println("Converged at iteration " + iter);
                }
                break;
            }
            prevLoss = loss;
            
            if (verbose > 1 && classIdx == 0 && iter % 100 == 0) {
                System.out.println("Iteration " + iter + ", Loss: " + loss);
            }
        }
    }
    
    /**
     * Add regularization gradient to weight gradients.
     */
    private void addRegularizationGradient(double[] gradW, double[] w) {
        if (penalty.equals("l2")) {
            for (int j = 0; j < nFeatures; j++) {
                gradW[j] += (1.0 / C) * w[j];
            }
        } else if (penalty.equals("l1")) {
            for (int j = 0; j < nFeatures; j++) {
                gradW[j] += (1.0 / C) * Math.signum(w[j]);
            }
        } else if (penalty.equals("elasticnet")) {
            for (int j = 0; j < nFeatures; j++) {
                gradW[j] += (1.0 / C) * (l1Ratio * Math.signum(w[j]) + (1 - l1Ratio) * w[j]);
            }
        }
    }
    
    /**
     * Compute logistic loss with regularization.
     */
    private double computeLoss(double[][] X, int[] y, int classIdx) {
        int nSamples = X.length;
        double loss = 0.0;
        
        // Log loss
        for (int i = 0; i < nSamples; i++) {
            double z = dotProduct(X[i], weights[classIdx]) + intercepts[classIdx];
            double p = sigmoid(z);
            
            // Clip to avoid log(0)
            p = Math.max(1e-15, Math.min(1 - 1e-15, p));
            
            if (y[i] == 1) {
                loss -= Math.log(p);
            } else {
                loss -= Math.log(1 - p);
            }
        }
        loss /= nSamples;
        
        // Add regularization
        if (penalty.equals("l2")) {
            double l2Penalty = 0.0;
            for (int j = 0; j < nFeatures; j++) {
                l2Penalty += weights[classIdx][j] * weights[classIdx][j];
            }
            loss += (0.5 / C) * l2Penalty;
        } else if (penalty.equals("l1")) {
            double l1Penalty = 0.0;
            for (int j = 0; j < nFeatures; j++) {
                l1Penalty += Math.abs(weights[classIdx][j]);
            }
            loss += (1.0 / C) * l1Penalty;
        } else if (penalty.equals("elasticnet")) {
            double l1Penalty = 0.0;
            double l2Penalty = 0.0;
            for (int j = 0; j < nFeatures; j++) {
                l1Penalty += Math.abs(weights[classIdx][j]);
                l2Penalty += weights[classIdx][j] * weights[classIdx][j];
            }
            loss += (1.0 / C) * (l1Ratio * l1Penalty + 0.5 * (1 - l1Ratio) * l2Penalty);
        }
        
        return loss;
    }
    
    /**
     * Predict class labels for samples.
     * 
     * @param X Test samples, shape (n_samples, n_features)
     * @return Predicted class labels
     */
    public int[] predict(double[][] X) {
        double[][] proba = predictProba(X);
        int[] predictions = new int[X.length];
        
        for (int i = 0; i < X.length; i++) {
            int maxIdx = 0;
            for (int j = 1; j < proba[i].length; j++) {
                if (proba[i][j] > proba[i][maxIdx]) {
                    maxIdx = j;
                }
            }
            predictions[i] = classes[maxIdx];
        }
        
        return predictions;
    }
    
    /**
     * Predict class probabilities for samples.
     * 
     * @param X Test samples, shape (n_samples, n_features)
     * @return Probability estimates, shape (n_samples, n_classes)
     */
    public double[][] predictProba(double[][] X) {
        if (weights == null) {
            throw new IllegalStateException("Model not fitted yet");
        }
        
        double[][] proba = new double[X.length][classes.length];
        
        if (isBinary) {
            // Binary classification
            for (int i = 0; i < X.length; i++) {
                double z = dotProduct(X[i], weights[0]) + intercepts[0];
                double p1 = sigmoid(z);
                proba[i][0] = 1 - p1;
                proba[i][1] = p1;
            }
        } else {
            // Multiclass: One-vs-Rest with normalization
            for (int i = 0; i < X.length; i++) {
                double sum = 0.0;
                for (int c = 0; c < classes.length; c++) {
                    double z = dotProduct(X[i], weights[c]) + intercepts[c];
                    proba[i][c] = sigmoid(z);
                    sum += proba[i][c];
                }
                
                // Normalize to sum to 1
                for (int c = 0; c < classes.length; c++) {
                    proba[i][c] /= sum;
                }
            }
        }
        
        return proba;
    }
    
    /**
     * Get the coefficients (weights) of the model.
     * 
     * @return Coefficients, shape (n_classes, n_features) or (1, n_features) for binary
     */
    public double[][] getCoefficients() {
        return weights;
    }
    
    /**
     * Get the intercepts of the model.
     * 
     * @return Intercepts, length n_classes or 1 for binary
     */
    public double[] getIntercepts() {
        return intercepts;
    }
    
    /**
     * Get the unique class labels.
     * 
     * @return Class labels
     */
    public int[] getClasses() {
        return classes;
    }
    
    /**
     * Get the loss history during training.
     * 
     * @return List of loss values
     */
    public List<Double> getLossHistory() {
        return new ArrayList<>(lossHistory);
    }
    
    /**
     * Sigmoid activation function.
     */
    private double sigmoid(double z) {
        if (z > 20) return 1.0;
        if (z < -20) return 0.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }
    
    /**
     * Compute dot product of two vectors.
     */
    private double dotProduct(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    /**
     * Shuffle an array in place.
     */
    private void shuffleArray(int[] array, Random rand) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    
    /**
     * Builder class for LogisticRegression.
     */
    public static class Builder {
        private String penalty = "l2";
        private double C = 1.0;
        private double l1Ratio = 0.5;
        private String solver = "gradient_descent";
        private int maxIter = 1000;
        private double tol = 1e-4;
        private double learningRate = 0.01;
        private int batchSize = 32;
        private boolean fitIntercept = true;
        private int randomState = 42;
        private int verbose = 0;
        
        /**
         * Set the penalty (regularization type).
         * 
         * @param penalty "l1", "l2", "elasticnet", or "none"
         * @return Builder instance
         */
        public Builder penalty(String penalty) {
            this.penalty = penalty;
            return this;
        }
        
        /**
         * Set the inverse of regularization strength.
         * Smaller values specify stronger regularization.
         * 
         * @param C Regularization parameter (must be positive)
         * @return Builder instance
         */
        public Builder C(double C) {
            if (C <= 0) {
                throw new IllegalArgumentException("C must be positive");
            }
            this.C = C;
            return this;
        }
        
        /**
         * Set the Elastic Net mixing parameter.
         * Only used if penalty="elasticnet".
         * 
         * @param l1Ratio Mixing parameter (0 = L2, 1 = L1)
         * @return Builder instance
         */
        public Builder l1Ratio(double l1Ratio) {
            if (l1Ratio < 0 || l1Ratio > 1) {
                throw new IllegalArgumentException("l1Ratio must be in [0, 1]");
            }
            this.l1Ratio = l1Ratio;
            return this;
        }
        
        /**
         * Set the optimization solver.
         * 
         * @param solver "gradient_descent", "sgd", or "newton_cg"
         * @return Builder instance
         */
        public Builder solver(String solver) {
            this.solver = solver;
            return this;
        }
        
        /**
         * Set the maximum number of iterations.
         * 
         * @param maxIter Maximum iterations
         * @return Builder instance
         */
        public Builder maxIter(int maxIter) {
            this.maxIter = maxIter;
            return this;
        }
        
        /**
         * Set the convergence tolerance.
         * 
         * @param tol Tolerance for stopping criterion
         * @return Builder instance
         */
        public Builder tol(double tol) {
            this.tol = tol;
            return this;
        }
        
        /**
         * Set the learning rate for gradient-based solvers.
         * 
         * @param learningRate Learning rate
         * @return Builder instance
         */
        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }
        
        /**
         * Set the batch size for SGD solver.
         * 
         * @param batchSize Batch size
         * @return Builder instance
         */
        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }
        
        /**
         * Set whether to fit an intercept term.
         * 
         * @param fitIntercept Whether to fit intercept
         * @return Builder instance
         */
        public Builder fitIntercept(boolean fitIntercept) {
            this.fitIntercept = fitIntercept;
            return this;
        }
        
        /**
         * Set the random state for reproducibility.
         * 
         * @param randomState Random seed
         * @return Builder instance
         */
        public Builder randomState(int randomState) {
            this.randomState = randomState;
            return this;
        }
        
        /**
         * Set the verbosity level.
         * 
         * @param verbose Verbosity (0 = silent, 1 = convergence, 2 = detailed)
         * @return Builder instance
         */
        public Builder verbose(int verbose) {
            this.verbose = verbose;
            return this;
        }
        
        /**
         * Build the LogisticRegression instance.
         * 
         * @return Configured LogisticRegression
         */
        public LogisticRegression build() {
            return new LogisticRegression(this);
        }
    }
}
