package com.mindforge.classification;

import java.util.Arrays;
import java.util.Random;

/**
 * Logistic Regression Classifier for binary and multiclass classification.
 * 
 * <p>Logistic Regression is a linear model for classification that models the
 * probability of class membership using the logistic (sigmoid) function for 
 * binary classification and softmax for multiclass classification.</p>
 * 
 * <p>Features:</p>
 * <ul>
 *   <li>Binary and multiclass classification (One-vs-Rest or Multinomial)</li>
 *   <li>L1 (Lasso), L2 (Ridge), and ElasticNet regularization</li>
 *   <li>Configurable optimization parameters</li>
 *   <li>Probability predictions</li>
 *   <li>Class weights for imbalanced datasets</li>
 * </ul>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}};
 * int[] y = {0, 0, 1, 1};
 * 
 * LogisticRegression lr = new LogisticRegression.Builder()
 *     .regularization(Regularization.L2)
 *     .C(1.0)
 *     .maxIterations(1000)
 *     .build();
 * 
 * lr.fit(X, y);
 * int prediction = lr.predict(new double[]{2.5, 3.5});
 * double[] probabilities = lr.predictProba(new double[]{2.5, 3.5});
 * }</pre>
 * 
 * @author MindForge Team
 * @version 1.2.0-alpha
 */
public class LogisticRegression implements Classifier<double[]>, ProbabilisticClassifier {
    
    /**
     * Regularization type for the logistic regression.
     */
    public enum Regularization {
        /** No regularization */
        NONE,
        /** L1 regularization (Lasso) - promotes sparsity */
        L1,
        /** L2 regularization (Ridge) - prevents large weights */
        L2,
        /** ElasticNet - combination of L1 and L2 */
        ELASTICNET
    }
    
    /**
     * Solver algorithm for optimization.
     */
    public enum Solver {
        /** Gradient descent with momentum */
        SGD,
        /** Limited-memory BFGS (quasi-Newton method) */
        LBFGS
    }
    
    /**
     * Multiclass strategy.
     */
    public enum MultiClass {
        /** One-vs-Rest: train binary classifiers for each class */
        OVR,
        /** Multinomial: use softmax for multiclass */
        MULTINOMIAL
    }
    
    // Hyperparameters
    private final Regularization regularization;
    private final double C;  // Inverse of regularization strength
    private final double l1Ratio;  // For ElasticNet: l1_ratio * L1 + (1 - l1_ratio) * L2
    private final int maxIterations;
    private final double tolerance;
    private final double learningRate;
    private final Solver solver;
    private final MultiClass multiClass;
    private final boolean fitIntercept;
    private final int randomState;
    private final double[] classWeights;
    
    // Model parameters
    private double[][] weights;  // Shape: (numClasses, numFeatures) or (1, numFeatures) for binary
    private double[] intercepts; // Shape: (numClasses,) or (1,) for binary
    private int numClasses;
    private int numFeatures;
    private int[] classes;
    private boolean fitted;
    private Random random;
    
    // Training history
    private double[] lossHistory;
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private LogisticRegression(Builder builder) {
        this.regularization = builder.regularization;
        this.C = builder.C;
        this.l1Ratio = builder.l1Ratio;
        this.maxIterations = builder.maxIterations;
        this.tolerance = builder.tolerance;
        this.learningRate = builder.learningRate;
        this.solver = builder.solver;
        this.multiClass = builder.multiClass;
        this.fitIntercept = builder.fitIntercept;
        this.randomState = builder.randomState;
        this.classWeights = builder.classWeights;
        this.fitted = false;
        this.random = new Random(randomState);
    }
    
    /**
     * Builder pattern for creating LogisticRegression instances.
     */
    public static class Builder {
        private Regularization regularization = Regularization.L2;
        private double C = 1.0;
        private double l1Ratio = 0.5;
        private int maxIterations = 1000;
        private double tolerance = 1e-4;
        private double learningRate = 0.1;
        private Solver solver = Solver.LBFGS;
        private MultiClass multiClass = MultiClass.MULTINOMIAL;
        private boolean fitIntercept = true;
        private int randomState = 42;
        private double[] classWeights = null;
        
        public Builder regularization(Regularization regularization) {
            this.regularization = regularization;
            return this;
        }
        
        public Builder C(double C) {
            if (C <= 0) {
                throw new IllegalArgumentException("C must be positive");
            }
            this.C = C;
            return this;
        }
        
        public Builder l1Ratio(double l1Ratio) {
            if (l1Ratio < 0 || l1Ratio > 1) {
                throw new IllegalArgumentException("l1Ratio must be in [0, 1]");
            }
            this.l1Ratio = l1Ratio;
            return this;
        }
        
        public Builder maxIterations(int maxIterations) {
            if (maxIterations < 1) {
                throw new IllegalArgumentException("maxIterations must be at least 1");
            }
            this.maxIterations = maxIterations;
            return this;
        }
        
        public Builder tolerance(double tolerance) {
            if (tolerance <= 0) {
                throw new IllegalArgumentException("tolerance must be positive");
            }
            this.tolerance = tolerance;
            return this;
        }
        
        public Builder learningRate(double learningRate) {
            if (learningRate <= 0) {
                throw new IllegalArgumentException("learningRate must be positive");
            }
            this.learningRate = learningRate;
            return this;
        }
        
        public Builder solver(Solver solver) {
            this.solver = solver;
            return this;
        }
        
        public Builder multiClass(MultiClass multiClass) {
            this.multiClass = multiClass;
            return this;
        }
        
        public Builder fitIntercept(boolean fitIntercept) {
            this.fitIntercept = fitIntercept;
            return this;
        }
        
        public Builder randomState(int randomState) {
            this.randomState = randomState;
            return this;
        }
        
        public Builder classWeights(double[] classWeights) {
            this.classWeights = classWeights;
            return this;
        }
        
        public LogisticRegression build() {
            return new LogisticRegression(this);
        }
    }
    
    /**
     * Default constructor with default hyperparameters.
     */
    public LogisticRegression() {
        this(new Builder());
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        fit(X, y);
    }
    
    /**
     * Fit the logistic regression classifier.
     * 
     * @param X Training feature matrix (n_samples x n_features)
     * @param y Training labels (n_samples)
     */
    public void fit(double[][] X, int[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Training data cannot be null");
        }
        if (X.length == 0 || X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same length and be non-empty");
        }
        
        int n = X.length;
        this.numFeatures = X[0].length;
        
        // Find unique classes
        this.classes = Arrays.stream(y).distinct().sorted().toArray();
        this.numClasses = classes.length;
        
        if (numClasses < 2) {
            throw new IllegalArgumentException("Need at least 2 classes for classification");
        }
        
        // Initialize weights
        if (numClasses == 2) {
            // Binary classification: single weight vector
            weights = new double[1][numFeatures];
            intercepts = new double[1];
        } else {
            // Multiclass: one weight vector per class
            weights = new double[numClasses][numFeatures];
            intercepts = new double[numClasses];
        }
        
        // Initialize with small random values
        for (int c = 0; c < weights.length; c++) {
            for (int f = 0; f < numFeatures; f++) {
                weights[c][f] = random.nextGaussian() * 0.01;
            }
            intercepts[c] = 0.0;
        }
        
        // Convert labels to class indices
        int[] yIndices = new int[n];
        for (int i = 0; i < n; i++) {
            yIndices[i] = Arrays.binarySearch(classes, y[i]);
        }
        
        // Compute sample weights
        double[] sampleWeights = new double[n];
        Arrays.fill(sampleWeights, 1.0);
        if (classWeights != null) {
            for (int i = 0; i < n; i++) {
                int classIdx = yIndices[i];
                if (classIdx < classWeights.length) {
                    sampleWeights[i] = classWeights[classIdx];
                }
            }
        }
        
        // Train using selected solver
        if (solver == Solver.SGD) {
            trainSGD(X, yIndices, sampleWeights);
        } else {
            trainLBFGS(X, yIndices, sampleWeights);
        }
        
        this.fitted = true;
    }
    
    /**
     * Train using Stochastic Gradient Descent.
     */
    private void trainSGD(double[][] X, int[] y, double[] sampleWeights) {
        int n = X.length;
        lossHistory = new double[maxIterations];
        
        double lr = learningRate;
        double momentum = 0.9;
        double[][] velocityW = new double[weights.length][numFeatures];
        double[] velocityB = new double[intercepts.length];
        
        for (int iter = 0; iter < maxIterations; iter++) {
            // Compute predictions
            double[][] probs = computeProbabilities(X);
            
            // Compute gradients
            double[][] gradW = new double[weights.length][numFeatures];
            double[] gradB = new double[intercepts.length];
            
            double loss = 0.0;
            
            if (numClasses == 2) {
                // Binary classification
                for (int i = 0; i < n; i++) {
                    double prob = probs[i][0];
                    double target = y[i];
                    double error = (prob - target) * sampleWeights[i];
                    
                    for (int f = 0; f < numFeatures; f++) {
                        gradW[0][f] += error * X[i][f];
                    }
                    if (fitIntercept) {
                        gradB[0] += error;
                    }
                    
                    // Log loss
                    prob = Math.max(1e-15, Math.min(1 - 1e-15, prob));
                    loss -= sampleWeights[i] * (target * Math.log(prob) + (1 - target) * Math.log(1 - prob));
                }
            } else {
                // Multiclass classification
                for (int i = 0; i < n; i++) {
                    int targetClass = y[i];
                    for (int c = 0; c < numClasses; c++) {
                        double prob = probs[i][c];
                        double target = (c == targetClass) ? 1.0 : 0.0;
                        double error = (prob - target) * sampleWeights[i];
                        
                        for (int f = 0; f < numFeatures; f++) {
                            gradW[c][f] += error * X[i][f];
                        }
                        if (fitIntercept) {
                            gradB[c] += error;
                        }
                    }
                    
                    // Cross-entropy loss
                    double prob = Math.max(1e-15, probs[i][targetClass]);
                    loss -= sampleWeights[i] * Math.log(prob);
                }
            }
            
            // Normalize gradients
            for (int c = 0; c < weights.length; c++) {
                for (int f = 0; f < numFeatures; f++) {
                    gradW[c][f] /= n;
                }
                gradB[c] /= n;
            }
            
            // Add regularization
            double regLoss = addRegularizationGradient(gradW);
            loss = loss / n + regLoss;
            lossHistory[iter] = loss;
            
            // Update with momentum
            for (int c = 0; c < weights.length; c++) {
                for (int f = 0; f < numFeatures; f++) {
                    velocityW[c][f] = momentum * velocityW[c][f] - lr * gradW[c][f];
                    weights[c][f] += velocityW[c][f];
                }
                if (fitIntercept) {
                    velocityB[c] = momentum * velocityB[c] - lr * gradB[c];
                    intercepts[c] += velocityB[c];
                }
            }
            
            // Apply L1 proximal operator (soft thresholding) for L1/ElasticNet
            if (regularization == Regularization.L1 || regularization == Regularization.ELASTICNET) {
                double lambda = getL1Lambda();
                for (int c = 0; c < weights.length; c++) {
                    for (int f = 0; f < numFeatures; f++) {
                        weights[c][f] = softThreshold(weights[c][f], lr * lambda);
                    }
                }
            }
            
            // Check convergence
            if (iter > 0 && Math.abs(lossHistory[iter] - lossHistory[iter - 1]) < tolerance) {
                lossHistory = Arrays.copyOf(lossHistory, iter + 1);
                break;
            }
            
            // Learning rate decay
            lr = learningRate / (1 + 0.01 * iter);
        }
    }
    
    /**
     * Train using L-BFGS optimization.
     */
    private void trainLBFGS(double[][] X, int[] y, double[] sampleWeights) {
        int n = X.length;
        int paramSize = weights.length * numFeatures + (fitIntercept ? intercepts.length : 0);
        
        // Flatten parameters
        double[] params = new double[paramSize];
        int idx = 0;
        for (int c = 0; c < weights.length; c++) {
            for (int f = 0; f < numFeatures; f++) {
                params[idx++] = weights[c][f];
            }
        }
        if (fitIntercept) {
            for (int c = 0; c < intercepts.length; c++) {
                params[idx++] = intercepts[c];
            }
        }
        
        // L-BFGS parameters
        int m = 10;  // Memory size
        double[][] s = new double[m][paramSize];
        double[][] yHist = new double[m][paramSize];
        double[] rho = new double[m];
        int historySize = 0;
        int historyIndex = 0;
        
        double[] prevParams = new double[paramSize];
        double[] prevGrad = new double[paramSize];
        
        lossHistory = new double[maxIterations];
        
        for (int iter = 0; iter < maxIterations; iter++) {
            // Unflatten parameters
            unflattenParameters(params);
            
            // Compute loss and gradient
            double[] gradFlat = new double[paramSize];
            double loss = computeLossAndGradient(X, y, sampleWeights, gradFlat);
            lossHistory[iter] = loss;
            
            // Check convergence
            double gradNorm = 0;
            for (double g : gradFlat) {
                gradNorm += g * g;
            }
            gradNorm = Math.sqrt(gradNorm);
            
            if (gradNorm < tolerance) {
                lossHistory = Arrays.copyOf(lossHistory, iter + 1);
                break;
            }
            
            // Compute search direction using L-BFGS two-loop recursion
            double[] q = gradFlat.clone();
            double[] alpha = new double[m];
            
            for (int i = historySize - 1; i >= 0; i--) {
                int j = (historyIndex - 1 - i + m) % m;
                if (j < 0) j += m;
                alpha[j] = rho[j] * dot(s[j], q);
                for (int k = 0; k < paramSize; k++) {
                    q[k] -= alpha[j] * yHist[j][k];
                }
            }
            
            // Initial Hessian approximation
            double gamma = 1.0;
            if (historySize > 0) {
                int lastIdx = (historyIndex - 1 + m) % m;
                double sy = dot(s[lastIdx], yHist[lastIdx]);
                double yy = dot(yHist[lastIdx], yHist[lastIdx]);
                if (yy > 0) {
                    gamma = sy / yy;
                }
            }
            
            double[] r = new double[paramSize];
            for (int k = 0; k < paramSize; k++) {
                r[k] = gamma * q[k];
            }
            
            for (int i = 0; i < historySize; i++) {
                int j = (historyIndex - historySize + i + m) % m;
                if (j < 0) j += m;
                double beta = rho[j] * dot(yHist[j], r);
                for (int k = 0; k < paramSize; k++) {
                    r[k] += s[j][k] * (alpha[j] - beta);
                }
            }
            
            // Search direction (negative gradient direction)
            for (int k = 0; k < paramSize; k++) {
                r[k] = -r[k];
            }
            
            // Line search with Armijo condition
            double stepSize = 1.0;
            double c1 = 1e-4;
            double gradDotDir = dot(gradFlat, r);
            
            double[] newParams = new double[paramSize];
            for (int ls = 0; ls < 20; ls++) {
                for (int k = 0; k < paramSize; k++) {
                    newParams[k] = params[k] + stepSize * r[k];
                }
                
                unflattenParameters(newParams);
                double newLoss = computeLoss(X, y, sampleWeights);
                
                if (newLoss <= loss + c1 * stepSize * gradDotDir) {
                    break;
                }
                stepSize *= 0.5;
            }
            
            // Update history
            if (iter > 0) {
                int currentIdx = historyIndex % m;
                for (int k = 0; k < paramSize; k++) {
                    s[currentIdx][k] = newParams[k] - prevParams[k];
                    yHist[currentIdx][k] = gradFlat[k] - prevGrad[k];
                }
                double sy = dot(s[currentIdx], yHist[currentIdx]);
                rho[currentIdx] = (sy > 0) ? 1.0 / sy : 0;
                
                historyIndex = (historyIndex + 1) % m;
                if (historySize < m) {
                    historySize++;
                }
            }
            
            // Save current state
            System.arraycopy(params, 0, prevParams, 0, paramSize);
            System.arraycopy(gradFlat, 0, prevGrad, 0, paramSize);
            System.arraycopy(newParams, 0, params, 0, paramSize);
        }
        
        // Final unflatten
        unflattenParameters(params);
    }
    
    private void unflattenParameters(double[] params) {
        int idx = 0;
        for (int c = 0; c < weights.length; c++) {
            for (int f = 0; f < numFeatures; f++) {
                weights[c][f] = params[idx++];
            }
        }
        if (fitIntercept) {
            for (int c = 0; c < intercepts.length; c++) {
                intercepts[c] = params[idx++];
            }
        }
    }
    
    private double computeLoss(double[][] X, int[] y, double[] sampleWeights) {
        int n = X.length;
        double[][] probs = computeProbabilities(X);
        
        double loss = 0.0;
        if (numClasses == 2) {
            for (int i = 0; i < n; i++) {
                double prob = Math.max(1e-15, Math.min(1 - 1e-15, probs[i][0]));
                double target = y[i];
                loss -= sampleWeights[i] * (target * Math.log(prob) + (1 - target) * Math.log(1 - prob));
            }
        } else {
            for (int i = 0; i < n; i++) {
                double prob = Math.max(1e-15, probs[i][y[i]]);
                loss -= sampleWeights[i] * Math.log(prob);
            }
        }
        
        loss /= n;
        loss += computeRegularizationLoss();
        
        return loss;
    }
    
    private double computeLossAndGradient(double[][] X, int[] y, double[] sampleWeights, double[] gradFlat) {
        int n = X.length;
        double[][] probs = computeProbabilities(X);
        
        double[][] gradW = new double[weights.length][numFeatures];
        double[] gradB = new double[intercepts.length];
        
        double loss = 0.0;
        
        if (numClasses == 2) {
            for (int i = 0; i < n; i++) {
                double prob = probs[i][0];
                double target = y[i];
                double error = (prob - target) * sampleWeights[i];
                
                for (int f = 0; f < numFeatures; f++) {
                    gradW[0][f] += error * X[i][f];
                }
                if (fitIntercept) {
                    gradB[0] += error;
                }
                
                prob = Math.max(1e-15, Math.min(1 - 1e-15, prob));
                loss -= sampleWeights[i] * (target * Math.log(prob) + (1 - target) * Math.log(1 - prob));
            }
        } else {
            for (int i = 0; i < n; i++) {
                int targetClass = y[i];
                for (int c = 0; c < numClasses; c++) {
                    double prob = probs[i][c];
                    double target = (c == targetClass) ? 1.0 : 0.0;
                    double error = (prob - target) * sampleWeights[i];
                    
                    for (int f = 0; f < numFeatures; f++) {
                        gradW[c][f] += error * X[i][f];
                    }
                    if (fitIntercept) {
                        gradB[c] += error;
                    }
                }
                
                double prob = Math.max(1e-15, probs[i][targetClass]);
                loss -= sampleWeights[i] * Math.log(prob);
            }
        }
        
        // Normalize
        for (int c = 0; c < weights.length; c++) {
            for (int f = 0; f < numFeatures; f++) {
                gradW[c][f] /= n;
            }
            gradB[c] /= n;
        }
        
        loss /= n;
        
        // Add regularization
        double regLoss = addRegularizationGradient(gradW);
        loss += regLoss;
        
        // Flatten gradients
        int idx = 0;
        for (int c = 0; c < weights.length; c++) {
            for (int f = 0; f < numFeatures; f++) {
                gradFlat[idx++] = gradW[c][f];
            }
        }
        if (fitIntercept) {
            for (int c = 0; c < intercepts.length; c++) {
                gradFlat[idx++] = gradB[c];
            }
        }
        
        return loss;
    }
    
    private double[][] computeProbabilities(double[][] X) {
        int n = X.length;
        
        if (numClasses == 2) {
            double[][] probs = new double[n][1];
            for (int i = 0; i < n; i++) {
                double z = intercepts[0];
                for (int f = 0; f < numFeatures; f++) {
                    z += weights[0][f] * X[i][f];
                }
                probs[i][0] = sigmoid(z);
            }
            return probs;
        } else {
            double[][] probs = new double[n][numClasses];
            for (int i = 0; i < n; i++) {
                double[] logits = new double[numClasses];
                double maxLogit = Double.NEGATIVE_INFINITY;
                
                for (int c = 0; c < numClasses; c++) {
                    logits[c] = intercepts[c];
                    for (int f = 0; f < numFeatures; f++) {
                        logits[c] += weights[c][f] * X[i][f];
                    }
                    maxLogit = Math.max(maxLogit, logits[c]);
                }
                
                // Softmax with numerical stability
                double sumExp = 0;
                for (int c = 0; c < numClasses; c++) {
                    probs[i][c] = Math.exp(logits[c] - maxLogit);
                    sumExp += probs[i][c];
                }
                for (int c = 0; c < numClasses; c++) {
                    probs[i][c] /= sumExp;
                }
            }
            return probs;
        }
    }
    
    private double addRegularizationGradient(double[][] gradW) {
        double lambda = 1.0 / C;
        double loss = 0.0;
        
        switch (regularization) {
            case L2:
                for (int c = 0; c < weights.length; c++) {
                    for (int f = 0; f < numFeatures; f++) {
                        gradW[c][f] += lambda * weights[c][f];
                        loss += 0.5 * lambda * weights[c][f] * weights[c][f];
                    }
                }
                break;
                
            case L1:
                for (int c = 0; c < weights.length; c++) {
                    for (int f = 0; f < numFeatures; f++) {
                        // Subgradient for L1
                        if (weights[c][f] > 0) {
                            gradW[c][f] += lambda;
                        } else if (weights[c][f] < 0) {
                            gradW[c][f] -= lambda;
                        }
                        loss += lambda * Math.abs(weights[c][f]);
                    }
                }
                break;
                
            case ELASTICNET:
                double l1Lambda = lambda * l1Ratio;
                double l2Lambda = lambda * (1 - l1Ratio);
                for (int c = 0; c < weights.length; c++) {
                    for (int f = 0; f < numFeatures; f++) {
                        // L2 part
                        gradW[c][f] += l2Lambda * weights[c][f];
                        loss += 0.5 * l2Lambda * weights[c][f] * weights[c][f];
                        // L1 part (subgradient)
                        if (weights[c][f] > 0) {
                            gradW[c][f] += l1Lambda;
                        } else if (weights[c][f] < 0) {
                            gradW[c][f] -= l1Lambda;
                        }
                        loss += l1Lambda * Math.abs(weights[c][f]);
                    }
                }
                break;
                
            case NONE:
            default:
                break;
        }
        
        return loss;
    }
    
    private double computeRegularizationLoss() {
        double lambda = 1.0 / C;
        double loss = 0.0;
        
        switch (regularization) {
            case L2:
                for (int c = 0; c < weights.length; c++) {
                    for (int f = 0; f < numFeatures; f++) {
                        loss += 0.5 * lambda * weights[c][f] * weights[c][f];
                    }
                }
                break;
                
            case L1:
                for (int c = 0; c < weights.length; c++) {
                    for (int f = 0; f < numFeatures; f++) {
                        loss += lambda * Math.abs(weights[c][f]);
                    }
                }
                break;
                
            case ELASTICNET:
                double l1Lambda = lambda * l1Ratio;
                double l2Lambda = lambda * (1 - l1Ratio);
                for (int c = 0; c < weights.length; c++) {
                    for (int f = 0; f < numFeatures; f++) {
                        loss += 0.5 * l2Lambda * weights[c][f] * weights[c][f];
                        loss += l1Lambda * Math.abs(weights[c][f]);
                    }
                }
                break;
                
            case NONE:
            default:
                break;
        }
        
        return loss;
    }
    
    private double getL1Lambda() {
        double lambda = 1.0 / C;
        if (regularization == Regularization.L1) {
            return lambda;
        } else if (regularization == Regularization.ELASTICNET) {
            return lambda * l1Ratio;
        }
        return 0;
    }
    
    private double softThreshold(double x, double threshold) {
        if (x > threshold) {
            return x - threshold;
        } else if (x < -threshold) {
            return x + threshold;
        }
        return 0;
    }
    
    private double sigmoid(double x) {
        if (x >= 0) {
            return 1.0 / (1.0 + Math.exp(-x));
        } else {
            double expX = Math.exp(x);
            return expX / (1.0 + expX);
        }
    }
    
    private double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    @Override
    public int predict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException("Input must have " + numFeatures + " features");
        }
        
        double[] proba = predictProba(x);
        int maxIdx = 0;
        for (int i = 1; i < proba.length; i++) {
            if (proba[i] > proba[maxIdx]) {
                maxIdx = i;
            }
        }
        return classes[maxIdx];
    }
    
    /**
     * Predicts class labels for multiple inputs.
     * 
     * @param X array of input features
     * @return array of predicted class labels
     */
    public int[] predict(double[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    @Override
    public double[] predictProba(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException("Input must have " + numFeatures + " features");
        }
        
        if (numClasses == 2) {
            double z = intercepts[0];
            for (int f = 0; f < numFeatures; f++) {
                z += weights[0][f] * x[f];
            }
            double prob = sigmoid(z);
            return new double[]{1 - prob, prob};
        } else {
            double[] logits = new double[numClasses];
            double maxLogit = Double.NEGATIVE_INFINITY;
            
            for (int c = 0; c < numClasses; c++) {
                logits[c] = intercepts[c];
                for (int f = 0; f < numFeatures; f++) {
                    logits[c] += weights[c][f] * x[f];
                }
                maxLogit = Math.max(maxLogit, logits[c]);
            }
            
            // Softmax
            double[] probs = new double[numClasses];
            double sumExp = 0;
            for (int c = 0; c < numClasses; c++) {
                probs[c] = Math.exp(logits[c] - maxLogit);
                sumExp += probs[c];
            }
            for (int c = 0; c < numClasses; c++) {
                probs[c] /= sumExp;
            }
            
            return probs;
        }
    }
    
    /**
     * Predicts class probabilities for multiple inputs.
     * 
     * @param X array of input features
     * @return 2D array where each row contains probabilities for each class
     */
    public double[][] predictProba(double[][] X) {
        double[][] probabilities = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            probabilities[i] = predictProba(X[i]);
        }
        return probabilities;
    }
    
    /**
     * Returns the decision function value for binary classification.
     * 
     * @param x input features
     * @return decision value (positive for class 1, negative for class 0)
     */
    public double decisionFunction(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (numClasses != 2) {
            throw new IllegalStateException("decision_function is only available for binary classification");
        }
        
        double z = intercepts[0];
        for (int f = 0; f < numFeatures; f++) {
            z += weights[0][f] * x[f];
        }
        return z;
    }
    
    @Override
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * Returns the learned coefficients (weights).
     * 
     * @return 2D array of coefficients (shape: numClasses x numFeatures)
     */
    public double[][] getCoefficients() {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained first");
        }
        double[][] coef = new double[weights.length][];
        for (int i = 0; i < weights.length; i++) {
            coef[i] = weights[i].clone();
        }
        return coef;
    }
    
    /**
     * Returns the learned intercepts.
     * 
     * @return array of intercepts
     */
    public double[] getIntercepts() {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained first");
        }
        return intercepts.clone();
    }
    
    /**
     * Returns the unique classes.
     * 
     * @return array of class labels
     */
    public int[] getClasses() {
        return classes != null ? classes.clone() : null;
    }
    
    /**
     * Returns the loss history during training.
     * 
     * @return array of loss values per iteration
     */
    public double[] getLossHistory() {
        return lossHistory != null ? lossHistory.clone() : null;
    }
    
    /**
     * Checks if the model has been trained.
     * 
     * @return true if model is fitted
     */
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Returns the sparsity of the model (percentage of zero coefficients).
     * Useful for L1 regularized models.
     * 
     * @return sparsity ratio [0, 1]
     */
    public double getSparsity() {
        if (!fitted) return 0;
        
        int totalCoefs = 0;
        int zeroCoefs = 0;
        
        for (double[] w : weights) {
            for (double coef : w) {
                totalCoefs++;
                if (Math.abs(coef) < 1e-10) {
                    zeroCoefs++;
                }
            }
        }
        
        return totalCoefs > 0 ? (double) zeroCoefs / totalCoefs : 0;
    }
    
    @Override
    public String toString() {
        return String.format("LogisticRegression(regularization=%s, C=%.4f, solver=%s, fitted=%s)",
                           regularization, C, solver, fitted);
    }
}
