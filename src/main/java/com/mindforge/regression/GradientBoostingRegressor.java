package com.mindforge.regression;

import java.util.*;

/**
 * Gradient Boosting Regressor for regression tasks.
 * 
 * <p>Gradient Boosting builds an ensemble of weak learners (decision trees) sequentially,
 * where each tree tries to correct the errors (residuals) of the previous ensemble.
 * Uses gradient descent to minimize a differentiable loss function.</p>
 * 
 * <p>Features:</p>
 * <ul>
 *   <li>Multiple loss functions: MSE (Least Squares), MAE (Least Absolute Deviation), Huber</li>
 *   <li>Configurable learning rate (shrinkage)</li>
 *   <li>Stochastic gradient boosting with subsampling</li>
 *   <li>Feature importance calculation</li>
 *   <li>Early stopping based on validation score</li>
 *   <li>Staged predictions for learning curve analysis</li>
 * </ul>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}};
 * double[] y = {1.5, 2.5, 3.5, 4.5};
 * 
 * GradientBoostingRegressor gb = new GradientBoostingRegressor.Builder()
 *     .nEstimators(100)
 *     .learningRate(0.1)
 *     .maxDepth(3)
 *     .loss(GradientBoostingRegressor.Loss.SQUARED_ERROR)
 *     .build();
 * 
 * gb.fit(X, y);
 * double prediction = gb.predict(new double[]{2.5, 3.5});
 * }</pre>
 * 
 * @author MindForge Team
 * @version 1.2.0-alpha
 */
public class GradientBoostingRegressor implements Regressor<double[]> {
    
    /**
     * Loss function for gradient boosting.
     */
    public enum Loss {
        /** Least Squares (L2 loss) - standard regression */
        SQUARED_ERROR,
        /** Least Absolute Deviation (L1 loss) - robust to outliers */
        ABSOLUTE_ERROR,
        /** Huber loss - smooth transition between L1 and L2 */
        HUBER,
        /** Quantile loss - for quantile regression */
        QUANTILE
    }
    
    /**
     * Internal tree node for gradient boosting.
     */
    private static class GBTree {
        int featureIndex;
        double threshold;
        GBTree left;
        GBTree right;
        boolean isLeaf;
        double leafValue;
        
        double predict(double[] x) {
            if (isLeaf) {
                return leafValue;
            }
            if (x[featureIndex] <= threshold) {
                return left != null ? left.predict(x) : leafValue;
            } else {
                return right != null ? right.predict(x) : leafValue;
            }
        }
    }
    
    // Hyperparameters
    private final int nEstimators;
    private final double learningRate;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int minSamplesLeaf;
    private final double subsample;
    private final Loss loss;
    private final double alpha;  // For Huber and Quantile loss
    private final Integer maxFeatures;
    private final double minImpurityDecrease;
    private final int randomState;
    private final double validationFraction;
    private final int nIterNoChange;
    private final double tol;
    
    // Model state
    private List<GBTree> trees;
    private double initialPrediction;
    private double[] featureImportance;
    private int numFeatures;
    private boolean fitted;
    private Random random;
    
    // Training history
    private double[] trainLossHistory;
    private double[] validLossHistory;
    private int nEstimatorsFitted;
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private GradientBoostingRegressor(Builder builder) {
        this.nEstimators = builder.nEstimators;
        this.learningRate = builder.learningRate;
        this.maxDepth = builder.maxDepth;
        this.minSamplesSplit = builder.minSamplesSplit;
        this.minSamplesLeaf = builder.minSamplesLeaf;
        this.subsample = builder.subsample;
        this.loss = builder.loss;
        this.alpha = builder.alpha;
        this.maxFeatures = builder.maxFeatures;
        this.minImpurityDecrease = builder.minImpurityDecrease;
        this.randomState = builder.randomState;
        this.validationFraction = builder.validationFraction;
        this.nIterNoChange = builder.nIterNoChange;
        this.tol = builder.tol;
        this.fitted = false;
        this.random = new Random(randomState);
        this.trees = new ArrayList<>();
    }
    
    /**
     * Builder pattern for creating GradientBoostingRegressor instances.
     */
    public static class Builder {
        private int nEstimators = 100;
        private double learningRate = 0.1;
        private int maxDepth = 3;
        private int minSamplesSplit = 2;
        private int minSamplesLeaf = 1;
        private double subsample = 1.0;
        private Loss loss = Loss.SQUARED_ERROR;
        private double alpha = 0.9;
        private Integer maxFeatures = null;
        private double minImpurityDecrease = 0.0;
        private int randomState = 42;
        private double validationFraction = 0.1;
        private int nIterNoChange = 10;
        private double tol = 1e-4;
        
        public Builder nEstimators(int nEstimators) {
            if (nEstimators < 1) {
                throw new IllegalArgumentException("nEstimators must be at least 1");
            }
            this.nEstimators = nEstimators;
            return this;
        }
        
        public Builder learningRate(double learningRate) {
            if (learningRate <= 0 || learningRate > 1) {
                throw new IllegalArgumentException("learningRate must be in (0, 1]");
            }
            this.learningRate = learningRate;
            return this;
        }
        
        public Builder maxDepth(int maxDepth) {
            if (maxDepth < 1) {
                throw new IllegalArgumentException("maxDepth must be at least 1");
            }
            this.maxDepth = maxDepth;
            return this;
        }
        
        public Builder minSamplesSplit(int minSamplesSplit) {
            if (minSamplesSplit < 2) {
                throw new IllegalArgumentException("minSamplesSplit must be at least 2");
            }
            this.minSamplesSplit = minSamplesSplit;
            return this;
        }
        
        public Builder minSamplesLeaf(int minSamplesLeaf) {
            if (minSamplesLeaf < 1) {
                throw new IllegalArgumentException("minSamplesLeaf must be at least 1");
            }
            this.minSamplesLeaf = minSamplesLeaf;
            return this;
        }
        
        public Builder subsample(double subsample) {
            if (subsample <= 0 || subsample > 1) {
                throw new IllegalArgumentException("subsample must be in (0, 1]");
            }
            this.subsample = subsample;
            return this;
        }
        
        public Builder loss(Loss loss) {
            this.loss = loss;
            return this;
        }
        
        public Builder alpha(double alpha) {
            if (alpha <= 0 || alpha >= 1) {
                throw new IllegalArgumentException("alpha must be in (0, 1)");
            }
            this.alpha = alpha;
            return this;
        }
        
        public Builder maxFeatures(Integer maxFeatures) {
            if (maxFeatures != null && maxFeatures <= 0) {
                throw new IllegalArgumentException("maxFeatures must be positive");
            }
            this.maxFeatures = maxFeatures;
            return this;
        }
        
        public Builder minImpurityDecrease(double minImpurityDecrease) {
            if (minImpurityDecrease < 0) {
                throw new IllegalArgumentException("minImpurityDecrease must be non-negative");
            }
            this.minImpurityDecrease = minImpurityDecrease;
            return this;
        }
        
        public Builder randomState(int randomState) {
            this.randomState = randomState;
            return this;
        }
        
        public Builder validationFraction(double validationFraction) {
            if (validationFraction < 0 || validationFraction >= 1) {
                throw new IllegalArgumentException("validationFraction must be in [0, 1)");
            }
            this.validationFraction = validationFraction;
            return this;
        }
        
        public Builder nIterNoChange(int nIterNoChange) {
            if (nIterNoChange < 1) {
                throw new IllegalArgumentException("nIterNoChange must be at least 1");
            }
            this.nIterNoChange = nIterNoChange;
            return this;
        }
        
        public Builder tol(double tol) {
            if (tol < 0) {
                throw new IllegalArgumentException("tol must be non-negative");
            }
            this.tol = tol;
            return this;
        }
        
        public GradientBoostingRegressor build() {
            return new GradientBoostingRegressor(this);
        }
    }
    
    /**
     * Default constructor with default hyperparameters.
     */
    public GradientBoostingRegressor() {
        this(new Builder());
    }
    
    /**
     * Simple constructor with main hyperparameters.
     */
    public GradientBoostingRegressor(int nEstimators, double learningRate, int maxDepth) {
        this(new Builder()
            .nEstimators(nEstimators)
            .learningRate(learningRate)
            .maxDepth(maxDepth));
    }
    
    @Override
    public void train(double[][] X, double[] y) {
        fit(X, y);
    }
    
    /**
     * Fit the gradient boosting regressor.
     * 
     * @param X Training feature matrix (n_samples x n_features)
     * @param y Training target values (n_samples)
     */
    public void fit(double[][] X, double[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Training data cannot be null");
        }
        if (X.length == 0 || X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same length and be non-empty");
        }
        
        int n = X.length;
        this.numFeatures = X[0].length;
        this.featureImportance = new double[numFeatures];
        trees.clear();
        
        // Split data for early stopping if needed
        double[][] XTrain = X;
        double[] yTrain = y;
        double[][] XValid = null;
        double[] yValid = null;
        
        boolean useEarlyStopping = validationFraction > 0 && nIterNoChange > 0;
        
        if (useEarlyStopping) {
            int validSize = (int) (n * validationFraction);
            int trainSize = n - validSize;
            
            // Shuffle indices
            int[] indices = new int[n];
            for (int i = 0; i < n; i++) indices[i] = i;
            for (int i = n - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
            
            XTrain = new double[trainSize][];
            yTrain = new double[trainSize];
            XValid = new double[validSize][];
            yValid = new double[validSize];
            
            for (int i = 0; i < trainSize; i++) {
                XTrain[i] = X[indices[i]];
                yTrain[i] = y[indices[i]];
            }
            for (int i = 0; i < validSize; i++) {
                XValid[i] = X[indices[trainSize + i]];
                yValid[i] = y[indices[trainSize + i]];
            }
        }
        
        int nTrain = XTrain.length;
        
        // Initialize with constant prediction
        initialPrediction = computeInitialPrediction(yTrain);
        
        // Current predictions
        double[] predictions = new double[nTrain];
        Arrays.fill(predictions, initialPrediction);
        
        double[] validPredictions = null;
        if (useEarlyStopping) {
            validPredictions = new double[XValid.length];
            Arrays.fill(validPredictions, initialPrediction);
        }
        
        // Training history
        trainLossHistory = new double[nEstimators];
        validLossHistory = useEarlyStopping ? new double[nEstimators] : null;
        
        // Early stopping state
        double bestValidLoss = Double.POSITIVE_INFINITY;
        int noImprovementCount = 0;
        
        // Boosting iterations
        for (int iter = 0; iter < nEstimators; iter++) {
            // Compute negative gradient (pseudo-residuals)
            double[] residuals = computeNegativeGradient(yTrain, predictions);
            
            // Subsample if needed
            int[] sampleIndices;
            if (subsample < 1.0) {
                int sampleSize = Math.max(1, (int) (nTrain * subsample));
                sampleIndices = new int[sampleSize];
                for (int i = 0; i < sampleSize; i++) {
                    sampleIndices[i] = random.nextInt(nTrain);
                }
            } else {
                sampleIndices = new int[nTrain];
                for (int i = 0; i < nTrain; i++) {
                    sampleIndices[i] = i;
                }
            }
            
            // Fit tree to residuals
            GBTree tree = fitTree(XTrain, residuals, yTrain, predictions, sampleIndices, 1);
            trees.add(tree);
            
            // Update predictions
            for (int i = 0; i < nTrain; i++) {
                predictions[i] += learningRate * tree.predict(XTrain[i]);
            }
            
            // Compute training loss
            trainLossHistory[iter] = computeLoss(yTrain, predictions);
            
            // Early stopping check
            if (useEarlyStopping) {
                for (int i = 0; i < XValid.length; i++) {
                    validPredictions[i] += learningRate * tree.predict(XValid[i]);
                }
                validLossHistory[iter] = computeLoss(yValid, validPredictions);
                
                if (validLossHistory[iter] < bestValidLoss - tol) {
                    bestValidLoss = validLossHistory[iter];
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                }
                
                if (noImprovementCount >= nIterNoChange) {
                    // Early stop
                    trainLossHistory = Arrays.copyOf(trainLossHistory, iter + 1);
                    validLossHistory = Arrays.copyOf(validLossHistory, iter + 1);
                    break;
                }
            }
        }
        
        nEstimatorsFitted = trees.size();
        
        // Normalize feature importance
        double totalImportance = Arrays.stream(featureImportance).sum();
        if (totalImportance > 0) {
            for (int i = 0; i < numFeatures; i++) {
                featureImportance[i] /= totalImportance;
            }
        }
        
        this.fitted = true;
    }
    
    private double computeInitialPrediction(double[] y) {
        switch (loss) {
            case ABSOLUTE_ERROR:
                // Median
                double[] sorted = y.clone();
                Arrays.sort(sorted);
                int mid = sorted.length / 2;
                if (sorted.length % 2 == 0) {
                    return (sorted[mid - 1] + sorted[mid]) / 2;
                }
                return sorted[mid];
                
            case QUANTILE:
                // Quantile
                double[] sortedQ = y.clone();
                Arrays.sort(sortedQ);
                int idx = (int) (sortedQ.length * alpha);
                return sortedQ[Math.min(idx, sortedQ.length - 1)];
                
            case HUBER:
            case SQUARED_ERROR:
            default:
                // Mean
                return Arrays.stream(y).average().orElse(0);
        }
    }
    
    private double[] computeNegativeGradient(double[] y, double[] predictions) {
        int n = y.length;
        double[] gradient = new double[n];
        
        switch (loss) {
            case ABSOLUTE_ERROR:
                for (int i = 0; i < n; i++) {
                    double residual = y[i] - predictions[i];
                    gradient[i] = Math.signum(residual);
                }
                break;
                
            case HUBER:
                // Compute threshold
                double[] absResiduals = new double[n];
                for (int i = 0; i < n; i++) {
                    absResiduals[i] = Math.abs(y[i] - predictions[i]);
                }
                Arrays.sort(absResiduals);
                double delta = absResiduals[(int) (n * alpha)];
                
                for (int i = 0; i < n; i++) {
                    double residual = y[i] - predictions[i];
                    if (Math.abs(residual) <= delta) {
                        gradient[i] = residual;
                    } else {
                        gradient[i] = delta * Math.signum(residual);
                    }
                }
                break;
                
            case QUANTILE:
                for (int i = 0; i < n; i++) {
                    double residual = y[i] - predictions[i];
                    gradient[i] = residual >= 0 ? alpha : alpha - 1;
                }
                break;
                
            case SQUARED_ERROR:
            default:
                for (int i = 0; i < n; i++) {
                    gradient[i] = y[i] - predictions[i];
                }
                break;
        }
        
        return gradient;
    }
    
    private double computeLoss(double[] y, double[] predictions) {
        int n = y.length;
        double loss = 0;
        
        switch (this.loss) {
            case ABSOLUTE_ERROR:
                for (int i = 0; i < n; i++) {
                    loss += Math.abs(y[i] - predictions[i]);
                }
                break;
                
            case HUBER:
                double[] absResiduals = new double[n];
                for (int i = 0; i < n; i++) {
                    absResiduals[i] = Math.abs(y[i] - predictions[i]);
                }
                double[] sortedRes = absResiduals.clone();
                Arrays.sort(sortedRes);
                double delta = sortedRes[(int) (n * alpha)];
                
                for (int i = 0; i < n; i++) {
                    double absR = absResiduals[i];
                    if (absR <= delta) {
                        loss += 0.5 * absR * absR;
                    } else {
                        loss += delta * absR - 0.5 * delta * delta;
                    }
                }
                break;
                
            case QUANTILE:
                for (int i = 0; i < n; i++) {
                    double residual = y[i] - predictions[i];
                    loss += residual >= 0 ? alpha * residual : (alpha - 1) * residual;
                }
                break;
                
            case SQUARED_ERROR:
            default:
                for (int i = 0; i < n; i++) {
                    double error = y[i] - predictions[i];
                    loss += error * error;
                }
                loss /= 2;
                break;
        }
        
        return loss / n;
    }
    
    private GBTree fitTree(double[][] X, double[] residuals, double[] y, 
                           double[] predictions, int[] indices, int depth) {
        GBTree node = new GBTree();
        
        if (depth >= maxDepth || indices.length < minSamplesSplit) {
            node.isLeaf = true;
            node.leafValue = computeLeafValue(residuals, y, predictions, indices);
            return node;
        }
        
        // Find best split
        Split bestSplit = findBestSplit(X, residuals, indices);
        
        if (bestSplit == null || 
            bestSplit.leftIndices.length < minSamplesLeaf || 
            bestSplit.rightIndices.length < minSamplesLeaf ||
            bestSplit.gain < minImpurityDecrease) {
            node.isLeaf = true;
            node.leafValue = computeLeafValue(residuals, y, predictions, indices);
            return node;
        }
        
        node.isLeaf = false;
        node.featureIndex = bestSplit.featureIndex;
        node.threshold = bestSplit.threshold;
        
        // Update feature importance
        featureImportance[bestSplit.featureIndex] += bestSplit.gain * indices.length;
        
        // Recursively build children
        node.left = fitTree(X, residuals, y, predictions, bestSplit.leftIndices, depth + 1);
        node.right = fitTree(X, residuals, y, predictions, bestSplit.rightIndices, depth + 1);
        
        return node;
    }
    
    private double computeLeafValue(double[] residuals, double[] y, 
                                     double[] predictions, int[] indices) {
        if (indices.length == 0) return 0;
        
        switch (loss) {
            case ABSOLUTE_ERROR:
                // Median of residuals
                double[] sortedResiduals = new double[indices.length];
                for (int i = 0; i < indices.length; i++) {
                    sortedResiduals[i] = residuals[indices[i]];
                }
                Arrays.sort(sortedResiduals);
                int mid = sortedResiduals.length / 2;
                if (sortedResiduals.length % 2 == 0) {
                    return (sortedResiduals[mid - 1] + sortedResiduals[mid]) / 2;
                }
                return sortedResiduals[mid];
                
            case HUBER:
                // Use line search or approximate with median
                double[] sortedRes = new double[indices.length];
                for (int i = 0; i < indices.length; i++) {
                    sortedRes[i] = y[indices[i]] - predictions[indices[i]];
                }
                Arrays.sort(sortedRes);
                int midIdx = sortedRes.length / 2;
                return sortedRes[midIdx];
                
            case QUANTILE:
                // Quantile of residuals
                double[] sortedQ = new double[indices.length];
                for (int i = 0; i < indices.length; i++) {
                    sortedQ[i] = residuals[indices[i]];
                }
                Arrays.sort(sortedQ);
                int qIdx = (int) (sortedQ.length * alpha);
                return sortedQ[Math.min(qIdx, sortedQ.length - 1)];
                
            case SQUARED_ERROR:
            default:
                // Mean of residuals
                double sum = 0;
                for (int idx : indices) {
                    sum += residuals[idx];
                }
                return sum / indices.length;
        }
    }
    
    private static class Split {
        int featureIndex;
        double threshold;
        int[] leftIndices;
        int[] rightIndices;
        double gain;
    }
    
    private Split findBestSplit(double[][] X, double[] residuals, int[] indices) {
        Split bestSplit = null;
        double bestGain = Double.NEGATIVE_INFINITY;
        
        // Calculate parent statistics
        double parentSum = 0;
        double parentSumSq = 0;
        for (int idx : indices) {
            parentSum += residuals[idx];
            parentSumSq += residuals[idx] * residuals[idx];
        }
        double parentVariance = parentSumSq / indices.length - 
                               (parentSum / indices.length) * (parentSum / indices.length);
        
        // Select features
        int[] featuresToTry;
        if (maxFeatures != null && maxFeatures < numFeatures) {
            featuresToTry = random.ints(0, numFeatures)
                                  .distinct()
                                  .limit(maxFeatures)
                                  .toArray();
        } else {
            featuresToTry = new int[numFeatures];
            for (int i = 0; i < numFeatures; i++) featuresToTry[i] = i;
        }
        
        for (int featureIdx : featuresToTry) {
            // Sort by feature value
            double[][] pairs = new double[indices.length][2];
            for (int i = 0; i < indices.length; i++) {
                pairs[i][0] = X[indices[i]][featureIdx];
                pairs[i][1] = indices[i];
            }
            Arrays.sort(pairs, Comparator.comparingDouble(a -> a[0]));
            
            // Compute cumulative sums
            double leftSum = 0;
            double leftSumSq = 0;
            double rightSum = parentSum;
            double rightSumSq = parentSumSq;
            
            for (int i = 0; i < indices.length - 1; i++) {
                int idx = (int) pairs[i][1];
                double val = residuals[idx];
                
                leftSum += val;
                leftSumSq += val * val;
                rightSum -= val;
                rightSumSq -= val * val;
                
                int leftCount = i + 1;
                int rightCount = indices.length - leftCount;
                
                // Skip if same value as next
                if (pairs[i][0] == pairs[i + 1][0]) continue;
                
                // Skip if would violate minSamplesLeaf
                if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) continue;
                
                double threshold = (pairs[i][0] + pairs[i + 1][0]) / 2.0;
                
                // Calculate variance reduction
                double leftMean = leftSum / leftCount;
                double rightMean = rightSum / rightCount;
                
                double leftVar = leftSumSq / leftCount - leftMean * leftMean;
                double rightVar = rightSumSq / rightCount - rightMean * rightMean;
                
                double weightedChildVar = (leftCount * leftVar + rightCount * rightVar) / indices.length;
                double gain = parentVariance - weightedChildVar;
                
                if (gain > bestGain) {
                    bestGain = gain;
                    
                    int[] leftIndices = new int[leftCount];
                    int[] rightIndices = new int[rightCount];
                    for (int j = 0; j <= i; j++) {
                        leftIndices[j] = (int) pairs[j][1];
                    }
                    for (int j = i + 1; j < indices.length; j++) {
                        rightIndices[j - i - 1] = (int) pairs[j][1];
                    }
                    
                    bestSplit = new Split();
                    bestSplit.featureIndex = featureIdx;
                    bestSplit.threshold = threshold;
                    bestSplit.leftIndices = leftIndices;
                    bestSplit.rightIndices = rightIndices;
                    bestSplit.gain = gain;
                }
            }
        }
        
        return bestSplit;
    }
    
    @Override
    public double predict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException("Input must have " + numFeatures + " features");
        }
        
        double prediction = initialPrediction;
        for (GBTree tree : trees) {
            prediction += learningRate * tree.predict(x);
        }
        return prediction;
    }
    
    /**
     * Predicts values for multiple inputs.
     * 
     * @param X array of input features
     * @return array of predicted values
     */
    public double[] predict(double[][] X) {
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    /**
     * Returns staged predictions (prediction at each boosting iteration).
     * Useful for plotting learning curves.
     * 
     * @param x input features
     * @return array of predictions after each tree is added
     */
    public double[] stagedPredict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        double[] staged = new double[trees.size()];
        double prediction = initialPrediction;
        
        for (int i = 0; i < trees.size(); i++) {
            prediction += learningRate * trees.get(i).predict(x);
            staged[i] = prediction;
        }
        
        return staged;
    }
    
    /**
     * Returns staged predictions for multiple inputs.
     * 
     * @param X array of input features
     * @return 2D array of predictions (samples x iterations)
     */
    public double[][] stagedPredict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        double[][] staged = new double[X.length][trees.size()];
        
        for (int i = 0; i < X.length; i++) {
            staged[i] = stagedPredict(X[i]);
        }
        
        return staged;
    }
    
    /**
     * Applies trees to samples and returns leaf indices.
     * 
     * @param X input features
     * @return 2D array of leaf indices (samples x trees)
     */
    public int[][] apply(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        int[][] leafIndices = new int[X.length][trees.size()];
        for (int i = 0; i < X.length; i++) {
            for (int t = 0; t < trees.size(); t++) {
                leafIndices[i][t] = getLeafIndex(X[i], trees.get(t), 0);
            }
        }
        return leafIndices;
    }
    
    private int getLeafIndex(double[] x, GBTree node, int currentIdx) {
        if (node.isLeaf) {
            return currentIdx;
        }
        if (x[node.featureIndex] <= node.threshold) {
            return getLeafIndex(x, node.left, 2 * currentIdx + 1);
        } else {
            return getLeafIndex(x, node.right, 2 * currentIdx + 2);
        }
    }
    
    /**
     * Get feature importance scores.
     * 
     * @return Array of feature importance values (normalized to sum to 1)
     */
    public double[] getFeatureImportance() {
        if (!fitted) {
            return null;
        }
        return featureImportance.clone();
    }
    
    /**
     * Get training loss history.
     */
    public double[] getTrainLossHistory() {
        return trainLossHistory != null ? trainLossHistory.clone() : null;
    }
    
    /**
     * Get validation loss history.
     */
    public double[] getValidLossHistory() {
        return validLossHistory != null ? validLossHistory.clone() : null;
    }
    
    /**
     * Get number of estimators actually fitted (may be less than nEstimators if early stopping).
     */
    public int getNEstimatorsFitted() {
        return nEstimatorsFitted;
    }
    
    /**
     * Get the configured number of estimators.
     */
    public int getNEstimators() {
        return nEstimators;
    }
    
    /**
     * Get learning rate.
     */
    public double getLearningRate() {
        return learningRate;
    }
    
    /**
     * Checks if the model has been trained.
     */
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Calculate R^2 score on given data.
     */
    public double score(double[][] X, double[] y) {
        double[] predictions = predict(X);
        
        double yMean = Arrays.stream(y).average().orElse(0);
        double ssRes = 0;
        double ssTot = 0;
        
        for (int i = 0; i < y.length; i++) {
            double error = y[i] - predictions[i];
            ssRes += error * error;
            double diff = y[i] - yMean;
            ssTot += diff * diff;
        }
        
        return ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
    }
    
    @Override
    public String toString() {
        return String.format(
            "GradientBoostingRegressor(nEstimators=%d, learningRate=%.4f, maxDepth=%d, loss=%s, fitted=%s)",
            nEstimators, learningRate, maxDepth, loss, fitted
        );
    }
}
