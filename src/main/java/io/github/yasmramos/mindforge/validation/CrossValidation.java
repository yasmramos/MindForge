package io.github.yasmramos.mindforge.validation;

import java.util.*;
import java.util.function.BiFunction;

/**
 * Cross-validation utilities for model evaluation.
 * 
 * Provides various cross-validation strategies:
 * - K-Fold Cross-Validation
 * - Stratified K-Fold Cross-Validation
 * - Leave-One-Out Cross-Validation (LOOCV)
 * - Shuffle Split Cross-Validation
 * - Train-Test Split
 * 
 * Example usage:
 * <pre>
 * // K-Fold validation
 * CrossValidationResult result = CrossValidation.kFold(
 *     (X, y) -> {
 *         KNearestNeighbors knn = new KNearestNeighbors(3);
 *         knn.train(X, y);
 *         return knn;
 *     },
 *     (model, X) -> model.predict(X),
 *     X, y, 5, 42
 * );
 * 
 * System.out.println("Mean Accuracy: " + result.getMean());
 * System.out.println("Std Dev: " + result.getStdDev());
 * </pre>
 */
public class CrossValidation {
    
    /**
     * Functional interface for training a model.
     * 
     * @param <M> model type
     */
    @FunctionalInterface
    public interface ModelTrainer<M> {
        /**
         * Trains a model with the given data.
         * 
         * @param X training features
         * @param y training labels
         * @return trained model
         */
        M train(double[][] X, int[] y);
    }
    
    /**
     * Functional interface for making predictions.
     * 
     * @param <M> model type
     */
    @FunctionalInterface
    public interface ModelPredictor<M> {
        /**
         * Makes predictions using a trained model.
         * 
         * @param model trained model
         * @param X test features
         * @return predictions
         */
        int[] predict(M model, double[][] X);
    }
    
    /**
     * Performs K-Fold cross-validation.
     * 
     * Splits the data into K folds, trains on K-1 folds and tests on the remaining fold,
     * repeating K times with each fold used as test set exactly once.
     * 
     * @param <M> model type
     * @param trainer function to train the model
     * @param predictor function to make predictions
     * @param X feature matrix
     * @param y target labels
     * @param k number of folds
     * @param randomState random seed for reproducibility (use null for no shuffling)
     * @return cross-validation results
     */
    public static <M> CrossValidationResult kFold(
            ModelTrainer<M> trainer,
            ModelPredictor<M> predictor,
            double[][] X,
            int[] y,
            int k,
            Integer randomState) {
        
        if (k <= 1) {
            throw new IllegalArgumentException("Number of folds must be greater than 1");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        if (k > X.length) {
            throw new IllegalArgumentException("Number of folds cannot exceed number of samples");
        }
        
        int n = X.length;
        int[] indices = createIndices(n, randomState);
        List<Double> scores = new ArrayList<>();
        
        int foldSize = n / k;
        
        for (int fold = 0; fold < k; fold++) {
            // Determine test indices for this fold
            int testStart = fold * foldSize;
            int testEnd = (fold == k - 1) ? n : (fold + 1) * foldSize;
            
            // Split data into train and test
            SplitData split = splitByIndices(X, y, indices, testStart, testEnd);
            
            // Train and evaluate
            M model = trainer.train(split.XTrain, split.yTrain);
            int[] predictions = predictor.predict(model, split.XTest);
            double accuracy = Metrics.accuracy(split.yTest, predictions);
            
            scores.add(accuracy);
        }
        
        return new CrossValidationResult(scores, "accuracy");
    }
    
    /**
     * Performs K-Fold cross-validation without shuffling.
     * 
     * @param <M> model type
     * @param trainer function to train the model
     * @param predictor function to make predictions
     * @param X feature matrix
     * @param y target labels
     * @param k number of folds
     * @return cross-validation results
     */
    public static <M> CrossValidationResult kFold(
            ModelTrainer<M> trainer,
            ModelPredictor<M> predictor,
            double[][] X,
            int[] y,
            int k) {
        return kFold(trainer, predictor, X, y, k, null);
    }
    
    /**
     * Performs Stratified K-Fold cross-validation.
     * 
     * Similar to K-Fold but ensures that each fold has approximately the same
     * proportion of samples from each class as the complete dataset.
     * 
     * @param <M> model type
     * @param trainer function to train the model
     * @param predictor function to make predictions
     * @param X feature matrix
     * @param y target labels
     * @param k number of folds
     * @param randomState random seed for reproducibility
     * @return cross-validation results
     */
    public static <M> CrossValidationResult stratifiedKFold(
            ModelTrainer<M> trainer,
            ModelPredictor<M> predictor,
            double[][] X,
            int[] y,
            int k,
            Integer randomState) {
        
        if (k <= 1) {
            throw new IllegalArgumentException("Number of folds must be greater than 1");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        // Group indices by class
        Map<Integer, List<Integer>> classBuckets = new HashMap<>();
        for (int i = 0; i < y.length; i++) {
            classBuckets.computeIfAbsent(y[i], key -> new ArrayList<>()).add(i);
        }
        
        // Shuffle indices within each class
        Random random = randomState != null ? new Random(randomState) : new Random();
        for (List<Integer> bucket : classBuckets.values()) {
            Collections.shuffle(bucket, random);
        }
        
        // Create stratified folds
        List<List<Integer>> folds = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            folds.add(new ArrayList<>());
        }
        
        // Distribute samples from each class across folds
        for (List<Integer> bucket : classBuckets.values()) {
            for (int i = 0; i < bucket.size(); i++) {
                folds.get(i % k).add(bucket.get(i));
            }
        }
        
        List<Double> scores = new ArrayList<>();
        
        // Perform cross-validation
        for (int fold = 0; fold < k; fold++) {
            // Create train and test sets
            List<Integer> testIndices = folds.get(fold);
            List<Integer> trainIndices = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                if (i != fold) {
                    trainIndices.addAll(folds.get(i));
                }
            }
            
            // Extract data
            double[][] XTrain = new double[trainIndices.size()][];
            int[] yTrain = new int[trainIndices.size()];
            double[][] XTest = new double[testIndices.size()][];
            int[] yTest = new int[testIndices.size()];
            
            for (int i = 0; i < trainIndices.size(); i++) {
                XTrain[i] = X[trainIndices.get(i)];
                yTrain[i] = y[trainIndices.get(i)];
            }
            
            for (int i = 0; i < testIndices.size(); i++) {
                XTest[i] = X[testIndices.get(i)];
                yTest[i] = y[testIndices.get(i)];
            }
            
            // Train and evaluate
            M model = trainer.train(XTrain, yTrain);
            int[] predictions = predictor.predict(model, XTest);
            double accuracy = Metrics.accuracy(yTest, predictions);
            
            scores.add(accuracy);
        }
        
        return new CrossValidationResult(scores, "accuracy");
    }
    
    /**
     * Performs Leave-One-Out Cross-Validation (LOOCV).
     * 
     * Uses n-1 samples for training and 1 sample for testing, repeating n times.
     * Computationally expensive for large datasets.
     * 
     * @param <M> model type
     * @param trainer function to train the model
     * @param predictor function to make predictions
     * @param X feature matrix
     * @param y target labels
     * @return cross-validation results
     */
    public static <M> CrossValidationResult leaveOneOut(
            ModelTrainer<M> trainer,
            ModelPredictor<M> predictor,
            double[][] X,
            int[] y) {
        
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        int n = X.length;
        List<Double> scores = new ArrayList<>();
        
        for (int i = 0; i < n; i++) {
            // Create train and test sets (leaving out sample i)
            double[][] XTrain = new double[n - 1][];
            int[] yTrain = new int[n - 1];
            double[][] XTest = new double[1][];
            int[] yTest = new int[1];
            
            int trainIdx = 0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    XTrain[trainIdx] = X[j];
                    yTrain[trainIdx] = y[j];
                    trainIdx++;
                }
            }
            
            XTest[0] = X[i];
            yTest[0] = y[i];
            
            // Train and evaluate
            M model = trainer.train(XTrain, yTrain);
            int[] predictions = predictor.predict(model, XTest);
            double accuracy = predictions[0] == yTest[0] ? 1.0 : 0.0;
            
            scores.add(accuracy);
        }
        
        return new CrossValidationResult(scores, "accuracy");
    }
    
    /**
     * Performs Shuffle Split cross-validation.
     * 
     * Randomly splits data into train and test sets multiple times.
     * Unlike K-Fold, samples may appear in multiple test sets or none.
     * 
     * @param <M> model type
     * @param trainer function to train the model
     * @param predictor function to make predictions
     * @param X feature matrix
     * @param y target labels
     * @param nSplits number of random splits
     * @param testSize fraction of data to use for testing (0.0 to 1.0)
     * @param randomState random seed for reproducibility
     * @return cross-validation results
     */
    public static <M> CrossValidationResult shuffleSplit(
            ModelTrainer<M> trainer,
            ModelPredictor<M> predictor,
            double[][] X,
            int[] y,
            int nSplits,
            double testSize,
            Integer randomState) {
        
        if (testSize <= 0.0 || testSize >= 1.0) {
            throw new IllegalArgumentException("Test size must be between 0.0 and 1.0");
        }
        if (nSplits <= 0) {
            throw new IllegalArgumentException("Number of splits must be greater than 0");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        int n = X.length;
        int testSamples = (int) (n * testSize);
        
        if (testSamples == 0 || testSamples == n) {
            throw new IllegalArgumentException("Invalid test size: results in empty train or test set");
        }
        
        Random random = randomState != null ? new Random(randomState) : new Random();
        List<Double> scores = new ArrayList<>();
        
        for (int split = 0; split < nSplits; split++) {
            // Create random permutation
            int[] indices = createIndices(n, random.nextInt());
            
            // Split into train and test
            SplitData splitData = splitByIndices(X, y, indices, 0, testSamples);
            
            // Train and evaluate
            M model = trainer.train(splitData.XTrain, splitData.yTrain);
            int[] predictions = predictor.predict(model, splitData.XTest);
            double accuracy = Metrics.accuracy(splitData.yTest, predictions);
            
            scores.add(accuracy);
        }
        
        return new CrossValidationResult(scores, "accuracy");
    }
    
    /**
     * Performs a simple train-test split.
     * 
     * @param X feature matrix
     * @param y target labels
     * @param testSize fraction of data to use for testing (0.0 to 1.0)
     * @param randomState random seed for reproducibility (use null for no shuffling)
     * @return split data
     */
    public static SplitData trainTestSplit(
            double[][] X,
            int[] y,
            double testSize,
            Integer randomState) {
        
        if (testSize <= 0.0 || testSize >= 1.0) {
            throw new IllegalArgumentException("Test size must be between 0.0 and 1.0");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        int n = X.length;
        int testSamples = (int) (n * testSize);
        
        if (testSamples == 0 || testSamples == n) {
            throw new IllegalArgumentException("Invalid test size: results in empty train or test set");
        }
        
        int[] indices = createIndices(n, randomState);
        return splitByIndices(X, y, indices, 0, testSamples);
    }
    
    /**
     * Creates an array of indices, optionally shuffled.
     * 
     * @param n number of indices
     * @param randomState random seed (null for no shuffling)
     * @return array of indices
     */
    private static int[] createIndices(int n, Integer randomState) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        
        if (randomState != null) {
            Random random = new Random(randomState);
            for (int i = n - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        
        return indices;
    }
    
    /**
     * Splits data by indices into train and test sets.
     * 
     * @param X feature matrix
     * @param y target labels
     * @param indices permuted indices
     * @param testStart start index for test set
     * @param testEnd end index for test set (exclusive)
     * @return split data
     */
    private static SplitData splitByIndices(
            double[][] X,
            int[] y,
            int[] indices,
            int testStart,
            int testEnd) {
        
        int n = X.length;
        int testSize = testEnd - testStart;
        int trainSize = n - testSize;
        
        double[][] XTrain = new double[trainSize][];
        int[] yTrain = new int[trainSize];
        double[][] XTest = new double[testSize][];
        int[] yTest = new int[testSize];
        
        int trainIdx = 0;
        int testIdx = 0;
        
        for (int i = 0; i < n; i++) {
            if (i >= testStart && i < testEnd) {
                XTest[testIdx] = X[indices[i]];
                yTest[testIdx] = y[indices[i]];
                testIdx++;
            } else {
                XTrain[trainIdx] = X[indices[i]];
                yTrain[trainIdx] = y[indices[i]];
                trainIdx++;
            }
        }
        
        return new SplitData(XTrain, yTrain, XTest, yTest);
    }
    
    /**
     * Container for split data.
     */
    public static class SplitData {
        public final double[][] XTrain;
        public final int[] yTrain;
        public final double[][] XTest;
        public final int[] yTest;
        
        public SplitData(double[][] XTrain, int[] yTrain, double[][] XTest, int[] yTest) {
            this.XTrain = XTrain;
            this.yTrain = yTrain;
            this.XTest = XTest;
            this.yTest = yTest;
        }
    }
}
