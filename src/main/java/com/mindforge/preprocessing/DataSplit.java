package com.mindforge.preprocessing;

import java.util.*;

/**
 * DataSplit provides utilities for splitting datasets into training and testing sets.
 * 
 * Supports:
 * - Random split
 * - Stratified split (for classification tasks)
 * - Shuffle option
 * 
 * Example:
 * <pre>
 * double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}};
 * int[] y = {0, 0, 1, 1};
 * TrainTestSplit split = DataSplit.trainTestSplit(X, y, 0.25, true, 42);
 * </pre>
 */
public class DataSplit {

    /**
     * Container for train/test split results.
     */
    public static class TrainTestSplit {
        public final double[][] XTrain;
        public final double[][] XTest;
        public final int[] yTrain;
        public final int[] yTest;

        public TrainTestSplit(double[][] XTrain, double[][] XTest, int[] yTrain, int[] yTest) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }

    /**
     * Container for train/test split results (regression version with double labels).
     */
    public static class TrainTestSplitRegression {
        public final double[][] XTrain;
        public final double[][] XTest;
        public final double[] yTrain;
        public final double[] yTest;

        public TrainTestSplitRegression(double[][] XTrain, double[][] XTest, double[] yTrain, double[] yTest) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }

    /**
     * Splits arrays into random train and test subsets.
     * 
     * @param X feature matrix
     * @param y target values (int for classification)
     * @param testSize proportion of the dataset to include in the test split (0.0 to 1.0)
     * @param shuffle whether to shuffle the data before splitting
     * @param randomSeed random seed for reproducibility (use null for random)
     * @return TrainTestSplit object containing the split data
     */
    public static TrainTestSplit trainTestSplit(double[][] X, int[] y, double testSize, boolean shuffle, Integer randomSeed) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("X and y cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        if (testSize <= 0 || testSize >= 1) {
            throw new IllegalArgumentException("testSize must be between 0 and 1");
        }

        int nSamples = X.length;
        int testSamples = (int) Math.round(nSamples * testSize);
        int trainSamples = nSamples - testSamples;

        // Create indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            indices.add(i);
        }

        // Shuffle if requested
        if (shuffle) {
            Random random = randomSeed != null ? new Random(randomSeed) : new Random();
            Collections.shuffle(indices, random);
        }

        // Split the data
        double[][] XTrain = new double[trainSamples][];
        double[][] XTest = new double[testSamples][];
        int[] yTrain = new int[trainSamples];
        int[] yTest = new int[testSamples];

        for (int i = 0; i < trainSamples; i++) {
            int idx = indices.get(i);
            XTrain[i] = X[idx].clone();
            yTrain[i] = y[idx];
        }

        for (int i = 0; i < testSamples; i++) {
            int idx = indices.get(trainSamples + i);
            XTest[i] = X[idx].clone();
            yTest[i] = y[idx];
        }

        return new TrainTestSplit(XTrain, XTest, yTrain, yTest);
    }

    /**
     * Splits arrays into random train and test subsets (regression version).
     * 
     * @param X feature matrix
     * @param y target values (double for regression)
     * @param testSize proportion of the dataset to include in the test split (0.0 to 1.0)
     * @param shuffle whether to shuffle the data before splitting
     * @param randomSeed random seed for reproducibility (use null for random)
     * @return TrainTestSplitRegression object containing the split data
     */
    public static TrainTestSplitRegression trainTestSplit(double[][] X, double[] y, double testSize, boolean shuffle, Integer randomSeed) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("X and y cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        if (testSize <= 0 || testSize >= 1) {
            throw new IllegalArgumentException("testSize must be between 0 and 1");
        }

        int nSamples = X.length;
        int testSamples = (int) Math.round(nSamples * testSize);
        int trainSamples = nSamples - testSamples;

        // Create indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            indices.add(i);
        }

        // Shuffle if requested
        if (shuffle) {
            Random random = randomSeed != null ? new Random(randomSeed) : new Random();
            Collections.shuffle(indices, random);
        }

        // Split the data
        double[][] XTrain = new double[trainSamples][];
        double[][] XTest = new double[testSamples][];
        double[] yTrain = new double[trainSamples];
        double[] yTest = new double[testSamples];

        for (int i = 0; i < trainSamples; i++) {
            int idx = indices.get(i);
            XTrain[i] = X[idx].clone();
            yTrain[i] = y[idx];
        }

        for (int i = 0; i < testSamples; i++) {
            int idx = indices.get(trainSamples + i);
            XTest[i] = X[idx].clone();
            yTest[i] = y[idx];
        }

        return new TrainTestSplitRegression(XTrain, XTest, yTrain, yTest);
    }

    /**
     * Splits arrays into stratified train and test subsets.
     * Ensures that the proportion of samples for each class is preserved.
     * 
     * @param X feature matrix
     * @param y target values
     * @param testSize proportion of the dataset to include in the test split (0.0 to 1.0)
     * @param randomSeed random seed for reproducibility (use null for random)
     * @return TrainTestSplit object containing the stratified split data
     */
    public static TrainTestSplit stratifiedTrainTestSplit(double[][] X, int[] y, double testSize, Integer randomSeed) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("X and y cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        if (testSize <= 0 || testSize >= 1) {
            throw new IllegalArgumentException("testSize must be between 0 and 1");
        }

        // Group indices by class
        Map<Integer, List<Integer>> classIndices = new HashMap<>();
        for (int i = 0; i < y.length; i++) {
            classIndices.computeIfAbsent(y[i], k -> new ArrayList<>()).add(i);
        }

        Random random = randomSeed != null ? new Random(randomSeed) : new Random();
        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> testIndices = new ArrayList<>();

        // Stratified split for each class
        for (List<Integer> indices : classIndices.values()) {
            Collections.shuffle(indices, random);
            int classTestSize = (int) Math.round(indices.size() * testSize);
            
            testIndices.addAll(indices.subList(0, classTestSize));
            trainIndices.addAll(indices.subList(classTestSize, indices.size()));
        }

        // Shuffle the final splits
        Collections.shuffle(trainIndices, random);
        Collections.shuffle(testIndices, random);

        // Build the split arrays
        double[][] XTrain = new double[trainIndices.size()][];
        double[][] XTest = new double[testIndices.size()][];
        int[] yTrain = new int[trainIndices.size()];
        int[] yTest = new int[testIndices.size()];

        for (int i = 0; i < trainIndices.size(); i++) {
            int idx = trainIndices.get(i);
            XTrain[i] = X[idx].clone();
            yTrain[i] = y[idx];
        }

        for (int i = 0; i < testIndices.size(); i++) {
            int idx = testIndices.get(i);
            XTest[i] = X[idx].clone();
            yTest[i] = y[idx];
        }

        return new TrainTestSplit(XTrain, XTest, yTrain, yTest);
    }
}
