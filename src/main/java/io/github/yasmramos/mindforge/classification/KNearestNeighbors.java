package io.github.yasmramos.mindforge.classification;

import io.github.yasmramos.mindforge.math.Distance;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * K-Nearest Neighbors classifier.
 * This is a simple, yet powerful classification algorithm that classifies
 * a data point based on the majority class of its k nearest neighbors.
 */
public class KNearestNeighbors implements Classifier<double[]> {
    
    private final int k;
    private double[][] trainingData;
    private int[] trainingLabels;
    private int numClasses;
    
    /**
     * Creates a new K-Nearest Neighbors classifier.
     * 
     * @param k number of neighbors to consider
     */
    public KNearestNeighbors(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        this.k = k;
    }
    
    @Override
    public void train(double[][] x, int[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("Data and labels must have the same length");
        }
        if (x.length < k) {
            throw new IllegalArgumentException("Training data size must be at least k");
        }
        
        this.trainingData = Arrays.copyOf(x, x.length);
        this.trainingLabels = Arrays.copyOf(y, y.length);
        
        // Determine number of classes
        int maxLabel = 0;
        for (int label : y) {
            if (label > maxLabel) {
                maxLabel = label;
            }
        }
        this.numClasses = maxLabel + 1;
    }
    
    @Override
    public int predict(double[] x) {
        if (trainingData == null) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        // Calculate distances to all training points
        double[] distances = new double[trainingData.length];
        for (int i = 0; i < trainingData.length; i++) {
            distances[i] = Distance.euclidean(x, trainingData[i]);
        }
        
        // Find k nearest neighbors
        int[] nearestIndices = findKNearest(distances);
        
        // Vote for the most common class among k nearest neighbors
        Map<Integer, Integer> votes = new HashMap<>();
        for (int idx : nearestIndices) {
            int label = trainingLabels[idx];
            votes.put(label, votes.getOrDefault(label, 0) + 1);
        }
        
        // Return class with most votes
        int maxVotes = 0;
        int predictedClass = -1;
        for (Map.Entry<Integer, Integer> entry : votes.entrySet()) {
            if (entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                predictedClass = entry.getKey();
            }
        }
        
        return predictedClass;
    }
    
    @Override
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * Finds the indices of k nearest neighbors.
     * 
     * @param distances array of distances
     * @return indices of k smallest distances
     */
    private int[] findKNearest(double[] distances) {
        int[] indices = new int[distances.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        
        // Partial sort to find k smallest
        for (int i = 0; i < k; i++) {
            int minIdx = i;
            for (int j = i + 1; j < distances.length; j++) {
                if (distances[indices[j]] < distances[indices[minIdx]]) {
                    minIdx = j;
                }
            }
            if (minIdx != i) {
                int temp = indices[i];
                indices[i] = indices[minIdx];
                indices[minIdx] = temp;
            }
        }
        
        return Arrays.copyOf(indices, k);
    }
    
    /**
     * Predicts class labels for multiple samples.
     * 
     * @param X array of feature vectors
     * @return array of predicted labels
     */
    public int[] predict(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    /**
     * Returns the value of k.
     * 
     * @return number of neighbors
     */
    public int getK() {
        return k;
    }
}
