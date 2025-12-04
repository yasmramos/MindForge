package com.mindforge.data;

import java.util.Arrays;
import java.util.Random;

/**
 * Represents a dataset with features and labels/targets.
 */
public class Dataset {
    
    private double[][] features;
    private double[] targets;
    private int[] labels;
    private String[] featureNames;
    private String[] targetNames;
    private String name;
    private String description;
    private boolean isClassification;
    
    /**
     * Create a classification dataset.
     * 
     * @param features feature matrix
     * @param labels class labels
     */
    public Dataset(double[][] features, int[] labels) {
        this.features = features;
        this.labels = labels;
        this.isClassification = true;
    }
    
    /**
     * Create a regression dataset.
     * 
     * @param features feature matrix
     * @param targets target values
     */
    public Dataset(double[][] features, double[] targets) {
        this.features = features;
        this.targets = targets;
        this.isClassification = false;
    }
    
    /**
     * Get the feature matrix.
     * 
     * @return features
     */
    public double[][] getFeatures() {
        return features;
    }
    
    /**
     * Get the class labels (for classification).
     * 
     * @return labels
     */
    public int[] getLabels() {
        return labels;
    }
    
    /**
     * Get the targets (for regression).
     * 
     * @return targets
     */
    public double[] getTargets() {
        return targets;
    }
    
    /**
     * Get the number of samples.
     * 
     * @return number of samples
     */
    public int getNumSamples() {
        return features.length;
    }
    
    /**
     * Get the number of features.
     * 
     * @return number of features
     */
    public int getNumFeatures() {
        return features.length > 0 ? features[0].length : 0;
    }
    
    /**
     * Get the number of classes (for classification).
     * 
     * @return number of classes
     */
    public int getNumClasses() {
        if (!isClassification || labels == null) return 0;
        int max = 0;
        for (int label : labels) {
            if (label > max) max = label;
        }
        return max + 1;
    }
    
    /**
     * Check if this is a classification dataset.
     * 
     * @return true if classification
     */
    public boolean isClassification() {
        return isClassification;
    }
    
    /**
     * Set feature names.
     * 
     * @param names feature names
     */
    public void setFeatureNames(String[] names) {
        this.featureNames = names;
    }
    
    /**
     * Get feature names.
     * 
     * @return feature names
     */
    public String[] getFeatureNames() {
        return featureNames;
    }
    
    /**
     * Set target/class names.
     * 
     * @param names target names
     */
    public void setTargetNames(String[] names) {
        this.targetNames = names;
    }
    
    /**
     * Get target/class names.
     * 
     * @return target names
     */
    public String[] getTargetNames() {
        return targetNames;
    }
    
    /**
     * Set dataset name.
     * 
     * @param name dataset name
     */
    public void setName(String name) {
        this.name = name;
    }
    
    /**
     * Get dataset name.
     * 
     * @return dataset name
     */
    public String getName() {
        return name;
    }
    
    /**
     * Set dataset description.
     * 
     * @param description description
     */
    public void setDescription(String description) {
        this.description = description;
    }
    
    /**
     * Get dataset description.
     * 
     * @return description
     */
    public String getDescription() {
        return description;
    }
    
    /**
     * Split the dataset into training and test sets.
     * 
     * @param testSize fraction of data to use for testing (0-1)
     * @param seed random seed
     * @return array of [trainSet, testSet]
     */
    public Dataset[] trainTestSplit(double testSize, long seed) {
        int n = features.length;
        int testCount = (int) (n * testSize);
        int trainCount = n - testCount;
        
        // Create shuffled indices
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        
        Random random = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Create train and test sets
        double[][] trainFeatures = new double[trainCount][];
        double[][] testFeatures = new double[testCount][];
        
        if (isClassification) {
            int[] trainLabels = new int[trainCount];
            int[] testLabels = new int[testCount];
            
            for (int i = 0; i < trainCount; i++) {
                trainFeatures[i] = features[indices[i]];
                trainLabels[i] = labels[indices[i]];
            }
            for (int i = 0; i < testCount; i++) {
                testFeatures[i] = features[indices[trainCount + i]];
                testLabels[i] = labels[indices[trainCount + i]];
            }
            
            Dataset trainSet = new Dataset(trainFeatures, trainLabels);
            Dataset testSet = new Dataset(testFeatures, testLabels);
            trainSet.setFeatureNames(featureNames);
            trainSet.setTargetNames(targetNames);
            testSet.setFeatureNames(featureNames);
            testSet.setTargetNames(targetNames);
            
            return new Dataset[]{trainSet, testSet};
        } else {
            double[] trainTargets = new double[trainCount];
            double[] testTargets = new double[testCount];
            
            for (int i = 0; i < trainCount; i++) {
                trainFeatures[i] = features[indices[i]];
                trainTargets[i] = targets[indices[i]];
            }
            for (int i = 0; i < testCount; i++) {
                testFeatures[i] = features[indices[trainCount + i]];
                testTargets[i] = targets[indices[trainCount + i]];
            }
            
            Dataset trainSet = new Dataset(trainFeatures, trainTargets);
            Dataset testSet = new Dataset(testFeatures, testTargets);
            trainSet.setFeatureNames(featureNames);
            testSet.setFeatureNames(featureNames);
            
            return new Dataset[]{trainSet, testSet};
        }
    }
    
    /**
     * Get a subset of the dataset.
     * 
     * @param start start index
     * @param end end index (exclusive)
     * @return subset dataset
     */
    public Dataset subset(int start, int end) {
        int size = end - start;
        double[][] subFeatures = new double[size][];
        
        for (int i = 0; i < size; i++) {
            subFeatures[i] = features[start + i];
        }
        
        if (isClassification) {
            int[] subLabels = Arrays.copyOfRange(labels, start, end);
            Dataset subset = new Dataset(subFeatures, subLabels);
            subset.setFeatureNames(featureNames);
            subset.setTargetNames(targetNames);
            return subset;
        } else {
            double[] subTargets = Arrays.copyOfRange(targets, start, end);
            Dataset subset = new Dataset(subFeatures, subTargets);
            subset.setFeatureNames(featureNames);
            return subset;
        }
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(name != null ? name : "Dataset");
        sb.append(" (");
        sb.append(getNumSamples()).append(" samples, ");
        sb.append(getNumFeatures()).append(" features");
        if (isClassification) {
            sb.append(", ").append(getNumClasses()).append(" classes");
        }
        sb.append(")");
        return sb.toString();
    }
}
