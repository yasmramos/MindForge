package com.mindforge.validation;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Confusion matrix for evaluating classification models.
 */
public class ConfusionMatrix {
    
    private int[][] matrix;
    private int numClasses;
    private String[] classLabels;
    
    /**
     * Create a confusion matrix from predictions and true labels.
     * 
     * @param yTrue true labels
     * @param yPred predicted labels
     */
    public ConfusionMatrix(int[] yTrue, int[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        // Find number of classes
        numClasses = 0;
        for (int i = 0; i < yTrue.length; i++) {
            numClasses = Math.max(numClasses, Math.max(yTrue[i], yPred[i]) + 1);
        }
        
        // Build confusion matrix
        matrix = new int[numClasses][numClasses];
        for (int i = 0; i < yTrue.length; i++) {
            matrix[yTrue[i]][yPred[i]]++;
        }
        
        // Default class labels
        classLabels = new String[numClasses];
        for (int i = 0; i < numClasses; i++) {
            classLabels[i] = String.valueOf(i);
        }
    }
    
    /**
     * Create a confusion matrix with custom class labels.
     * 
     * @param yTrue true labels
     * @param yPred predicted labels
     * @param labels class labels
     */
    public ConfusionMatrix(int[] yTrue, int[] yPred, String[] labels) {
        this(yTrue, yPred);
        if (labels.length == numClasses) {
            this.classLabels = labels;
        }
    }
    
    /**
     * Get the raw confusion matrix.
     * 
     * @return confusion matrix as 2D array
     */
    public int[][] getMatrix() {
        return matrix;
    }
    
    /**
     * Get true positives for a class.
     * 
     * @param classIndex class index
     * @return true positives count
     */
    public int getTruePositives(int classIndex) {
        return matrix[classIndex][classIndex];
    }
    
    /**
     * Get false positives for a class.
     * 
     * @param classIndex class index
     * @return false positives count
     */
    public int getFalsePositives(int classIndex) {
        int fp = 0;
        for (int i = 0; i < numClasses; i++) {
            if (i != classIndex) {
                fp += matrix[i][classIndex];
            }
        }
        return fp;
    }
    
    /**
     * Get false negatives for a class.
     * 
     * @param classIndex class index
     * @return false negatives count
     */
    public int getFalseNegatives(int classIndex) {
        int fn = 0;
        for (int i = 0; i < numClasses; i++) {
            if (i != classIndex) {
                fn += matrix[classIndex][i];
            }
        }
        return fn;
    }
    
    /**
     * Get true negatives for a class.
     * 
     * @param classIndex class index
     * @return true negatives count
     */
    public int getTrueNegatives(int classIndex) {
        int tn = 0;
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (i != classIndex && j != classIndex) {
                    tn += matrix[i][j];
                }
            }
        }
        return tn;
    }
    
    /**
     * Calculate precision for a class.
     * 
     * @param classIndex class index
     * @return precision
     */
    public double getPrecision(int classIndex) {
        int tp = getTruePositives(classIndex);
        int fp = getFalsePositives(classIndex);
        return tp + fp == 0 ? 0.0 : (double) tp / (tp + fp);
    }
    
    /**
     * Calculate recall for a class.
     * 
     * @param classIndex class index
     * @return recall
     */
    public double getRecall(int classIndex) {
        int tp = getTruePositives(classIndex);
        int fn = getFalseNegatives(classIndex);
        return tp + fn == 0 ? 0.0 : (double) tp / (tp + fn);
    }
    
    /**
     * Calculate F1 score for a class.
     * 
     * @param classIndex class index
     * @return F1 score
     */
    public double getF1Score(int classIndex) {
        double precision = getPrecision(classIndex);
        double recall = getRecall(classIndex);
        return precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall);
    }
    
    /**
     * Calculate overall accuracy.
     * 
     * @return accuracy
     */
    public double getAccuracy() {
        int correct = 0;
        int total = 0;
        for (int i = 0; i < numClasses; i++) {
            correct += matrix[i][i];
            for (int j = 0; j < numClasses; j++) {
                total += matrix[i][j];
            }
        }
        return total == 0 ? 0.0 : (double) correct / total;
    }
    
    /**
     * Calculate macro-averaged precision.
     * 
     * @return macro precision
     */
    public double getMacroPrecision() {
        double sum = 0.0;
        for (int i = 0; i < numClasses; i++) {
            sum += getPrecision(i);
        }
        return sum / numClasses;
    }
    
    /**
     * Calculate macro-averaged recall.
     * 
     * @return macro recall
     */
    public double getMacroRecall() {
        double sum = 0.0;
        for (int i = 0; i < numClasses; i++) {
            sum += getRecall(i);
        }
        return sum / numClasses;
    }
    
    /**
     * Calculate macro-averaged F1 score.
     * 
     * @return macro F1 score
     */
    public double getMacroF1Score() {
        double sum = 0.0;
        for (int i = 0; i < numClasses; i++) {
            sum += getF1Score(i);
        }
        return sum / numClasses;
    }
    
    /**
     * Calculate weighted F1 score.
     * 
     * @return weighted F1 score
     */
    public double getWeightedF1Score() {
        double sum = 0.0;
        int total = 0;
        
        for (int i = 0; i < numClasses; i++) {
            int support = 0;
            for (int j = 0; j < numClasses; j++) {
                support += matrix[i][j];
            }
            sum += getF1Score(i) * support;
            total += support;
        }
        
        return total == 0 ? 0.0 : sum / total;
    }
    
    /**
     * Get the number of classes.
     * 
     * @return number of classes
     */
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * Get class labels.
     * 
     * @return class labels
     */
    public String[] getClassLabels() {
        return classLabels;
    }
    
    /**
     * Set class labels.
     * 
     * @param labels class labels
     */
    public void setClassLabels(String[] labels) {
        if (labels.length == numClasses) {
            this.classLabels = labels;
        }
    }
    
    /**
     * Get a formatted string representation of the confusion matrix.
     * 
     * @return formatted confusion matrix
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Confusion Matrix:\n");
        
        // Header
        sb.append(String.format("%10s", ""));
        for (String label : classLabels) {
            sb.append(String.format("%10s", "Pred:" + label));
        }
        sb.append("\n");
        
        // Rows
        for (int i = 0; i < numClasses; i++) {
            sb.append(String.format("%10s", "True:" + classLabels[i]));
            for (int j = 0; j < numClasses; j++) {
                sb.append(String.format("%10d", matrix[i][j]));
            }
            sb.append("\n");
        }
        
        // Metrics
        sb.append("\nMetrics:\n");
        sb.append(String.format("Accuracy: %.4f\n", getAccuracy()));
        sb.append(String.format("Macro Precision: %.4f\n", getMacroPrecision()));
        sb.append(String.format("Macro Recall: %.4f\n", getMacroRecall()));
        sb.append(String.format("Macro F1-Score: %.4f\n", getMacroF1Score()));
        sb.append(String.format("Weighted F1-Score: %.4f\n", getWeightedF1Score()));
        
        return sb.toString();
    }
    
    /**
     * Generate a classification report.
     * 
     * @return classification report string
     */
    public String classificationReport() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-15s %10s %10s %10s %10s\n", 
                "", "precision", "recall", "f1-score", "support"));
        sb.append("\n");
        
        int totalSupport = 0;
        for (int i = 0; i < numClasses; i++) {
            int support = 0;
            for (int j = 0; j < numClasses; j++) {
                support += matrix[i][j];
            }
            totalSupport += support;
            
            sb.append(String.format("%-15s %10.4f %10.4f %10.4f %10d\n",
                    classLabels[i],
                    getPrecision(i),
                    getRecall(i),
                    getF1Score(i),
                    support));
        }
        
        sb.append("\n");
        sb.append(String.format("%-15s %10.4f %10.4f %10.4f %10d\n",
                "accuracy", getAccuracy(), getAccuracy(), getAccuracy(), totalSupport));
        sb.append(String.format("%-15s %10.4f %10.4f %10.4f %10d\n",
                "macro avg", getMacroPrecision(), getMacroRecall(), getMacroF1Score(), totalSupport));
        sb.append(String.format("%-15s %10.4f %10.4f %10.4f %10d\n",
                "weighted avg", getMacroPrecision(), getMacroRecall(), getWeightedF1Score(), totalSupport));
        
        return sb.toString();
    }
}
