package com.mindforge.validation;

/**
 * Common evaluation metrics for machine learning models.
 */
public class Metrics {
    
    /**
     * Calculates the accuracy of classification predictions.
     * 
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return accuracy (proportion of correct predictions)
     */
    public static double accuracy(int[] yTrue, int[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) {
                correct++;
            }
        }
        return (double) correct / yTrue.length;
    }
    
    /**
     * Calculates the Mean Squared Error (MSE) for regression.
     * 
     * @param yTrue true values
     * @param yPred predicted values
     * @return mean squared error
     */
    public static double mse(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double diff = yTrue[i] - yPred[i];
            sum += diff * diff;
        }
        return sum / yTrue.length;
    }
    
    /**
     * Calculates the Root Mean Squared Error (RMSE) for regression.
     * 
     * @param yTrue true values
     * @param yPred predicted values
     * @return root mean squared error
     */
    public static double rmse(double[] yTrue, double[] yPred) {
        return Math.sqrt(mse(yTrue, yPred));
    }
    
    /**
     * Calculates the Mean Absolute Error (MAE) for regression.
     * 
     * @param yTrue true values
     * @param yPred predicted values
     * @return mean absolute error
     */
    public static double mae(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            sum += Math.abs(yTrue[i] - yPred[i]);
        }
        return sum / yTrue.length;
    }
    
    /**
     * Calculates the R-squared (coefficient of determination) for regression.
     * 
     * @param yTrue true values
     * @param yPred predicted values
     * @return R-squared value
     */
    public static double r2Score(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        // Calculate mean of true values
        double mean = 0.0;
        for (double y : yTrue) {
            mean += y;
        }
        mean /= yTrue.length;
        
        // Calculate total sum of squares and residual sum of squares
        double ssTot = 0.0;
        double ssRes = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            ssTot += Math.pow(yTrue[i] - mean, 2);
            ssRes += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        if (ssTot == 0.0) {
            return 1.0;
        }
        
        return 1.0 - (ssRes / ssTot);
    }
    
    /**
     * Calculates precision for binary classification.
     * 
     * @param yTrue true labels
     * @param yPred predicted labels
     * @param positiveClass the class considered as positive
     * @return precision
     */
    public static double precision(int[] yTrue, int[] yPred, int positiveClass) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int truePositives = 0;
        int falsePositives = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yPred[i] == positiveClass) {
                if (yTrue[i] == positiveClass) {
                    truePositives++;
                } else {
                    falsePositives++;
                }
            }
        }
        
        int totalPredictedPositive = truePositives + falsePositives;
        return totalPredictedPositive == 0 ? 0.0 : (double) truePositives / totalPredictedPositive;
    }
    
    /**
     * Calculates recall for binary classification.
     * 
     * @param yTrue true labels
     * @param yPred predicted labels
     * @param positiveClass the class considered as positive
     * @return recall
     */
    public static double recall(int[] yTrue, int[] yPred, int positiveClass) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int truePositives = 0;
        int falseNegatives = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == positiveClass) {
                if (yPred[i] == positiveClass) {
                    truePositives++;
                } else {
                    falseNegatives++;
                }
            }
        }
        
        int totalActualPositive = truePositives + falseNegatives;
        return totalActualPositive == 0 ? 0.0 : (double) truePositives / totalActualPositive;
    }
    
    /**
     * Calculates F1 score for binary classification.
     * 
     * @param yTrue true labels
     * @param yPred predicted labels
     * @param positiveClass the class considered as positive
     * @return F1 score
     */
    public static double f1Score(int[] yTrue, int[] yPred, int positiveClass) {
        double prec = precision(yTrue, yPred, positiveClass);
        double rec = recall(yTrue, yPred, positiveClass);
        
        if (prec + rec == 0.0) {
            return 0.0;
        }
        
        return 2.0 * (prec * rec) / (prec + rec);
    }
}
