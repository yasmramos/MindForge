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
    
    /**
     * Calculates log loss (cross-entropy loss) for binary classification.
     * 
     * @param yTrue true labels (0 or 1)
     * @param yProba predicted probabilities for positive class
     * @return log loss
     */
    public static double logLoss(int[] yTrue, double[] yProba) {
        if (yTrue.length != yProba.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        double eps = 1e-15;
        
        for (int i = 0; i < yTrue.length; i++) {
            double p = Math.max(eps, Math.min(1 - eps, yProba[i]));
            if (yTrue[i] == 1) {
                sum -= Math.log(p);
            } else {
                sum -= Math.log(1 - p);
            }
        }
        
        return sum / yTrue.length;
    }
    
    /**
     * Calculates multi-class log loss.
     * 
     * @param yTrue true labels (class indices)
     * @param yProba predicted probabilities for each class [samples][classes]
     * @return log loss
     */
    public static double logLossMulticlass(int[] yTrue, double[][] yProba) {
        if (yTrue.length != yProba.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        double eps = 1e-15;
        
        for (int i = 0; i < yTrue.length; i++) {
            double p = Math.max(eps, Math.min(1 - eps, yProba[i][yTrue[i]]));
            sum -= Math.log(p);
        }
        
        return sum / yTrue.length;
    }
    
    /**
     * Calculates Cohen's Kappa score.
     * Measures inter-rater agreement for categorical items.
     * 
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return kappa score
     */
    public static double cohenKappa(int[] yTrue, int[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int n = yTrue.length;
        
        // Find number of classes
        int numClasses = 0;
        for (int i = 0; i < n; i++) {
            numClasses = Math.max(numClasses, Math.max(yTrue[i], yPred[i]) + 1);
        }
        
        // Build confusion matrix
        int[][] cm = new int[numClasses][numClasses];
        for (int i = 0; i < n; i++) {
            cm[yTrue[i]][yPred[i]]++;
        }
        
        // Calculate observed agreement
        double po = 0.0;
        for (int i = 0; i < numClasses; i++) {
            po += cm[i][i];
        }
        po /= n;
        
        // Calculate expected agreement
        double pe = 0.0;
        for (int k = 0; k < numClasses; k++) {
            double rowSum = 0, colSum = 0;
            for (int i = 0; i < numClasses; i++) {
                rowSum += cm[k][i];
                colSum += cm[i][k];
            }
            pe += (rowSum * colSum) / (n * n);
        }
        
        if (pe == 1.0) {
            return 1.0;
        }
        
        return (po - pe) / (1 - pe);
    }
    
    /**
     * Calculates Matthews Correlation Coefficient (MCC) for binary classification.
     * 
     * @param yTrue true labels (0 or 1)
     * @param yPred predicted labels (0 or 1)
     * @return MCC value between -1 and 1
     */
    public static double matthewsCorrelation(int[] yTrue, int[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int tp = 0, tn = 0, fp = 0, fn = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1 && yPred[i] == 1) tp++;
            else if (yTrue[i] == 0 && yPred[i] == 0) tn++;
            else if (yTrue[i] == 0 && yPred[i] == 1) fp++;
            else if (yTrue[i] == 1 && yPred[i] == 0) fn++;
        }
        
        double numerator = (double) tp * tn - (double) fp * fn;
        double denominator = Math.sqrt((double) (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
        
        if (denominator == 0) {
            return 0.0;
        }
        
        return numerator / denominator;
    }
    
    /**
     * Calculates adjusted R-squared for regression.
     * 
     * @param yTrue true values
     * @param yPred predicted values
     * @param numFeatures number of features in the model
     * @return adjusted R-squared
     */
    public static double adjustedR2(double[] yTrue, double[] yPred, int numFeatures) {
        double r2 = r2Score(yTrue, yPred);
        int n = yTrue.length;
        
        if (n <= numFeatures + 1) {
            return r2;
        }
        
        return 1 - (1 - r2) * (n - 1) / (n - numFeatures - 1);
    }
    
    /**
     * Calculates Mean Absolute Percentage Error (MAPE).
     * 
     * @param yTrue true values
     * @param yPred predicted values
     * @return MAPE as a percentage
     */
    public static double mape(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] != 0) {
                sum += Math.abs((yTrue[i] - yPred[i]) / yTrue[i]);
                count++;
            }
        }
        
        return count == 0 ? 0.0 : (sum / count) * 100;
    }
    
    /**
     * Calculates Symmetric Mean Absolute Percentage Error (SMAPE).
     * 
     * @param yTrue true values
     * @param yPred predicted values
     * @return SMAPE as a percentage
     */
    public static double smape(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            double denom = Math.abs(yTrue[i]) + Math.abs(yPred[i]);
            if (denom != 0) {
                sum += Math.abs(yTrue[i] - yPred[i]) / denom;
                count++;
            }
        }
        
        return count == 0 ? 0.0 : (sum / count) * 200;
    }
    
    /**
     * Calculates specificity (true negative rate) for binary classification.
     * 
     * @param yTrue true labels
     * @param yPred predicted labels
     * @param positiveClass the class considered as positive
     * @return specificity
     */
    public static double specificity(int[] yTrue, int[] yPred, int positiveClass) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int trueNegatives = 0;
        int falsePositives = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] != positiveClass) {
                if (yPred[i] != positiveClass) {
                    trueNegatives++;
                } else {
                    falsePositives++;
                }
            }
        }
        
        int totalNegative = trueNegatives + falsePositives;
        return totalNegative == 0 ? 0.0 : (double) trueNegatives / totalNegative;
    }
    
    /**
     * Calculates balanced accuracy (average of recall for each class).
     * 
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return balanced accuracy
     */
    public static double balancedAccuracy(int[] yTrue, int[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        // Find number of classes
        int numClasses = 0;
        for (int i = 0; i < yTrue.length; i++) {
            numClasses = Math.max(numClasses, Math.max(yTrue[i], yPred[i]) + 1);
        }
        
        double sumRecall = 0.0;
        int validClasses = 0;
        
        for (int c = 0; c < numClasses; c++) {
            int tp = 0, fn = 0;
            for (int i = 0; i < yTrue.length; i++) {
                if (yTrue[i] == c) {
                    if (yPred[i] == c) tp++;
                    else fn++;
                }
            }
            
            if (tp + fn > 0) {
                sumRecall += (double) tp / (tp + fn);
                validClasses++;
            }
        }
        
        return validClasses == 0 ? 0.0 : sumRecall / validClasses;
    }
    
    /**
     * Calculates Hamming loss for multi-label classification.
     * 
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return Hamming loss
     */
    public static double hammingLoss(int[] yTrue, int[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int incorrect = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] != yPred[i]) {
                incorrect++;
            }
        }
        
        return (double) incorrect / yTrue.length;
    }
    
    /**
     * Calculates explained variance score for regression.
     * 
     * @param yTrue true values
     * @param yPred predicted values
     * @return explained variance
     */
    public static double explainedVariance(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int n = yTrue.length;
        
        // Calculate residuals
        double[] residuals = new double[n];
        double residualMean = 0.0;
        for (int i = 0; i < n; i++) {
            residuals[i] = yTrue[i] - yPred[i];
            residualMean += residuals[i];
        }
        residualMean /= n;
        
        // Variance of residuals
        double residualVar = 0.0;
        for (double r : residuals) {
            residualVar += (r - residualMean) * (r - residualMean);
        }
        residualVar /= n;
        
        // Variance of true values
        double trueMean = 0.0;
        for (double y : yTrue) {
            trueMean += y;
        }
        trueMean /= n;
        
        double trueVar = 0.0;
        for (double y : yTrue) {
            trueVar += (y - trueMean) * (y - trueMean);
        }
        trueVar /= n;
        
        if (trueVar == 0) {
            return 0.0;
        }
        
        return 1 - residualVar / trueVar;
    }
}
