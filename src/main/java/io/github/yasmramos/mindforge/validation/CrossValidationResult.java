package io.github.yasmramos.mindforge.validation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Stores the results of cross-validation evaluation.
 * 
 * Contains scores for each fold and provides methods to compute
 * aggregate statistics like mean and standard deviation.
 * 
 * Example usage:
 * <pre>
 * CrossValidationResult result = CrossValidation.kFold(classifier, X, y, 5);
 * System.out.println("Mean Accuracy: " + result.getMean());
 * System.out.println("Std Dev: " + result.getStdDev());
 * System.out.println("All Scores: " + Arrays.toString(result.getScores()));
 * </pre>
 */
public class CrossValidationResult {
    
    private final List<Double> scores;
    private final String metricName;
    
    /**
     * Creates a new CrossValidationResult.
     * 
     * @param scores scores from each fold
     * @param metricName name of the metric used
     */
    public CrossValidationResult(List<Double> scores, String metricName) {
        this.scores = new ArrayList<>(scores);
        this.metricName = metricName;
    }
    
    /**
     * Creates a new CrossValidationResult.
     * 
     * @param scores array of scores from each fold
     * @param metricName name of the metric used
     */
    public CrossValidationResult(double[] scores, String metricName) {
        this.scores = new ArrayList<>();
        for (double score : scores) {
            this.scores.add(score);
        }
        this.metricName = metricName;
    }
    
    /**
     * Returns the scores from all folds.
     * 
     * @return array of scores
     */
    public double[] getScores() {
        return scores.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Returns the mean score across all folds.
     * 
     * @return mean score
     */
    public double getMean() {
        return scores.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);
    }
    
    /**
     * Returns the standard deviation of scores across all folds.
     * 
     * @return standard deviation
     */
    public double getStdDev() {
        double mean = getMean();
        double variance = scores.stream()
                .mapToDouble(score -> Math.pow(score - mean, 2))
                .average()
                .orElse(0.0);
        return Math.sqrt(variance);
    }
    
    /**
     * Returns the minimum score across all folds.
     * 
     * @return minimum score
     */
    public double getMin() {
        return scores.stream()
                .mapToDouble(Double::doubleValue)
                .min()
                .orElse(0.0);
    }
    
    /**
     * Returns the maximum score across all folds.
     * 
     * @return maximum score
     */
    public double getMax() {
        return scores.stream()
                .mapToDouble(Double::doubleValue)
                .max()
                .orElse(0.0);
    }
    
    /**
     * Returns the name of the metric used.
     * 
     * @return metric name
     */
    public String getMetricName() {
        return metricName;
    }
    
    /**
     * Returns the number of folds.
     * 
     * @return number of folds
     */
    public int getNumFolds() {
        return scores.size();
    }
    
    @Override
    public String toString() {
        return String.format(
            "CrossValidationResult{metric='%s', folds=%d, mean=%.4f, std=%.4f, min=%.4f, max=%.4f}",
            metricName, getNumFolds(), getMean(), getStdDev(), getMin(), getMax()
        );
    }
}
