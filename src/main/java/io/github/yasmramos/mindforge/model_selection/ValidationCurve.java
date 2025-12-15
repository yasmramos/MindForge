package io.github.yasmramos.mindforge.model_selection;

import io.github.yasmramos.mindforge.classification.Classifier;
import io.github.yasmramos.mindforge.validation.Metrics;

import java.io.Serializable;
import java.util.Random;
import java.util.function.Function;

/**
 * Validation curve calculation.
 * Determines training and test scores for varying parameter values.
 */
public class ValidationCurve implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int cv;
    private final long randomSeed;
    
    private double[] paramRange;
    private double[] trainScoresMean;
    private double[] trainScoresStd;
    private double[] testScoresMean;
    private double[] testScoresStd;
    
    public ValidationCurve(int cv, long randomSeed) {
        this.cv = cv;
        this.randomSeed = randomSeed;
    }
    
    public ValidationCurve() {
        this(5, 42);
    }
    
    /**
     * Generate validation curve data.
     * @param modelFactory Function that creates model given parameter value
     * @param X Training features
     * @param y Training labels
     * @param paramRange Array of parameter values to test
     */
    public void generate(Function<Double, Classifier> modelFactory, double[][] X, int[] y, double[] paramRange) {
        this.paramRange = paramRange;
        int nParams = paramRange.length;
        
        trainScoresMean = new double[nParams];
        trainScoresStd = new double[nParams];
        testScoresMean = new double[nParams];
        testScoresStd = new double[nParams];
        
        Random random = new Random(randomSeed);
        
        for (int paramIdx = 0; paramIdx < nParams; paramIdx++) {
            double paramValue = paramRange[paramIdx];
            double[] trainScores = new double[cv];
            double[] testScores = new double[cv];
            
            int[] indices = new int[X.length];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            shuffle(indices, random);
            
            for (int fold = 0; fold < cv; fold++) {
                int testSize = X.length / cv;
                int testStart = fold * testSize;
                int testEnd = (fold == cv - 1) ? X.length : testStart + testSize;
                
                int trainSize = X.length - (testEnd - testStart);
                double[][] XTrain = new double[trainSize][];
                int[] yTrain = new int[trainSize];
                double[][] XTest = new double[testEnd - testStart][];
                int[] yTest = new int[testEnd - testStart];
                
                int trainIdx = 0, testIdx = 0;
                for (int i = 0; i < X.length; i++) {
                    int idx = indices[i];
                    if (i >= testStart && i < testEnd) {
                        XTest[testIdx] = X[idx];
                        yTest[testIdx++] = y[idx];
                    } else {
                        XTrain[trainIdx] = X[idx];
                        yTrain[trainIdx++] = y[idx];
                    }
                }
                
                try {
                    Classifier model = modelFactory.apply(paramValue);
                    model.fit(XTrain, yTrain);
                    
                    int[] trainPred = model.predict(XTrain);
                    int[] testPred = model.predict(XTest);
                    
                    trainScores[fold] = Metrics.accuracy(yTrain, trainPred);
                    testScores[fold] = Metrics.accuracy(yTest, testPred);
                } catch (Exception e) {
                    trainScores[fold] = 0;
                    testScores[fold] = 0;
                }
            }
            
            trainScoresMean[paramIdx] = mean(trainScores);
            trainScoresStd[paramIdx] = std(trainScores);
            testScoresMean[paramIdx] = mean(testScores);
            testScoresStd[paramIdx] = std(testScores);
        }
    }
    
    private void shuffle(int[] array, Random random) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    
    private double mean(double[] values) {
        double sum = 0;
        for (double v : values) sum += v;
        return sum / values.length;
    }
    
    private double std(double[] values) {
        double m = mean(values);
        double sum = 0;
        for (double v : values) {
            double diff = v - m;
            sum += diff * diff;
        }
        return Math.sqrt(sum / values.length);
    }
    
    public double[] getParamRange() { return paramRange; }
    public double[] getTrainScoresMean() { return trainScoresMean; }
    public double[] getTrainScoresStd() { return trainScoresStd; }
    public double[] getTestScoresMean() { return testScoresMean; }
    public double[] getTestScoresStd() { return testScoresStd; }
}
