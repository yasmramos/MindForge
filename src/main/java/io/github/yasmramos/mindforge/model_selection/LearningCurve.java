package io.github.yasmramos.mindforge.model_selection;

import io.github.yasmramos.mindforge.classification.Classifier;
import io.github.yasmramos.mindforge.validation.Metrics;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Learning curve calculation.
 * Determines training and test scores for varying training set sizes.
 */
public class LearningCurve implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int cv;
    private final long randomSeed;
    
    private double[] trainSizes;
    private double[] trainScoresMean;
    private double[] trainScoresStd;
    private double[] testScoresMean;
    private double[] testScoresStd;
    
    public LearningCurve(int cv, long randomSeed) {
        this.cv = cv;
        this.randomSeed = randomSeed;
    }
    
    public LearningCurve() {
        this(5, 42);
    }
    
    /**
     * Generate learning curve data.
     * @param modelSupplier Supplier that creates new model instances
     * @param X Training features
     * @param y Training labels
     * @param trainSizes Array of training sizes (as fractions between 0 and 1)
     */
    public void generate(Supplier<Classifier> modelSupplier, double[][] X, int[] y, double[] trainSizes) {
        this.trainSizes = trainSizes;
        int nSizes = trainSizes.length;
        
        trainScoresMean = new double[nSizes];
        trainScoresStd = new double[nSizes];
        testScoresMean = new double[nSizes];
        testScoresStd = new double[nSizes];
        
        Random random = new Random(randomSeed);
        
        for (int sizeIdx = 0; sizeIdx < nSizes; sizeIdx++) {
            double fraction = trainSizes[sizeIdx];
            double[] trainScores = new double[cv];
            double[] testScores = new double[cv];
            
            for (int fold = 0; fold < cv; fold++) {
                // Create train/test split
                int[] indices = new int[X.length];
                for (int i = 0; i < indices.length; i++) indices[i] = i;
                shuffle(indices, random);
                
                int testSize = X.length / cv;
                int testStart = fold * testSize;
                int testEnd = (fold == cv - 1) ? X.length : testStart + testSize;
                
                int totalTrainSize = X.length - (testEnd - testStart);
                int actualTrainSize = (int) (totalTrainSize * fraction);
                actualTrainSize = Math.max(1, actualTrainSize);
                
                double[][] XTrain = new double[actualTrainSize][];
                int[] yTrain = new int[actualTrainSize];
                double[][] XTest = new double[testEnd - testStart][];
                int[] yTest = new int[testEnd - testStart];
                
                int trainIdx = 0, testIdx = 0;
                for (int i = 0; i < X.length; i++) {
                    int idx = indices[i];
                    if (i >= testStart && i < testEnd) {
                        XTest[testIdx] = X[idx];
                        yTest[testIdx++] = y[idx];
                    } else if (trainIdx < actualTrainSize) {
                        XTrain[trainIdx] = X[idx];
                        yTrain[trainIdx++] = y[idx];
                    }
                }
                
                try {
                    Classifier model = modelSupplier.get();
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
            
            trainScoresMean[sizeIdx] = mean(trainScores);
            trainScoresStd[sizeIdx] = std(trainScores);
            testScoresMean[sizeIdx] = mean(testScores);
            testScoresStd[sizeIdx] = std(testScores);
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
    
    public double[] getTrainSizes() { return trainSizes; }
    public double[] getTrainScoresMean() { return trainScoresMean; }
    public double[] getTrainScoresStd() { return trainScoresStd; }
    public double[] getTestScoresMean() { return testScoresMean; }
    public double[] getTestScoresStd() { return testScoresStd; }
}
