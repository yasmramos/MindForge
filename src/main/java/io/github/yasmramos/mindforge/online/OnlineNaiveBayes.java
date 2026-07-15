package io.github.yasmramos.mindforge.online;

import java.util.Arrays;

/**
 * Online Naive Bayes Classifier for streaming data.
 * Updates model parameters incrementally as new data arrives.
 */
public class OnlineNaiveBayes {
    
    private int nFeatures;
    private int nClasses;
    private double[][] featureMeans;
    private double[][] featureVariances;
    private double[] classPriors;
    private int[] classCounts;
    private int totalSamples;
    private boolean initialized;
    
    public OnlineNaiveBayes(int nFeatures, int nClasses) {
        this.nFeatures = nFeatures;
        this.nClasses = nClasses;
        this.featureMeans = new double[nClasses][nFeatures];
        this.featureVariances = new double[nClasses][nFeatures];
        this.classPriors = new double[nClasses];
        this.classCounts = new int[nClasses];
        this.totalSamples = 0;
        this.initialized = false;
    }
    
    public void partialFit(double[][] X, int[] y) {
        if (!initialized) {
            initialize(X, y);
            return;
        }
        
        for (int i = 0; i < X.length; i++) {
            int label = y[i];
            if (label < 0 || label >= nClasses) continue;
            
            int prevCount = classCounts[label];
            int newCount = prevCount + 1;
            
            double[] oldMean = Arrays.copyOf(featureMeans[label], nFeatures);
            
            for (int j = 0; j < nFeatures; j++) {
                double x = X[i][j];
                double oldTotalSum = oldMean[j] * prevCount;
                double newTotalSum = oldTotalSum + x;
                featureMeans[label][j] = newTotalSum / newCount;
                
                if (prevCount > 1) {
                    double oldVariance = featureVariances[label][j];
                    double delta = x - oldMean[j];
                    double newVariance = ((prevCount - 1) * oldVariance + delta * delta * prevCount / newCount) / prevCount;
                    featureVariances[label][j] = Math.max(newVariance, 1e-10);
                } else {
                    featureVariances[label][j] = 1e-10;
                }
            }
            
            classCounts[label] = newCount;
            totalSamples++;
        }
        
        updatePriors();
    }
    
    private void initialize(double[][] X, int[] y) {
        for (int i = 0; i < X.length; i++) {
            int label = (int)y[i];
            if (label < 0 || label >= nClasses) continue;
            
            classCounts[label]++;
            totalSamples++;
            
            for (int j = 0; j < nFeatures; j++) {
                double x = X[i][j];
                int count = classCounts[label];
                double oldMean = featureMeans[label][j];
                featureMeans[label][j] = oldMean + (x - oldMean) / count;
                
                if (count > 1) {
                    double delta = x - oldMean;
                    featureVariances[label][j] += delta * (x - featureMeans[label][j]);
                }
            }
        }
        
        for (int c = 0; c < nClasses; c++) {
            if (classCounts[c] > 1) {
                for (int j = 0; j < nFeatures; j++) {
                    featureVariances[c][j] = Math.max(featureVariances[c][j] / (classCounts[c] - 1), 1e-10);
                }
            } else {
                for (int j = 0; j < nFeatures; j++) {
                    featureVariances[c][j] = 1e-10;
                }
            }
        }
        
        updatePriors();
        initialized = true;
    }
    
    private void updatePriors() {
        for (int c = 0; c < nClasses; c++) {
            classPriors[c] = (classCounts[c] + 1.0) / (totalSamples + nClasses);
        }
    }
    
    public int predict(double[] x) {
        double maxLogProb = Double.NEGATIVE_INFINITY;
        int predictedClass = 0;
        
        for (int c = 0; c < nClasses; c++) {
            if (classCounts[c] == 0) continue;
            
            double logProb = Math.log(classPriors[c]);
            
            for (int j = 0; j < nFeatures; j++) {
                double mean = featureMeans[c][j];
                double variance = Math.max(featureVariances[c][j], 1e-10);
                double diff = x[j] - mean;
                logProb -= 0.5 * (Math.log(2 * Math.PI * variance) + (diff * diff) / variance);
            }
            
            if (logProb > maxLogProb) {
                maxLogProb = logProb;
                predictedClass = c;
            }
        }
        
        return predictedClass;
    }
    
    public double[] predictProba(double[] x) {
        double[] logProbs = new double[nClasses];
        
        for (int c = 0; c < nClasses; c++) {
            if (classCounts[c] == 0) {
                logProbs[c] = Double.NEGATIVE_INFINITY;
                continue;
            }
            
            double logProb = Math.log(classPriors[c]);
            
            for (int j = 0; j < nFeatures; j++) {
                double mean = featureMeans[c][j];
                double variance = Math.max(featureVariances[c][j], 1e-10);
                double diff = x[j] - mean;
                logProb -= 0.5 * (Math.log(2 * Math.PI * variance) + (diff * diff) / variance);
            }
            
            logProbs[c] = logProb;
        }
        
        double maxLogProb = Arrays.stream(logProbs).max().orElse(0.0);
        double[] probs = new double[nClasses];
        double sum = 0.0;
        
        for (int c = 0; c < nClasses; c++) {
            probs[c] = Math.exp(logProbs[c] - maxLogProb);
            sum += probs[c];
        }
        
        for (int c = 0; c < nClasses; c++) {
            probs[c] /= sum;
        }
        
        return probs;
    }
}
