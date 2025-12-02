package com.mindforge.classification;

import java.io.Serializable;
import java.util.*;

/**
 * Stacking Classifier.
 * 
 * A meta-ensemble that trains multiple base classifiers and uses
 * their predictions as features for a final meta-classifier.
 * 
 * @author MindForge
 */
public class StackingClassifier implements Classifier<double[]>, Serializable {
    private static final long serialVersionUID = 1L;
    
    private List<Classifier<double[]>> baseClassifiers;
    private Classifier<double[]> metaClassifier;
    private boolean useProba;
    private boolean passthrough;
    private int cvFolds;
    
    private boolean trained;
    private int[] classes;
    private int numClasses;
    
    /**
     * Creates a StackingClassifier with default meta-classifier.
     */
    public StackingClassifier() {
        this.baseClassifiers = new ArrayList<>();
        this.metaClassifier = new DecisionTreeClassifier();
        this.useProba = false;
        this.passthrough = false;
        this.cvFolds = 5;
        this.trained = false;
    }
    
    /**
     * Creates a StackingClassifier with specified meta-classifier.
     */
    public StackingClassifier(Classifier<double[]> metaClassifier) {
        this();
        if (metaClassifier == null) {
            throw new IllegalArgumentException("Meta-classifier cannot be null");
        }
        this.metaClassifier = metaClassifier;
    }
    
    /**
     * Builder pattern for StackingClassifier.
     */
    public static class Builder {
        private StackingClassifier stacking = new StackingClassifier();
        
        public Builder addClassifier(Classifier<double[]> classifier) {
            if (classifier != null) {
                stacking.baseClassifiers.add(classifier);
            }
            return this;
        }
        
        public Builder setMetaClassifier(Classifier<double[]> meta) {
            if (meta != null) {
                stacking.metaClassifier = meta;
            }
            return this;
        }
        
        public Builder useProba(boolean useProba) {
            stacking.useProba = useProba;
            return this;
        }
        
        public Builder passthrough(boolean passthrough) {
            stacking.passthrough = passthrough;
            return this;
        }
        
        public Builder cvFolds(int folds) {
            if (folds >= 2) {
                stacking.cvFolds = folds;
            }
            return this;
        }
        
        public StackingClassifier build() {
            return stacking;
        }
    }
    
    /**
     * Adds a base classifier.
     */
    public StackingClassifier addClassifier(Classifier<double[]> classifier) {
        if (classifier == null) {
            throw new IllegalArgumentException("Classifier cannot be null");
        }
        baseClassifiers.add(classifier);
        return this;
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("X and y cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same length");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("X cannot be empty");
        }
        if (baseClassifiers.isEmpty()) {
            throw new IllegalStateException("No base classifiers added");
        }
        
        int n = X.length;
        int nClassifiers = baseClassifiers.size();
        
        // Find unique classes
        Set<Integer> classSet = new TreeSet<>();
        for (int label : y) {
            classSet.add(label);
        }
        numClasses = classSet.size();
        classes = new int[numClasses];
        int idx = 0;
        for (int c : classSet) {
            classes[idx++] = c;
        }
        
        // Generate meta-features using cross-validation
        int metaFeatureSize = useProba ? nClassifiers * numClasses : nClassifiers;
        if (passthrough) {
            metaFeatureSize += X[0].length;
        }
        
        double[][] metaFeatures = new double[n][metaFeatureSize];
        
        // Simple holdout approach for generating meta-features
        // Train each base classifier on all data, get predictions
        int foldSize = n / cvFolds;
        
        for (int fold = 0; fold < cvFolds; fold++) {
            int startIdx = fold * foldSize;
            int endIdx = (fold == cvFolds - 1) ? n : (fold + 1) * foldSize;
            
            // Create train/val split for this fold
            List<Integer> trainIndices = new ArrayList<>();
            List<Integer> valIndices = new ArrayList<>();
            
            for (int i = 0; i < n; i++) {
                if (i >= startIdx && i < endIdx) {
                    valIndices.add(i);
                } else {
                    trainIndices.add(i);
                }
            }
            
            // Get training data for this fold
            double[][] XTrain = new double[trainIndices.size()][];
            int[] yTrain = new int[trainIndices.size()];
            for (int i = 0; i < trainIndices.size(); i++) {
                XTrain[i] = X[trainIndices.get(i)];
                yTrain[i] = y[trainIndices.get(i)];
            }
            
            // Train base classifiers and generate predictions for validation
            for (int c = 0; c < nClassifiers; c++) {
                Classifier<double[]> clf = createCopy(baseClassifiers.get(c));
                clf.train(XTrain, yTrain);
                
                for (int valIdx : valIndices) {
                    if (useProba && clf instanceof ProbabilisticClassifier) {
                        double[] proba = ((ProbabilisticClassifier<double[]>) clf).predictProba(X[valIdx]);
                        for (int k = 0; k < numClasses; k++) {
                            metaFeatures[valIdx][c * numClasses + k] = k < proba.length ? proba[k] : 0;
                        }
                    } else {
                        int pred = clf.predict(X[valIdx]);
                        metaFeatures[valIdx][c] = pred;
                    }
                }
            }
        }
        
        // Add passthrough features
        if (passthrough) {
            int offset = useProba ? nClassifiers * numClasses : nClassifiers;
            for (int i = 0; i < n; i++) {
                System.arraycopy(X[i], 0, metaFeatures[i], offset, X[i].length);
            }
        }
        
        // Train all base classifiers on full data
        for (Classifier<double[]> clf : baseClassifiers) {
            clf.train(X, y);
        }
        
        // Train meta-classifier
        metaClassifier.train(metaFeatures, y);
        
        trained = true;
    }
    
    /**
     * Creates a copy of the classifier for CV training.
     */
    private Classifier<double[]> createCopy(Classifier<double[]> original) {
        // For simplicity, create new instances of known types
        if (original instanceof DecisionTreeClassifier) {
            return new DecisionTreeClassifier();
        } else if (original instanceof KNearestNeighbors) {
            return new KNearestNeighbors(3);
        } else if (original instanceof GaussianNaiveBayes) {
            return new GaussianNaiveBayes();
        } else if (original instanceof BaggingClassifier) {
            return new BaggingClassifier.Builder().nEstimators(10).build();
        }
        // Fallback: return original (not ideal but works for prediction)
        return original;
    }
    
    @Override
    public int predict(double[] x) {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        if (x == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        
        double[] metaFeature = generateMetaFeature(x);
        return metaClassifier.predict(metaFeature);
    }
    
    @Override
    public int[] predict(double[][] X) {
        if (X == null) {
            throw new IllegalArgumentException("X cannot be null");
        }
        
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    private double[] generateMetaFeature(double[] x) {
        int nClassifiers = baseClassifiers.size();
        int featureSize = useProba ? nClassifiers * numClasses : nClassifiers;
        if (passthrough) {
            featureSize += x.length;
        }
        
        double[] metaFeature = new double[featureSize];
        
        for (int c = 0; c < nClassifiers; c++) {
            Classifier<double[]> clf = baseClassifiers.get(c);
            
            if (useProba && clf instanceof ProbabilisticClassifier) {
                double[] proba = ((ProbabilisticClassifier<double[]>) clf).predictProba(x);
                for (int k = 0; k < numClasses; k++) {
                    metaFeature[c * numClasses + k] = k < proba.length ? proba[k] : 0;
                }
            } else {
                metaFeature[c] = clf.predict(x);
            }
        }
        
        if (passthrough) {
            int offset = useProba ? nClassifiers * numClasses : nClassifiers;
            System.arraycopy(x, 0, metaFeature, offset, x.length);
        }
        
        return metaFeature;
    }
    
    @Override
    public int getNumClasses() {
        return numClasses;
    }
    
    public double score(double[][] X, int[] y) {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        
        int[] predictions = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
    }
    
    public boolean isTrained() {
        return trained;
    }
    
    // Getters
    public List<Classifier<double[]>> getBaseClassifiers() {
        return new ArrayList<>(baseClassifiers);
    }
    
    public Classifier<double[]> getMetaClassifier() {
        return metaClassifier;
    }
    
    public int getNumBaseClassifiers() {
        return baseClassifiers.size();
    }
    
    public int[] getClasses() {
        return classes != null ? classes.clone() : null;
    }
}
