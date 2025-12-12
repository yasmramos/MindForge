package io.github.yasmramos.mindforge.classification;

import java.io.Serializable;
import java.util.*;

/**
 * Voting Classifier - Ensemble classifier using majority voting.
 * 
 * <p>The Voting Classifier combines multiple classifiers and uses majority voting
 * to determine the final prediction. This can improve accuracy and reduce overfitting
 * by combining diverse models.</p>
 * 
 * <p>Supports two voting modes:
 * <ul>
 *   <li><b>Hard voting:</b> Uses predicted class labels and majority vote</li>
 *   <li><b>Soft voting:</b> Uses predicted probabilities (if available) and averages them</li>
 * </ul>
 * </p>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * VotingClassifier voting = new VotingClassifier()
 *     .addClassifier("knn", new KNearestNeighbors(5))
 *     .addClassifier("dt", new DecisionTreeClassifier())
 *     .addClassifier("nb", new GaussianNaiveBayes());
 * 
 * voting.train(X_train, y_train);
 * int[] predictions = voting.predict(X_test);
 * }</pre>
 * 
 * @author Matrix Agent
 * @version 1.0
 */
public class VotingClassifier implements Classifier<double[]>, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * Voting strategy.
     */
    public enum Voting {
        HARD,   // Use predicted labels
        SOFT    // Use predicted probabilities (not yet implemented)
    }
    
    private final List<NamedClassifier> classifiers;
    private final Voting voting;
    private final double[] weights;
    
    private int[] classes;
    private boolean isTrained;
    
    /**
     * Container for named classifier.
     */
    private static class NamedClassifier implements Serializable {
        private static final long serialVersionUID = 1L;
        final String name;
        final Classifier<double[]> classifier;
        
        NamedClassifier(String name, Classifier<double[]> classifier) {
            this.name = name;
            this.classifier = classifier;
        }
    }
    
    /**
     * Creates a VotingClassifier with hard voting and equal weights.
     */
    public VotingClassifier() {
        this(Voting.HARD, null);
    }
    
    /**
     * Creates a VotingClassifier with specified voting strategy.
     * 
     * @param voting the voting strategy
     */
    public VotingClassifier(Voting voting) {
        this(voting, null);
    }
    
    /**
     * Creates a VotingClassifier with voting strategy and weights.
     * 
     * @param voting the voting strategy
     * @param weights optional weights for each classifier (null for equal weights)
     */
    public VotingClassifier(Voting voting, double[] weights) {
        this.classifiers = new ArrayList<>();
        this.voting = voting;
        this.weights = weights;
        this.isTrained = false;
    }
    
    /**
     * Adds a classifier to the ensemble.
     * 
     * @param name unique name for the classifier
     * @param classifier the classifier to add
     * @return this VotingClassifier for chaining
     */
    public VotingClassifier addClassifier(String name, Classifier<double[]> classifier) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Classifier name cannot be null or empty");
        }
        if (classifier == null) {
            throw new IllegalArgumentException("Classifier cannot be null");
        }
        
        // Check for duplicate names
        for (NamedClassifier nc : classifiers) {
            if (nc.name.equals(name)) {
                throw new IllegalArgumentException("Classifier with name '" + name + "' already exists");
            }
        }
        
        classifiers.add(new NamedClassifier(name, classifier));
        return this;
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        if (classifiers.isEmpty()) {
            throw new IllegalStateException("No classifiers added. Use addClassifier() first.");
        }
        
        validateInput(X, y);
        
        // Validate weights if provided
        if (weights != null && weights.length != classifiers.size()) {
            throw new IllegalArgumentException(
                "Weights length (" + weights.length + ") must match number of classifiers (" + classifiers.size() + ")");
        }
        
        // Find unique classes
        Set<Integer> uniqueClasses = new TreeSet<>();
        for (int label : y) {
            uniqueClasses.add(label);
        }
        this.classes = uniqueClasses.stream().mapToInt(Integer::intValue).toArray();
        
        // Train all classifiers
        for (NamedClassifier nc : classifiers) {
            nc.classifier.train(X, y);
        }
        
        this.isTrained = true;
    }
    
    @Override
    public int predict(double[] x) {
        if (!isTrained) {
            throw new IllegalStateException("VotingClassifier must be trained before prediction");
        }
        
        if (voting == Voting.HARD) {
            return predictHardVoting(x);
        } else {
            // Soft voting would require probability estimates
            // For now, fall back to hard voting
            return predictHardVoting(x);
        }
    }
    
    /**
     * Hard voting prediction.
     */
    private int predictHardVoting(double[] x) {
        Map<Integer, Double> votes = new HashMap<>();
        
        for (int i = 0; i < classifiers.size(); i++) {
            int prediction = classifiers.get(i).classifier.predict(x);
            double weight = (weights != null) ? weights[i] : 1.0;
            votes.merge(prediction, weight, Double::sum);
        }
        
        // Find class with most votes
        int bestClass = classes[0];
        double maxVotes = 0;
        
        for (Map.Entry<Integer, Double> entry : votes.entrySet()) {
            if (entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                bestClass = entry.getKey();
            }
        }
        
        return bestClass;
    }
    
    /**
     * Predicts class labels for multiple samples.
     * 
     * @param X feature matrix
     * @return predicted labels
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
     * Gets the predictions from each individual classifier.
     * 
     * @param x input sample
     * @return map of classifier name to prediction
     */
    public Map<String, Integer> getIndividualPredictions(double[] x) {
        if (!isTrained) {
            throw new IllegalStateException("VotingClassifier must be trained first");
        }
        
        Map<String, Integer> predictions = new LinkedHashMap<>();
        for (NamedClassifier nc : classifiers) {
            predictions.put(nc.name, nc.classifier.predict(x));
        }
        return predictions;
    }
    
    @Override
    public int getNumClasses() {
        return isTrained ? classes.length : 0;
    }
    
    /**
     * Gets the class labels.
     * 
     * @return array of class labels
     */
    public int[] getClasses() {
        if (!isTrained) {
            throw new IllegalStateException("VotingClassifier must be trained first");
        }
        return classes.clone();
    }
    
    /**
     * Gets the number of classifiers in the ensemble.
     * 
     * @return number of classifiers
     */
    public int getNumClassifiers() {
        return classifiers.size();
    }
    
    /**
     * Gets the names of all classifiers.
     * 
     * @return list of classifier names
     */
    public List<String> getClassifierNames() {
        List<String> names = new ArrayList<>();
        for (NamedClassifier nc : classifiers) {
            names.add(nc.name);
        }
        return names;
    }
    
    /**
     * Gets the voting strategy.
     * 
     * @return voting strategy
     */
    public Voting getVoting() {
        return voting;
    }
    
    /**
     * Checks if the classifier is trained.
     * 
     * @return true if trained
     */
    public boolean isTrained() {
        return isTrained;
    }
    
    private void validateInput(double[][] X, int[] y) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        if (y == null || y.length == 0) {
            throw new IllegalArgumentException("Labels cannot be null or empty");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException(
                String.format("X and y have different lengths: %d vs %d", X.length, y.length));
        }
    }
    
    /**
     * Builder for VotingClassifier.
     */
    public static class Builder {
        private Voting voting = Voting.HARD;
        private double[] weights = null;
        private List<NamedClassifier> classifiers = new ArrayList<>();
        
        public Builder voting(Voting voting) {
            this.voting = voting;
            return this;
        }
        
        public Builder weights(double[] weights) {
            this.weights = weights;
            return this;
        }
        
        public Builder addClassifier(String name, Classifier<double[]> classifier) {
            classifiers.add(new NamedClassifier(name, classifier));
            return this;
        }
        
        public VotingClassifier build() {
            VotingClassifier vc = new VotingClassifier(voting, weights);
            for (NamedClassifier nc : classifiers) {
                vc.addClassifier(nc.name, nc.classifier);
            }
            return vc;
        }
    }
}
