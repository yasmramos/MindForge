package io.github.yasmramos.mindforge.nlp;

import java.io.Serializable;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Convert a collection of text documents to a matrix of TF-IDF features.
 * TF-IDF = Term Frequency * Inverse Document Frequency
 * Similar to scikit-learn's TfidfVectorizer.
 */
public class TfidfVectorizer implements Serializable {
    private static final long serialVersionUID = 1L;
    
    public enum Norm { L1, L2, NONE }
    
    private final boolean lowercase;
    private final boolean useIdf;
    private final boolean smoothIdf;
    private final boolean sublinearTf;
    private final Norm norm;
    private final int minDf;
    private final int maxDf;
    private final Integer maxFeatures;
    private final String tokenPattern;
    private final Set<String> stopWords;
    private final int[] ngramRange;
    
    private Map<String, Integer> vocabulary;
    private String[] featureNames;
    private double[] idfWeights;
    private Pattern compiledPattern;
    private int nDocuments;
    
    private TfidfVectorizer(Builder builder) {
        this.lowercase = builder.lowercase;
        this.useIdf = builder.useIdf;
        this.smoothIdf = builder.smoothIdf;
        this.sublinearTf = builder.sublinearTf;
        this.norm = builder.norm;
        this.minDf = builder.minDf;
        this.maxDf = builder.maxDf;
        this.maxFeatures = builder.maxFeatures;
        this.tokenPattern = builder.tokenPattern;
        this.stopWords = builder.stopWords;
        this.ngramRange = builder.ngramRange;
        this.compiledPattern = Pattern.compile(tokenPattern);
    }
    
    /**
     * Learn vocabulary and IDF weights from documents.
     * @param documents Array of text documents
     */
    public void fit(String[] documents) {
        nDocuments = documents.length;
        Map<String, Integer> termDocFreq = new HashMap<>();
        Map<String, Integer> termFreq = new HashMap<>();
        
        for (String doc : documents) {
            List<String> tokens = tokenize(doc);
            Set<String> docTerms = new HashSet<>(tokens);
            
            for (String token : tokens) {
                termFreq.merge(token, 1, Integer::sum);
            }
            for (String term : docTerms) {
                termDocFreq.merge(term, 1, Integer::sum);
            }
        }
        
        // Filter by document frequency
        List<Map.Entry<String, Integer>> validTerms = new ArrayList<>();
        
        for (Map.Entry<String, Integer> entry : termDocFreq.entrySet()) {
            int df = entry.getValue();
            double dfRatio = (double) df / nDocuments;
            if (df >= minDf && (maxDf <= 0 || dfRatio <= maxDf)) {
                validTerms.add(entry);
            }
        }
        
        // Sort by total frequency (descending) then alphabetically
        validTerms.sort((a, b) -> {
            int freqA = termFreq.getOrDefault(a.getKey(), 0);
            int freqB = termFreq.getOrDefault(b.getKey(), 0);
            int cmp = Integer.compare(freqB, freqA);
            return cmp != 0 ? cmp : a.getKey().compareTo(b.getKey());
        });
        
        // Limit features if specified
        if (maxFeatures != null && validTerms.size() > maxFeatures) {
            validTerms = validTerms.subList(0, maxFeatures);
        }
        
        // Sort alphabetically for consistent ordering
        validTerms.sort(Comparator.comparing(Map.Entry::getKey));
        
        // Build vocabulary and IDF weights
        vocabulary = new HashMap<>();
        featureNames = new String[validTerms.size()];
        idfWeights = new double[validTerms.size()];
        
        for (int i = 0; i < validTerms.size(); i++) {
            String term = validTerms.get(i).getKey();
            int df = validTerms.get(i).getValue();
            
            vocabulary.put(term, i);
            featureNames[i] = term;
            
            if (useIdf) {
                if (smoothIdf) {
                    idfWeights[i] = Math.log((double) (nDocuments + 1) / (df + 1)) + 1;
                } else {
                    idfWeights[i] = Math.log((double) nDocuments / df) + 1;
                }
            } else {
                idfWeights[i] = 1.0;
            }
        }
    }
    
    /**
     * Transform documents to TF-IDF matrix.
     * @param documents Array of text documents
     * @return Matrix of shape [n_documents, n_features]
     */
    public double[][] transform(String[] documents) {
        if (vocabulary == null) {
            throw new IllegalStateException("Vectorizer must be fitted before transform");
        }
        
        double[][] result = new double[documents.length][vocabulary.size()];
        
        for (int i = 0; i < documents.length; i++) {
            List<String> tokens = tokenize(documents[i]);
            Map<String, Integer> counts = countTokens(tokens);
            
            // Calculate TF-IDF
            for (Map.Entry<String, Integer> entry : counts.entrySet()) {
                Integer idx = vocabulary.get(entry.getKey());
                if (idx != null) {
                    double tf = entry.getValue();
                    if (sublinearTf) {
                        tf = 1 + Math.log(tf);
                    }
                    result[i][idx] = tf * idfWeights[idx];
                }
            }
            
            // Normalize
            if (norm != Norm.NONE) {
                normalizeVector(result[i]);
            }
        }
        
        return result;
    }
    
    /**
     * Fit and transform in one step.
     * @param documents Array of text documents
     * @return Matrix of shape [n_documents, n_features]
     */
    public double[][] fitTransform(String[] documents) {
        fit(documents);
        return transform(documents);
    }
    
    private void normalizeVector(double[] vector) {
        double normValue = 0;
        
        if (norm == Norm.L2) {
            for (double v : vector) {
                normValue += v * v;
            }
            normValue = Math.sqrt(normValue);
        } else if (norm == Norm.L1) {
            for (double v : vector) {
                normValue += Math.abs(v);
            }
        }
        
        if (normValue > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= normValue;
            }
        }
    }
    
    private List<String> tokenize(String text) {
        if (lowercase) {
            text = text.toLowerCase();
        }
        
        List<String> tokens = new ArrayList<>();
        var matcher = compiledPattern.matcher(text);
        List<String> baseTokens = new ArrayList<>();
        
        while (matcher.find()) {
            String token = matcher.group();
            if (stopWords == null || !stopWords.contains(token)) {
                baseTokens.add(token);
            }
        }
        
        // Generate n-grams
        for (int n = ngramRange[0]; n <= ngramRange[1]; n++) {
            for (int i = 0; i <= baseTokens.size() - n; i++) {
                StringBuilder ngram = new StringBuilder();
                for (int j = 0; j < n; j++) {
                    if (j > 0) ngram.append(" ");
                    ngram.append(baseTokens.get(i + j));
                }
                tokens.add(ngram.toString());
            }
        }
        
        return tokens;
    }
    
    private Map<String, Integer> countTokens(List<String> tokens) {
        Map<String, Integer> counts = new HashMap<>();
        for (String token : tokens) {
            counts.merge(token, 1, Integer::sum);
        }
        return counts;
    }
    
    // Getters
    public Map<String, Integer> getVocabulary() { return vocabulary != null ? new HashMap<>(vocabulary) : null; }
    public String[] getFeatureNames() { return featureNames != null ? featureNames.clone() : null; }
    public double[] getIdfWeights() { return idfWeights != null ? idfWeights.clone() : null; }
    public int getVocabularySize() { return vocabulary != null ? vocabulary.size() : 0; }
    
    public static class Builder {
        private boolean lowercase = true;
        private boolean useIdf = true;
        private boolean smoothIdf = true;
        private boolean sublinearTf = false;
        private Norm norm = Norm.L2;
        private int minDf = 1;
        private int maxDf = -1;
        private Integer maxFeatures = null;
        private String tokenPattern = "\\b\\w+\\b";
        private Set<String> stopWords = null;
        private int[] ngramRange = {1, 1};
        
        public Builder lowercase(boolean lowercase) { this.lowercase = lowercase; return this; }
        public Builder useIdf(boolean useIdf) { this.useIdf = useIdf; return this; }
        public Builder smoothIdf(boolean smoothIdf) { this.smoothIdf = smoothIdf; return this; }
        public Builder sublinearTf(boolean sublinearTf) { this.sublinearTf = sublinearTf; return this; }
        public Builder norm(Norm norm) { this.norm = norm; return this; }
        public Builder minDf(int minDf) { this.minDf = minDf; return this; }
        public Builder maxDf(int maxDf) { this.maxDf = maxDf; return this; }
        public Builder maxFeatures(int maxFeatures) { this.maxFeatures = maxFeatures; return this; }
        public Builder tokenPattern(String pattern) { this.tokenPattern = pattern; return this; }
        public Builder stopWords(Set<String> stopWords) { this.stopWords = stopWords; return this; }
        public Builder ngramRange(int min, int max) { this.ngramRange = new int[]{min, max}; return this; }
        
        public TfidfVectorizer build() { return new TfidfVectorizer(this); }
    }
}
