package io.github.yasmramos.mindforge.nlp;

import java.io.Serializable;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Convert a collection of text documents to a matrix of token counts.
 * Similar to scikit-learn's CountVectorizer.
 */
public class CountVectorizer implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final boolean lowercase;
    private final boolean binary;
    private final int minDf;
    private final int maxDf;
    private final Integer maxFeatures;
    private final String tokenPattern;
    private final Set<String> stopWords;
    private final int[] ngramRange;
    
    private Map<String, Integer> vocabulary;
    private String[] featureNames;
    private Pattern compiledPattern;
    
    private CountVectorizer(Builder builder) {
        this.lowercase = builder.lowercase;
        this.binary = builder.binary;
        this.minDf = builder.minDf;
        this.maxDf = builder.maxDf;
        this.maxFeatures = builder.maxFeatures;
        this.tokenPattern = builder.tokenPattern;
        this.stopWords = builder.stopWords;
        this.ngramRange = builder.ngramRange;
        this.compiledPattern = Pattern.compile(tokenPattern);
    }
    
    /**
     * Learn the vocabulary from documents.
     * @param documents Array of text documents
     */
    public void fit(String[] documents) {
        Map<String, Integer> termDocFreq = new HashMap<>();
        
        for (String doc : documents) {
            Set<String> docTerms = new HashSet<>(tokenize(doc));
            for (String term : docTerms) {
                termDocFreq.merge(term, 1, Integer::sum);
            }
        }
        
        // Filter by document frequency
        int nDocs = documents.length;
        List<Map.Entry<String, Integer>> validTerms = new ArrayList<>();
        
        for (Map.Entry<String, Integer> entry : termDocFreq.entrySet()) {
            int df = entry.getValue();
            if (df >= minDf && (maxDf <= 0 || df <= maxDf * nDocs)) {
                validTerms.add(entry);
            }
        }
        
        // Sort by frequency (descending) then alphabetically
        validTerms.sort((a, b) -> {
            int cmp = b.getValue().compareTo(a.getValue());
            return cmp != 0 ? cmp : a.getKey().compareTo(b.getKey());
        });
        
        // Limit features if specified
        if (maxFeatures != null && validTerms.size() > maxFeatures) {
            validTerms = validTerms.subList(0, maxFeatures);
        }
        
        // Build vocabulary
        vocabulary = new HashMap<>();
        featureNames = new String[validTerms.size()];
        
        // Sort alphabetically for consistent ordering
        validTerms.sort(Comparator.comparing(Map.Entry::getKey));
        
        for (int i = 0; i < validTerms.size(); i++) {
            String term = validTerms.get(i).getKey();
            vocabulary.put(term, i);
            featureNames[i] = term;
        }
    }
    
    /**
     * Transform documents to a term-document matrix.
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
            
            for (Map.Entry<String, Integer> entry : counts.entrySet()) {
                Integer idx = vocabulary.get(entry.getKey());
                if (idx != null) {
                    result[i][idx] = binary ? 1.0 : entry.getValue();
                }
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
    public int getVocabularySize() { return vocabulary != null ? vocabulary.size() : 0; }
    
    public static Set<String> getEnglishStopWords() {
        return new HashSet<>(Arrays.asList(
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with", "the", "this", "but", "they",
            "have", "had", "what", "when", "where", "who", "which", "why", "how"
        ));
    }
    
    public static class Builder {
        private boolean lowercase = true;
        private boolean binary = false;
        private int minDf = 1;
        private int maxDf = -1;
        private Integer maxFeatures = null;
        private String tokenPattern = "\\b\\w+\\b";
        private Set<String> stopWords = null;
        private int[] ngramRange = {1, 1};
        
        public Builder lowercase(boolean lowercase) { this.lowercase = lowercase; return this; }
        public Builder binary(boolean binary) { this.binary = binary; return this; }
        public Builder minDf(int minDf) { this.minDf = minDf; return this; }
        public Builder maxDf(int maxDf) { this.maxDf = maxDf; return this; }
        public Builder maxFeatures(int maxFeatures) { this.maxFeatures = maxFeatures; return this; }
        public Builder tokenPattern(String pattern) { this.tokenPattern = pattern; return this; }
        public Builder stopWords(Set<String> stopWords) { this.stopWords = stopWords; return this; }
        public Builder ngramRange(int min, int max) { this.ngramRange = new int[]{min, max}; return this; }
        
        public CountVectorizer build() { return new CountVectorizer(this); }
    }
}
