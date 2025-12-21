package io.github.yasmramos.mindforge.nlp;

import java.io.Serializable;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Simple Word2Vec implementation using Skip-gram with negative sampling.
 * Learns dense vector representations of words from text.
 */
public class Word2Vec implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int vectorSize;
    private final int windowSize;
    private final int minCount;
    private final int negativeSamples;
    private final double learningRate;
    private final int epochs;
    private final long seed;
    
    private Map<String, Integer> wordIndex;
    private String[] indexToWord;
    private double[][] wordVectors;
    private double[][] contextVectors;
    private int[] wordCounts;
    private double[] samplingTable;
    
    private static final int TABLE_SIZE = 100000000;
    
    private Word2Vec(Builder builder) {
        this.vectorSize = builder.vectorSize;
        this.windowSize = builder.windowSize;
        this.minCount = builder.minCount;
        this.negativeSamples = builder.negativeSamples;
        this.learningRate = builder.learningRate;
        this.epochs = builder.epochs;
        this.seed = builder.seed;
    }
    
    /**
     * Train Word2Vec on a corpus of documents.
     * @param documents Array of text documents
     */
    public void fit(String[] documents) {
        Random random = new Random(seed);
        
        // Tokenize and count words
        List<List<String>> tokenizedDocs = new ArrayList<>();
        Map<String, Integer> wordFreq = new HashMap<>();
        
        Pattern pattern = Pattern.compile("\\b\\w+\\b");
        for (String doc : documents) {
            List<String> tokens = new ArrayList<>();
            var matcher = pattern.matcher(doc.toLowerCase());
            while (matcher.find()) {
                String token = matcher.group();
                tokens.add(token);
                wordFreq.merge(token, 1, Integer::sum);
            }
            tokenizedDocs.add(tokens);
        }
        
        // Build vocabulary (filter by minCount)
        List<Map.Entry<String, Integer>> sortedWords = new ArrayList<>(wordFreq.entrySet());
        sortedWords.removeIf(e -> e.getValue() < minCount);
        sortedWords.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        
        int vocabSize = sortedWords.size();
        wordIndex = new HashMap<>();
        indexToWord = new String[vocabSize];
        wordCounts = new int[vocabSize];
        
        for (int i = 0; i < vocabSize; i++) {
            String word = sortedWords.get(i).getKey();
            wordIndex.put(word, i);
            indexToWord[i] = word;
            wordCounts[i] = sortedWords.get(i).getValue();
        }
        
        // Initialize vectors
        wordVectors = new double[vocabSize][vectorSize];
        contextVectors = new double[vocabSize][vectorSize];
        
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < vectorSize; j++) {
                wordVectors[i][j] = (random.nextDouble() - 0.5) / vectorSize;
                contextVectors[i][j] = 0;
            }
        }
        
        // Build unigram table for negative sampling
        buildSamplingTable();
        
        // Convert tokenized docs to indices
        List<int[]> indexedDocs = new ArrayList<>();
        for (List<String> tokens : tokenizedDocs) {
            List<Integer> indices = new ArrayList<>();
            for (String token : tokens) {
                Integer idx = wordIndex.get(token);
                if (idx != null) {
                    indices.add(idx);
                }
            }
            if (!indices.isEmpty()) {
                indexedDocs.add(indices.stream().mapToInt(i -> i).toArray());
            }
        }
        
        // Training
        for (int epoch = 0; epoch < epochs; epoch++) {
            double currentLr = learningRate * (1 - (double) epoch / epochs);
            currentLr = Math.max(currentLr, learningRate * 0.0001);
            
            for (int[] doc : indexedDocs) {
                trainDocument(doc, currentLr, random);
            }
        }
    }
    
    private void trainDocument(int[] doc, double lr, Random random) {
        for (int pos = 0; pos < doc.length; pos++) {
            int wordIdx = doc[pos];
            
            // Dynamic window
            int reducedWindow = random.nextInt(windowSize) + 1;
            
            for (int j = pos - reducedWindow; j <= pos + reducedWindow; j++) {
                if (j < 0 || j >= doc.length || j == pos) continue;
                
                int contextIdx = doc[j];
                trainPair(wordIdx, contextIdx, true, lr);
                
                // Negative sampling
                for (int k = 0; k < negativeSamples; k++) {
                    int negIdx = sampleNegative(random);
                    if (negIdx != contextIdx) {
                        trainPair(wordIdx, negIdx, false, lr);
                    }
                }
            }
        }
    }
    
    private void trainPair(int wordIdx, int contextIdx, boolean positive, double lr) {
        double[] wordVec = wordVectors[wordIdx];
        double[] contextVec = contextVectors[contextIdx];
        
        // Calculate dot product
        double dot = 0;
        for (int i = 0; i < vectorSize; i++) {
            dot += wordVec[i] * contextVec[i];
        }
        
        // Sigmoid
        double sigmoid = 1.0 / (1.0 + Math.exp(-dot));
        double target = positive ? 1.0 : 0.0;
        double gradient = lr * (target - sigmoid);
        
        // Update vectors
        for (int i = 0; i < vectorSize; i++) {
            double temp = wordVec[i];
            wordVec[i] += gradient * contextVec[i];
            contextVec[i] += gradient * temp;
        }
    }
    
    private void buildSamplingTable() {
        double power = 0.75;
        double trainWordsPow = 0;
        
        for (int count : wordCounts) {
            trainWordsPow += Math.pow(count, power);
        }
        
        samplingTable = new double[wordCounts.length];
        double cumulative = 0;
        
        for (int i = 0; i < wordCounts.length; i++) {
            cumulative += Math.pow(wordCounts[i], power) / trainWordsPow;
            samplingTable[i] = cumulative;
        }
    }
    
    private int sampleNegative(Random random) {
        double r = random.nextDouble();
        int low = 0, high = samplingTable.length - 1;
        
        while (low < high) {
            int mid = (low + high) / 2;
            if (samplingTable[mid] < r) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        
        return low;
    }
    
    /**
     * Get the vector representation of a word.
     * @param word The word
     * @return Vector representation or null if word not in vocabulary
     */
    public double[] getVector(String word) {
        Integer idx = wordIndex.get(word.toLowerCase());
        if (idx == null) return null;
        return wordVectors[idx].clone();
    }
    
    /**
     * Find most similar words to a given word.
     * @param word The query word
     * @param topN Number of similar words to return
     * @return List of (word, similarity) pairs
     */
    public List<Map.Entry<String, Double>> mostSimilar(String word, int topN) {
        double[] queryVec = getVector(word);
        if (queryVec == null) return Collections.emptyList();
        
        Integer queryIdx = wordIndex.get(word.toLowerCase());
        PriorityQueue<Map.Entry<String, Double>> heap = new PriorityQueue<>(
            Comparator.comparingDouble(Map.Entry::getValue)
        );
        
        for (int i = 0; i < wordVectors.length; i++) {
            if (i == queryIdx) continue;
            
            double similarity = cosineSimilarity(queryVec, wordVectors[i]);
            heap.offer(new AbstractMap.SimpleEntry<>(indexToWord[i], similarity));
            
            if (heap.size() > topN) {
                heap.poll();
            }
        }
        
        List<Map.Entry<String, Double>> result = new ArrayList<>(heap);
        result.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        return result;
    }
    
    /**
     * Calculate similarity between two words.
     * @param word1 First word
     * @param word2 Second word
     * @return Cosine similarity or NaN if either word not found
     */
    public double similarity(String word1, String word2) {
        double[] vec1 = getVector(word1);
        double[] vec2 = getVector(word2);
        if (vec1 == null || vec2 == null) return Double.NaN;
        return cosineSimilarity(vec1, vec2);
    }
    
    private double cosineSimilarity(double[] a, double[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }
    
    // Getters
    public int getVocabularySize() { return wordIndex != null ? wordIndex.size() : 0; }
    public Set<String> getVocabulary() { return wordIndex != null ? new HashSet<>(wordIndex.keySet()) : null; }
    public int getVectorSize() { return vectorSize; }
    
    public static class Builder {
        private int vectorSize = 100;
        private int windowSize = 5;
        private int minCount = 5;
        private int negativeSamples = 5;
        private double learningRate = 0.025;
        private int epochs = 5;
        private long seed = 42;
        
        public Builder vectorSize(int size) { this.vectorSize = size; return this; }
        public Builder windowSize(int size) { this.windowSize = size; return this; }
        public Builder minCount(int count) { this.minCount = count; return this; }
        public Builder negativeSamples(int samples) { this.negativeSamples = samples; return this; }
        public Builder learningRate(double lr) { this.learningRate = lr; return this; }
        public Builder epochs(int epochs) { this.epochs = epochs; return this; }
        public Builder seed(long seed) { this.seed = seed; return this; }
        
        public Word2Vec build() { return new Word2Vec(this); }
    }
}
