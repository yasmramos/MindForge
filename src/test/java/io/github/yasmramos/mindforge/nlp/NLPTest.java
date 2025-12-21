package io.github.yasmramos.mindforge.nlp;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Tests for NLP classes.
 */
public class NLPTest {
    
    @Test
    public void testCountVectorizerFitTransform() {
        String[] docs = {
            "the cat sat on the mat",
            "the dog sat on the log",
            "the cat and the dog"
        };
        
        CountVectorizer cv = new CountVectorizer.Builder().build();
        double[][] result = cv.fitTransform(docs);
        
        assertNotNull(result);
        assertEquals(3, result.length);
        assertTrue(cv.getVocabularySize() > 0);
    }
    
    @Test
    public void testCountVectorizerVocabulary() {
        String[] docs = {"hello world", "hello java"};
        
        CountVectorizer cv = new CountVectorizer.Builder().build();
        cv.fit(docs);
        
        Map<String, Integer> vocab = cv.getVocabulary();
        assertNotNull(vocab);
        assertTrue(vocab.containsKey("hello"));
        assertTrue(vocab.containsKey("world"));
        assertTrue(vocab.containsKey("java"));
    }
    
    @Test
    public void testCountVectorizerBinary() {
        String[] docs = {"hello hello hello world"};
        
        CountVectorizer cv = new CountVectorizer.Builder()
            .binary(true)
            .build();
        double[][] result = cv.fitTransform(docs);
        
        // With binary=true, all non-zero values should be 1
        for (double val : result[0]) {
            assertTrue(val == 0.0 || val == 1.0);
        }
    }
    
    @Test
    public void testCountVectorizerStopWords() {
        String[] docs = {"the quick brown fox"};
        
        CountVectorizer cv = new CountVectorizer.Builder()
            .stopWords(CountVectorizer.getEnglishStopWords())
            .build();
        cv.fit(docs);
        
        Map<String, Integer> vocab = cv.getVocabulary();
        assertFalse(vocab.containsKey("the"));
        assertTrue(vocab.containsKey("quick"));
    }
    
    @Test
    public void testCountVectorizerNgrams() {
        String[] docs = {"hello world"};
        
        CountVectorizer cv = new CountVectorizer.Builder()
            .ngramRange(1, 2)
            .build();
        cv.fit(docs);
        
        Map<String, Integer> vocab = cv.getVocabulary();
        assertTrue(vocab.containsKey("hello"));
        assertTrue(vocab.containsKey("world"));
        assertTrue(vocab.containsKey("hello world"));
    }
    
    @Test
    public void testCountVectorizerMaxFeatures() {
        String[] docs = {"one two three four five six seven"};
        
        CountVectorizer cv = new CountVectorizer.Builder()
            .maxFeatures(3)
            .build();
        cv.fit(docs);
        
        assertEquals(3, cv.getVocabularySize());
    }
    
    @Test
    public void testTfidfVectorizerFitTransform() {
        String[] docs = {
            "this is the first document",
            "this document is the second document",
            "and this is the third one"
        };
        
        TfidfVectorizer tfidf = new TfidfVectorizer.Builder().build();
        double[][] result = tfidf.fitTransform(docs);
        
        assertNotNull(result);
        assertEquals(3, result.length);
        assertTrue(tfidf.getVocabularySize() > 0);
    }
    
    @Test
    public void testTfidfVectorizerNormalization() {
        String[] docs = {"hello world foo bar"};
        
        TfidfVectorizer tfidf = new TfidfVectorizer.Builder()
            .norm(TfidfVectorizer.Norm.L2)
            .build();
        double[][] result = tfidf.fitTransform(docs);
        
        // Check L2 norm equals 1
        double norm = 0;
        for (double val : result[0]) {
            norm += val * val;
        }
        assertEquals(1.0, Math.sqrt(norm), 0.001);
    }
    
    @Test
    public void testTfidfVectorizerNoNorm() {
        String[] docs = {"hello world"};
        
        TfidfVectorizer tfidf = new TfidfVectorizer.Builder()
            .norm(TfidfVectorizer.Norm.NONE)
            .build();
        double[][] result = tfidf.fitTransform(docs);
        
        // Without normalization, values can be > 1
        assertNotNull(result);
    }
    
    @Test
    public void testTfidfVectorizerIdfWeights() {
        String[] docs = {
            "common rare",
            "common word",
            "common term"
        };
        
        TfidfVectorizer tfidf = new TfidfVectorizer.Builder()
            .norm(TfidfVectorizer.Norm.NONE)
            .build();
        tfidf.fit(docs);
        
        double[] idf = tfidf.getIdfWeights();
        assertNotNull(idf);
        
        // "common" appears in all docs, should have lower IDF
        Map<String, Integer> vocab = tfidf.getVocabulary();
        int commonIdx = vocab.get("common");
        int rareIdx = vocab.get("rare");
        
        assertTrue(idf[rareIdx] > idf[commonIdx]);
    }
    
    @Test
    public void testTfidfVectorizerSublinearTf() {
        String[] docs = {"word word word word"};
        
        TfidfVectorizer tfidf = new TfidfVectorizer.Builder()
            .sublinearTf(true)
            .norm(TfidfVectorizer.Norm.NONE)
            .build();
        double[][] result = tfidf.fitTransform(docs);
        
        // With sublinear TF, tf = 1 + log(count)
        assertNotNull(result);
    }
    
    @Test
    public void testWord2VecFit() {
        String[] docs = {
            "the king loves the queen",
            "the queen loves the king",
            "the prince loves the princess",
            "the princess loves the prince",
            "king and queen rule the kingdom",
            "prince and princess live in the castle"
        };
        
        Word2Vec w2v = new Word2Vec.Builder()
            .vectorSize(10)
            .windowSize(2)
            .minCount(1)
            .epochs(10)
            .build();
        
        w2v.fit(docs);
        
        assertTrue(w2v.getVocabularySize() > 0);
        assertTrue(w2v.getVocabulary().contains("king"));
    }
    
    @Test
    public void testWord2VecGetVector() {
        String[] docs = {"hello world hello java world"};
        
        Word2Vec w2v = new Word2Vec.Builder()
            .vectorSize(5)
            .minCount(1)
            .epochs(5)
            .build();
        
        w2v.fit(docs);
        
        double[] vec = w2v.getVector("hello");
        assertNotNull(vec);
        assertEquals(5, vec.length);
        
        // Unknown word should return null
        assertNull(w2v.getVector("unknown"));
    }
    
    @Test
    public void testWord2VecSimilarity() {
        String[] docs = {
            "cat dog pet animal",
            "cat pet cute",
            "dog pet friendly",
            "car vehicle drive",
            "car vehicle road"
        };
        
        Word2Vec w2v = new Word2Vec.Builder()
            .vectorSize(20)
            .windowSize(3)
            .minCount(1)
            .epochs(50)
            .build();
        
        w2v.fit(docs);
        
        double simCatDog = w2v.similarity("cat", "dog");
        assertFalse(Double.isNaN(simCatDog));
        
        // Both words should have vectors
        assertNotNull(w2v.getVector("cat"));
        assertNotNull(w2v.getVector("dog"));
    }
    
    @Test
    public void testWord2VecMostSimilar() {
        String[] docs = {
            "apple banana fruit orange",
            "apple fruit sweet",
            "banana fruit yellow",
            "orange fruit citrus"
        };
        
        Word2Vec w2v = new Word2Vec.Builder()
            .vectorSize(10)
            .minCount(1)
            .epochs(20)
            .build();
        
        w2v.fit(docs);
        
        List<Map.Entry<String, Double>> similar = w2v.mostSimilar("apple", 2);
        assertNotNull(similar);
        assertEquals(2, similar.size());
    }
    
    @Test
    public void testWord2VecUnknownWord() {
        String[] docs = {"hello world"};
        
        Word2Vec w2v = new Word2Vec.Builder()
            .minCount(1)
            .build();
        
        w2v.fit(docs);
        
        double sim = w2v.similarity("hello", "unknown");
        assertTrue(Double.isNaN(sim));
        
        List<Map.Entry<String, Double>> similar = w2v.mostSimilar("unknown", 5);
        assertTrue(similar.isEmpty());
    }
}
