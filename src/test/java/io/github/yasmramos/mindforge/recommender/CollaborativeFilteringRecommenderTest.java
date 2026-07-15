package io.github.yasmramos.mindforge.recommender;

import io.github.yasmramos.mindforge.data.Dataset;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

/**
 * Unit tests for CollaborativeFilteringRecommender.
 */
public class CollaborativeFilteringRecommenderTest {
    
    @Test
    public void testFitAndPredict() {
        double[][] ratings = {
            {5.0, 3.0, 0.0, 4.0},
            {4.0, 0.0, 5.0, 3.0},
            {3.0, 4.0, 4.0, 0.0},
            {0.0, 5.0, 3.0, 5.0}
        };
        int[] users = {0, 1, 2, 3};
        Dataset dataset = new Dataset(ratings, users);
        
        CollaborativeFilteringRecommender recommender = new CollaborativeFilteringRecommender("cosine");
        recommender.fit(dataset);
        
        double prediction = recommender.predict(0, 2);
        assertTrue(prediction > 0.0, "Prediction should be positive");
        assertTrue(prediction <= 5.0, "Prediction should be within rating range");
    }
    
    @Test
    public void testRecommend() {
        double[][] ratings = {
            {5.0, 3.0, 0.0, 4.0, 0.0},
            {4.0, 0.0, 5.0, 3.0, 4.0},
            {3.0, 4.0, 4.0, 0.0, 5.0},
            {0.0, 5.0, 3.0, 5.0, 3.0}
        };
        int[] users = {0, 1, 2, 3};
        Dataset dataset = new Dataset(ratings, users);
        
        CollaborativeFilteringRecommender recommender = new CollaborativeFilteringRecommender("pearson");
        recommender.fit(dataset);
        
        List<Integer> recommendations = recommender.recommend(0, 2);
        assertEquals(2, recommendations.size(), "Should recommend 2 items");
        assertFalse(recommendations.contains(0), "Should not recommend already rated items");
        assertFalse(recommendations.contains(1), "Should not recommend already rated items");
        assertFalse(recommendations.contains(3), "Should not recommend already rated items");
    }
    
    @Test
    public void testDifferentSimilarityMetrics() {
        double[][] ratings = {
            {5.0, 3.0, 0.0, 4.0},
            {4.0, 0.0, 5.0, 3.0},
            {3.0, 4.0, 4.0, 0.0}
        };
        int[] users = {0, 1, 2};
        Dataset dataset = new Dataset(ratings, users);
        
        String[] metrics = {"cosine", "pearson", "euclidean"};
        for (String metric : metrics) {
            CollaborativeFilteringRecommender recommender = new CollaborativeFilteringRecommender(metric);
            recommender.fit(dataset);
            double prediction = recommender.predict(0, 2);
            assertTrue(prediction >= 0.0, "Prediction with " + metric + " should be non-negative");
        }
    }
}
