package io.github.yasmramos.mindforge.recommender;

import io.github.yasmramos.mindforge.data.Dataset;

import java.util.*;

/**
 * Collaborative Filtering Recommender using User-Based approach.
 * Predicts user preferences based on similarity with other users.
 */
public class CollaborativeFilteringRecommender {
    
    private double[][] userItemMatrix;
    private Map<Integer, Integer> userIdxMap;
    private Map<Integer, Integer> itemIdxMap;
    private Map<Integer, Integer> reverseUserIdxMap;
    private Map<Integer, Integer> reverseItemIdxMap;
    private String similarityMetric;
    
    public CollaborativeFilteringRecommender(String similarityMetric) {
        this.similarityMetric = similarityMetric;
        this.userIdxMap = new HashMap<>();
        this.itemIdxMap = new HashMap<>();
        this.reverseUserIdxMap = new HashMap<>();
        this.reverseItemIdxMap = new HashMap<>();
    }
    
    public void fit(Dataset dataset) {
        int nUsers = dataset.getLabels().length;
        double[][] features = dataset.getFeatures();
        Set<Integer> allItems = new HashSet<>();
        
        for (int i = 0; i < nUsers; i++) {
            userIdxMap.put(i, i);
            reverseUserIdxMap.put(i, i);
            double[] ratings = features[i];
            for (int j = 0; j < ratings.length; j++) {
                if (ratings[j] != 0) {
                    allItems.add(j);
                }
            }
        }
        
        List<Integer> itemList = new ArrayList<>(allItems);
        Collections.sort(itemList);
        for (int i = 0; i < itemList.size(); i++) {
            itemIdxMap.put(itemList.get(i), i);
            reverseItemIdxMap.put(i, itemList.get(i));
        }
        
        int nItems = itemList.size();
        userItemMatrix = new double[nUsers][nItems];
        
        for (int i = 0; i < nUsers; i++) {
            double[] ratings = features[i];
            for (int j = 0; j < ratings.length; j++) {
                if (ratings[j] != 0 && itemIdxMap.containsKey(j)) {
                    userItemMatrix[i][itemIdxMap.get(j)] = ratings[j];
                }
            }
        }
    }
    
    public double predict(int userId, int itemId) {
        if (!userIdxMap.containsKey(userId) || !itemIdxMap.containsKey(itemId)) {
            return 0.0;
        }
        
        int uIdx = userIdxMap.get(userId);
        int iIdx = itemIdxMap.get(itemId);
        
        if (userItemMatrix[uIdx][iIdx] != 0) {
            return userItemMatrix[uIdx][iIdx];
        }
        
        Map<Integer, Double> similarities = new HashMap<>();
        for (int otherU = 0; otherU < userItemMatrix.length; otherU++) {
            if (otherU != uIdx && userItemMatrix[otherU][iIdx] != 0) {
                double sim = computeSimilarity(uIdx, otherU);
                if (sim > 0) {
                    similarities.put(otherU, sim);
                }
            }
        }
        
        if (similarities.isEmpty()) {
            return 0.0;
        }
        
        double numerator = 0.0;
        double denominator = 0.0;
        
        for (Map.Entry<Integer, Double> entry : similarities.entrySet()) {
            int otherU = entry.getKey();
            double sim = entry.getValue();
            double rating = userItemMatrix[otherU][iIdx];
            numerator += sim * rating;
            denominator += Math.abs(sim);
        }
        
        return denominator > 0 ? numerator / denominator : 0.0;
    }
    
    public List<Integer> recommend(int userId, int topK) {
        if (!userIdxMap.containsKey(userId)) {
            return new ArrayList<>();
        }
        
        int uIdx = userIdxMap.get(userId);
        List<int[]> predictions = new ArrayList<>();
        
        for (int iIdx = 0; iIdx < userItemMatrix[0].length; iIdx++) {
            if (userItemMatrix[uIdx][iIdx] == 0) {
                int itemId = reverseItemIdxMap.get(iIdx);
                double pred = predict(userId, itemId);
                predictions.add(new int[]{itemId, (int)(pred * 1000)});
            }
        }
        
        predictions.sort((a, b) -> Double.compare(b[1], a[1]));
        
        List<Integer> recommendations = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, predictions.size()); i++) {
            recommendations.add(predictions.get(i)[0]);
        }
        
        return recommendations;
    }
    
    private double computeSimilarity(int u1, int u2) {
        switch (similarityMetric.toLowerCase()) {
            case "cosine":
                return cosineSimilarity(u1, u2);
            case "pearson":
                return pearsonSimilarity(u1, u2);
            case "euclidean":
                return 1.0 / (1.0 + euclideanDistance(u1, u2));
            default:
                return cosineSimilarity(u1, u2);
        }
    }
    
    private double cosineSimilarity(int u1, int u2) {
        double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
        for (int i = 0; i < userItemMatrix[0].length; i++) {
            if (userItemMatrix[u1][i] != 0 && userItemMatrix[u2][i] != 0) {
                dot += userItemMatrix[u1][i] * userItemMatrix[u2][i];
                norm1 += userItemMatrix[u1][i] * userItemMatrix[u1][i];
                norm2 += userItemMatrix[u2][i] * userItemMatrix[u2][i];
            }
        }
        return (norm1 > 0 && norm2 > 0) ? dot / (Math.sqrt(norm1) * Math.sqrt(norm2)) : 0.0;
    }
    
    private double pearsonSimilarity(int u1, int u2) {
        List<Double> common1 = new ArrayList<>();
        List<Double> common2 = new ArrayList<>();
        
        for (int i = 0; i < userItemMatrix[0].length; i++) {
            if (userItemMatrix[u1][i] != 0 && userItemMatrix[u2][i] != 0) {
                common1.add(userItemMatrix[u1][i]);
                common2.add(userItemMatrix[u2][i]);
            }
        }
        
        if (common1.size() < 2) return 0.0;
        
        double mean1 = common1.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double mean2 = common2.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        double num = 0.0, den1 = 0.0, den2 = 0.0;
        for (int i = 0; i < common1.size(); i++) {
            double diff1 = common1.get(i) - mean1;
            double diff2 = common2.get(i) - mean2;
            num += diff1 * diff2;
            den1 += diff1 * diff1;
            den2 += diff2 * diff2;
        }
        
        return (den1 > 0 && den2 > 0) ? num / (Math.sqrt(den1) * Math.sqrt(den2)) : 0.0;
    }
    
    private double euclideanDistance(int u1, int u2) {
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < userItemMatrix[0].length; i++) {
            if (userItemMatrix[u1][i] != 0 && userItemMatrix[u2][i] != 0) {
                sum += Math.pow(userItemMatrix[u1][i] - userItemMatrix[u2][i], 2);
                count++;
            }
        }
        return count > 0 ? Math.sqrt(sum / count) : Double.MAX_VALUE;
    }
}
