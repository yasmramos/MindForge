package io.github.yasmramos.mindforge.recommender;

import io.github.yasmramos.mindforge.data.Dataset;

import java.util.*;

/**
 * Item-Based Collaborative Filtering Recommender.
 * Predicts user preferences based on similarity between items.
 */
public class ItemBasedRecommender {
    
    private double[][] userItemMatrix;
    private double[][] itemSimilarityMatrix;
    private Map<Integer, Integer> userIdxMap;
    private Map<Integer, Integer> itemIdxMap;
    private Map<Integer, Integer> reverseUserIdxMap;
    private Map<Integer, Integer> reverseItemIdxMap;
    private String similarityMetric;
    
    public ItemBasedRecommender(String similarityMetric) {
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
        
        computeItemSimilarities();
    }
    
    private void computeItemSimilarities() {
        int nItems = userItemMatrix[0].length;
        itemSimilarityMatrix = new double[nItems][nItems];
        
        for (int i = 0; i < nItems; i++) {
            for (int j = i; j < nItems; j++) {
                if (i == j) {
                    itemSimilarityMatrix[i][j] = 1.0;
                } else {
                    double sim = computeSimilarity(i, j);
                    itemSimilarityMatrix[i][j] = sim;
                    itemSimilarityMatrix[j][i] = sim;
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
        
        List<int[]> similarItems = new ArrayList<>();
        for (int otherI = 0; otherI < itemSimilarityMatrix.length; otherI++) {
            if (otherI != iIdx && userItemMatrix[uIdx][otherI] != 0) {
                double sim = itemSimilarityMatrix[iIdx][otherI];
                if (sim > 0) {
                    similarItems.add(new int[]{otherI, (int)(sim * 1000)});
                }
            }
        }
        
        if (similarItems.isEmpty()) {
            return 0.0;
        }
        
        similarItems.sort((a, b) -> Double.compare(b[1], a[1]));
        
        double numerator = 0.0;
        double denominator = 0.0;
        int count = 0;
        
        for (int[] item : similarItems) {
            if (count >= 20) break;
            int otherI = item[0];
            double sim = item[1] / 1000.0;
            double rating = userItemMatrix[uIdx][otherI];
            numerator += sim * rating;
            denominator += Math.abs(sim);
            count++;
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
    
    private double computeSimilarity(int i1, int i2) {
        switch (similarityMetric.toLowerCase()) {
            case "cosine":
                return cosineSimilarity(i1, i2);
            case "pearson":
                return pearsonSimilarity(i1, i2);
            default:
                return cosineSimilarity(i1, i2);
        }
    }
    
    private double cosineSimilarity(int i1, int i2) {
        double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
        for (int u = 0; u < userItemMatrix.length; u++) {
            if (userItemMatrix[u][i1] != 0 && userItemMatrix[u][i2] != 0) {
                dot += userItemMatrix[u][i1] * userItemMatrix[u][i2];
                norm1 += userItemMatrix[u][i1] * userItemMatrix[u][i1];
                norm2 += userItemMatrix[u][i2] * userItemMatrix[u][i2];
            }
        }
        return (norm1 > 0 && norm2 > 0) ? dot / (Math.sqrt(norm1) * Math.sqrt(norm2)) : 0.0;
    }
    
    private double pearsonSimilarity(int i1, int i2) {
        List<Double> common1 = new ArrayList<>();
        List<Double> common2 = new ArrayList<>();
        
        for (int u = 0; u < userItemMatrix.length; u++) {
            if (userItemMatrix[u][i1] != 0 && userItemMatrix[u][i2] != 0) {
                common1.add(userItemMatrix[u][i1]);
                common2.add(userItemMatrix[u][i2]);
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
}
