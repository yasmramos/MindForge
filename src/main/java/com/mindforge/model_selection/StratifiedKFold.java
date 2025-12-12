package com.mindforge.model_selection;

import java.util.*;

/**
 * Stratified K-Fold cross-validation.
 * Preserves the percentage of samples for each class.
 */
public class StratifiedKFold {
    
    private final int nSplits;
    private final boolean shuffle;
    private final Random random;
    
    public StratifiedKFold(int nSplits) {
        this(nSplits, false, null);
    }
    
    public StratifiedKFold(int nSplits, boolean shuffle, Random random) {
        if (nSplits < 2) {
            throw new IllegalArgumentException("nSplits must be at least 2");
        }
        this.nSplits = nSplits;
        this.shuffle = shuffle;
        this.random = random != null ? random : new Random();
    }
    
    public List<int[][]> split(double[][] X, int[] y) {
        return split(y);
    }
    
    public List<int[][]> split(int[] y) {
        int n = y.length;
        
        // Group indices by class
        Map<Integer, List<Integer>> classIndices = new HashMap<>();
        for (int i = 0; i < n; i++) {
            classIndices.computeIfAbsent(y[i], k -> new ArrayList<>()).add(i);
        }
        
        // Shuffle within each class if needed
        if (shuffle) {
            for (List<Integer> indices : classIndices.values()) {
                Collections.shuffle(indices, random);
            }
        }
        
        // Initialize fold assignments
        List<List<Integer>> foldIndices = new ArrayList<>();
        for (int i = 0; i < nSplits; i++) {
            foldIndices.add(new ArrayList<>());
        }
        
        // Distribute samples from each class across folds
        for (List<Integer> indices : classIndices.values()) {
            int foldIdx = 0;
            for (int idx : indices) {
                foldIndices.get(foldIdx).add(idx);
                foldIdx = (foldIdx + 1) % nSplits;
            }
        }
        
        // Generate train/test splits
        List<int[][]> splits = new ArrayList<>();
        for (int fold = 0; fold < nSplits; fold++) {
            List<Integer> testList = foldIndices.get(fold);
            List<Integer> trainList = new ArrayList<>();
            
            for (int i = 0; i < nSplits; i++) {
                if (i != fold) {
                    trainList.addAll(foldIndices.get(i));
                }
            }
            
            int[] trainIdx = trainList.stream().mapToInt(Integer::intValue).toArray();
            int[] testIdx = testList.stream().mapToInt(Integer::intValue).toArray();
            
            splits.add(new int[][]{trainIdx, testIdx});
        }
        
        return splits;
    }
    
    public int getNSplits() { return nSplits; }
}
