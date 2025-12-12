package io.github.yasmramos.mindforge.model_selection;

import java.util.*;

/**
 * K-Fold cross-validation iterator.
 * Provides train/test indices to split data into train/test sets.
 */
public class KFold {
    
    private final int nSplits;
    private final boolean shuffle;
    private final Random random;
    
    public KFold(int nSplits) {
        this(nSplits, false, null);
    }
    
    public KFold(int nSplits, boolean shuffle, Random random) {
        if (nSplits < 2) {
            throw new IllegalArgumentException("nSplits must be at least 2");
        }
        this.nSplits = nSplits;
        this.shuffle = shuffle;
        this.random = random != null ? random : new Random();
    }
    
    public List<int[][]> split(int nSamples) {
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) indices[i] = i;
        
        if (shuffle) {
            for (int i = nSamples - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        
        List<int[][]> folds = new ArrayList<>();
        int foldSize = nSamples / nSplits;
        int remainder = nSamples % nSplits;
        
        int start = 0;
        for (int fold = 0; fold < nSplits; fold++) {
            int end = start + foldSize + (fold < remainder ? 1 : 0);
            
            int[] testIdx = Arrays.copyOfRange(indices, start, end);
            int[] trainIdx = new int[nSamples - testIdx.length];
            
            int trainPos = 0;
            for (int i = 0; i < start; i++) trainIdx[trainPos++] = indices[i];
            for (int i = end; i < nSamples; i++) trainIdx[trainPos++] = indices[i];
            
            folds.add(new int[][]{trainIdx, testIdx});
            start = end;
        }
        
        return folds;
    }
    
    public List<int[][]> split(double[][] X) {
        return split(X.length);
    }
    
    public int getNSplits() { return nSplits; }
}
