package io.github.yasmramos.mindforge.model_selection;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class KFoldTest {
    
    @Test
    void testBasicKFold() {
        KFold kf = new KFold(5);
        List<int[][]> splits = kf.split(100);
        
        assertEquals(5, splits.size());
        
        Set<Integer> allTest = new HashSet<>();
        for (int[][] split : splits) {
            int[] train = split[0];
            int[] test = split[1];
            
            assertEquals(80, train.length);
            assertEquals(20, test.length);
            
            for (int idx : test) allTest.add(idx);
        }
        assertEquals(100, allTest.size());
    }
    
    @Test
    void testShuffledKFold() {
        KFold kf = new KFold(3, true, new Random(42));
        List<int[][]> splits = kf.split(30);
        
        assertEquals(3, splits.size());
        
        // Check no overlap in test sets
        Set<Integer> allTest = new HashSet<>();
        for (int[][] split : splits) {
            for (int idx : split[1]) {
                assertFalse(allTest.contains(idx));
                allTest.add(idx);
            }
        }
    }
    
    @Test
    void testUnevenSplit() {
        KFold kf = new KFold(3);
        List<int[][]> splits = kf.split(10);
        
        int totalTest = 0;
        for (int[][] split : splits) {
            totalTest += split[1].length;
        }
        assertEquals(10, totalTest);
    }
}
