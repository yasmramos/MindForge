package io.github.yasmramos.mindforge.model_selection;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class StratifiedKFoldTest {
    
    @Test
    void testStratification() {
        int[] y = new int[100];
        for (int i = 0; i < 70; i++) y[i] = 0;
        for (int i = 70; i < 100; i++) y[i] = 1;
        
        StratifiedKFold skf = new StratifiedKFold(5);
        List<int[][]> splits = skf.split(y);
        
        assertEquals(5, splits.size());
        
        for (int[][] split : splits) {
            int[] testIdx = split[1];
            int class0 = 0, class1 = 0;
            for (int idx : testIdx) {
                if (y[idx] == 0) class0++;
                else class1++;
            }
            // Should maintain roughly 70/30 ratio
            assertTrue(class0 >= 10 && class0 <= 18);
            assertTrue(class1 >= 4 && class1 <= 10);
        }
    }
    
    @Test
    void testNoDuplicates() {
        int[] y = {0, 0, 0, 1, 1, 1, 2, 2, 2};
        StratifiedKFold skf = new StratifiedKFold(3);
        List<int[][]> splits = skf.split(y);
        
        Set<Integer> allTest = new HashSet<>();
        for (int[][] split : splits) {
            for (int idx : split[1]) {
                assertFalse(allTest.contains(idx));
                allTest.add(idx);
            }
        }
        assertEquals(9, allTest.size());
    }
}
