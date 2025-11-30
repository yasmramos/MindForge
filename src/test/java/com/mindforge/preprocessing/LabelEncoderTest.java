package com.mindforge.preprocessing;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class LabelEncoderTest {

    @Test
    void testBasicEncoding() {
        String[] labels = {"cat", "dog", "cat", "bird", "dog"};
        
        LabelEncoder encoder = new LabelEncoder();
        int[] encoded = encoder.fitTransform(labels);

        // "cat" -> 0, "dog" -> 1, "bird" -> 2
        assertArrayEquals(new int[]{0, 1, 0, 2, 1}, encoded);
    }

    @Test
    void testInverseTransform() {
        String[] labels = {"red", "green", "blue", "red", "blue"};
        
        LabelEncoder encoder = new LabelEncoder();
        int[] encoded = encoder.fitTransform(labels);
        String[] decoded = encoder.inverseTransform(encoded);

        assertArrayEquals(labels, decoded);
    }

    @Test
    void testGetClasses() {
        String[] labels = {"apple", "banana", "apple", "cherry", "banana"};
        
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(labels);
        String[] classes = encoder.getClasses();

        // Should return unique classes in order of first appearance
        assertArrayEquals(new String[]{"apple", "banana", "cherry"}, classes);
    }

    @Test
    void testGetNumClasses() {
        String[] labels = {"A", "B", "C", "A", "B", "C"};
        
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(labels);

        assertEquals(3, encoder.getNumClasses());
    }

    @Test
    void testEncodeDecodeIndividual() {
        String[] labels = {"cat", "dog", "bird"};
        
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(labels);

        assertEquals(0, encoder.encode("cat"));
        assertEquals(1, encoder.encode("dog"));
        assertEquals(2, encoder.encode("bird"));

        assertEquals("cat", encoder.decode(0));
        assertEquals("dog", encoder.decode(1));
        assertEquals("bird", encoder.decode(2));
    }

    @Test
    void testFitAndTransformSeparately() {
        String[] trainLabels = {"cat", "dog", "bird"};
        String[] testLabels = {"dog", "cat", "bird", "dog"};
        
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(trainLabels);
        int[] encoded = encoder.transform(testLabels);

        assertArrayEquals(new int[]{1, 0, 2, 1}, encoded);
    }

    @Test
    void testUnseenLabelError() {
        String[] trainLabels = {"cat", "dog"};
        String[] testLabels = {"cat", "bird"};  // "bird" is unseen
        
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(trainLabels);

        assertThrows(IllegalArgumentException.class, () -> encoder.transform(testLabels));
    }

    @Test
    void testUnseenEncodedValueError() {
        String[] labels = {"cat", "dog"};
        
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(labels);

        int[] invalidEncoded = {0, 1, 5};  // 5 is invalid
        assertThrows(IllegalArgumentException.class, () -> encoder.inverseTransform(invalidEncoded));
    }

    @Test
    void testNotFittedError() {
        LabelEncoder encoder = new LabelEncoder();
        String[] labels = {"cat", "dog"};
        int[] encoded = {0, 1};

        assertThrows(IllegalStateException.class, () -> encoder.transform(labels));
        assertThrows(IllegalStateException.class, () -> encoder.inverseTransform(encoded));
        assertThrows(IllegalStateException.class, () -> encoder.getClasses());
        assertThrows(IllegalStateException.class, () -> encoder.getNumClasses());
        assertThrows(IllegalStateException.class, () -> encoder.encode("cat"));
        assertThrows(IllegalStateException.class, () -> encoder.decode(0));
    }

    @Test
    void testNullLabelsError() {
        LabelEncoder encoder = new LabelEncoder();
        String[] labels = {"cat", null, "dog"};

        assertThrows(IllegalArgumentException.class, () -> encoder.fit(labels));
    }

    @Test
    void testEmptyLabelsError() {
        LabelEncoder encoder = new LabelEncoder();
        String[] labels = {};

        assertThrows(IllegalArgumentException.class, () -> encoder.fit(labels));
    }

    @Test
    void testIsFitted() {
        LabelEncoder encoder = new LabelEncoder();
        assertFalse(encoder.isFitted());

        String[] labels = {"cat", "dog"};
        encoder.fit(labels);
        assertTrue(encoder.isFitted());
    }

    @Test
    void testOrderPreserved() {
        // Test that the order of first appearance is preserved
        String[] labels = {"zebra", "apple", "monkey", "apple", "zebra"};
        
        LabelEncoder encoder = new LabelEncoder();
        encoder.fit(labels);

        // Order should be: zebra (0), apple (1), monkey (2)
        assertEquals(0, encoder.encode("zebra"));
        assertEquals(1, encoder.encode("apple"));
        assertEquals(2, encoder.encode("monkey"));
    }

    @Test
    void testSingleClass() {
        String[] labels = {"cat", "cat", "cat"};
        
        LabelEncoder encoder = new LabelEncoder();
        int[] encoded = encoder.fitTransform(labels);

        assertArrayEquals(new int[]{0, 0, 0}, encoded);
        assertEquals(1, encoder.getNumClasses());
    }
}
