package com.mindforge.preprocessing;

import java.util.*;

/**
 * LabelEncoder encodes categorical labels into numeric values.
 * 
 * Each unique label is assigned an integer value from 0 to n_classes - 1.
 * 
 * Example:
 * <pre>
 * String[] labels = {"cat", "dog", "cat", "bird", "dog"};
 * LabelEncoder encoder = new LabelEncoder();
 * int[] encoded = encoder.fitTransform(labels);
 * // Result: [0, 1, 0, 2, 1]
 * String[] decoded = encoder.inverseTransform(encoded);
 * // Result: ["cat", "dog", "cat", "bird", "dog"]
 * </pre>
 */
public class LabelEncoder {
    private Map<String, Integer> labelToIndex;
    private Map<Integer, String> indexToLabel;
    private boolean fitted;

    /**
     * Creates a new LabelEncoder.
     */
    public LabelEncoder() {
        this.labelToIndex = new LinkedHashMap<>();
        this.indexToLabel = new HashMap<>();
        this.fitted = false;
    }

    /**
     * Fits the encoder by learning the mapping from labels to indices.
     * 
     * @param labels array of string labels
     */
    public void fit(String[] labels) {
        if (labels == null || labels.length == 0) {
            throw new IllegalArgumentException("Labels cannot be null or empty");
        }

        labelToIndex.clear();
        indexToLabel.clear();

        int currentIndex = 0;
        for (String label : labels) {
            if (label == null) {
                throw new IllegalArgumentException("Labels cannot contain null values");
            }
            if (!labelToIndex.containsKey(label)) {
                labelToIndex.put(label, currentIndex);
                indexToLabel.put(currentIndex, label);
                currentIndex++;
            }
        }

        this.fitted = true;
    }

    /**
     * Transforms labels to their encoded integer representations.
     * 
     * @param labels array of string labels
     * @return array of encoded integer values
     */
    public int[] transform(String[] labels) {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before transformation");
        }
        if (labels == null || labels.length == 0) {
            throw new IllegalArgumentException("Labels cannot be null or empty");
        }

        int[] encoded = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            if (!labelToIndex.containsKey(labels[i])) {
                throw new IllegalArgumentException(
                    "Label '" + labels[i] + "' not found in fitted labels"
                );
            }
            encoded[i] = labelToIndex.get(labels[i]);
        }

        return encoded;
    }

    /**
     * Fits the encoder and transforms the labels in one step.
     * 
     * @param labels array of string labels
     * @return array of encoded integer values
     */
    public int[] fitTransform(String[] labels) {
        fit(labels);
        return transform(labels);
    }

    /**
     * Transforms encoded integer values back to their original labels.
     * 
     * @param encoded array of encoded integer values
     * @return array of original string labels
     */
    public String[] inverseTransform(int[] encoded) {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before inverse transformation");
        }
        if (encoded == null || encoded.length == 0) {
            throw new IllegalArgumentException("Encoded values cannot be null or empty");
        }

        String[] labels = new String[encoded.length];
        for (int i = 0; i < encoded.length; i++) {
            if (!indexToLabel.containsKey(encoded[i])) {
                throw new IllegalArgumentException(
                    "Encoded value " + encoded[i] + " not found in fitted encoder"
                );
            }
            labels[i] = indexToLabel.get(encoded[i]);
        }

        return labels;
    }

    /**
     * Gets all unique classes (labels) that were fitted.
     * 
     * @return array of unique class labels
     */
    public String[] getClasses() {
        if (!fitted) {
            throw new IllegalStateException("Encoder has not been fitted yet");
        }
        return labelToIndex.keySet().toArray(new String[0]);
    }

    /**
     * Gets the number of unique classes.
     * 
     * @return number of classes
     */
    public int getNumClasses() {
        if (!fitted) {
            throw new IllegalStateException("Encoder has not been fitted yet");
        }
        return labelToIndex.size();
    }

    /**
     * Gets the encoded value for a specific label.
     * 
     * @param label the label to encode
     * @return the encoded integer value
     */
    public int encode(String label) {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before encoding");
        }
        if (!labelToIndex.containsKey(label)) {
            throw new IllegalArgumentException("Label '" + label + "' not found in fitted labels");
        }
        return labelToIndex.get(label);
    }

    /**
     * Gets the original label for a specific encoded value.
     * 
     * @param encoded the encoded integer value
     * @return the original label
     */
    public String decode(int encoded) {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before decoding");
        }
        if (!indexToLabel.containsKey(encoded)) {
            throw new IllegalArgumentException("Encoded value " + encoded + " not found in fitted encoder");
        }
        return indexToLabel.get(encoded);
    }

    /**
     * Checks if the encoder has been fitted.
     * 
     * @return true if fitted, false otherwise
     */
    public boolean isFitted() {
        return fitted;
    }
}
