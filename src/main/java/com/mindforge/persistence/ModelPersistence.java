package com.mindforge.persistence;

import java.io.*;

/**
 * Utility class for saving and loading machine learning models.
 * 
 * This class provides static methods to persist any serializable model
 * to disk and load it back. Models are saved using Java's native serialization.
 * 
 * Example usage:
 * <pre>
 * // Save a trained model
 * DecisionTreeClassifier model = new DecisionTreeClassifier();
 * model.fit(X, y);
 * ModelPersistence.save(model, "model.bin");
 * 
 * // Load the model
 * DecisionTreeClassifier loaded = ModelPersistence.load("model.bin");
 * int[] predictions = loaded.predict(X_test);
 * </pre>
 * 
 * @author MindForge
 * @version 1.0.8-alpha
 */
public class ModelPersistence {
    
    private static final String MAGIC_HEADER = "MINDFORGE_MODEL_V1";
    
    /**
     * Saves a model to a file.
     * 
     * @param model The model to save (must be Serializable)
     * @param filePath Path where the model will be saved
     * @throws ModelPersistenceException if saving fails
     */
    public static void save(Serializable model, String filePath) {
        if (model == null) {
            throw new IllegalArgumentException("Model cannot be null");
        }
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty");
        }
        
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(filePath)))) {
            // Write magic header for validation
            oos.writeUTF(MAGIC_HEADER);
            // Write model class name
            oos.writeUTF(model.getClass().getName());
            // Write timestamp
            oos.writeLong(System.currentTimeMillis());
            // Write the model
            oos.writeObject(model);
        } catch (IOException e) {
            throw new ModelPersistenceException("Failed to save model to: " + filePath, e);
        }
    }
    
    /**
     * Loads a model from a file.
     * 
     * @param <T> The type of the model
     * @param filePath Path to the saved model file
     * @return The loaded model
     * @throws ModelPersistenceException if loading fails
     */
    @SuppressWarnings("unchecked")
    public static <T> T load(String filePath) {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty");
        }
        
        File file = new File(filePath);
        if (!file.exists()) {
            throw new ModelPersistenceException("Model file not found: " + filePath);
        }
        
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(filePath)))) {
            // Validate magic header
            String header = ois.readUTF();
            if (!MAGIC_HEADER.equals(header)) {
                throw new ModelPersistenceException(
                    "Invalid model file format. Expected MindForge model file.");
            }
            // Read model class name (for info)
            String className = ois.readUTF();
            // Read timestamp (for info)
            long timestamp = ois.readLong();
            // Read the model
            return (T) ois.readObject();
        } catch (IOException e) {
            throw new ModelPersistenceException("Failed to load model from: " + filePath, e);
        } catch (ClassNotFoundException e) {
            throw new ModelPersistenceException("Model class not found. Ensure all dependencies are available.", e);
        }
    }
    
    /**
     * Loads a model from a file with type checking.
     * 
     * @param <T> The expected type of the model
     * @param filePath Path to the saved model file
     * @param modelClass The expected class of the model
     * @return The loaded model
     * @throws ModelPersistenceException if loading fails or type mismatch
     */
    public static <T> T load(String filePath, Class<T> modelClass) {
        Object model = load(filePath);
        if (!modelClass.isInstance(model)) {
            throw new ModelPersistenceException(
                "Type mismatch: expected " + modelClass.getName() + 
                " but found " + model.getClass().getName());
        }
        return modelClass.cast(model);
    }
    
    /**
     * Gets metadata about a saved model without fully loading it.
     * 
     * @param filePath Path to the saved model file
     * @return ModelMetadata containing information about the saved model
     * @throws ModelPersistenceException if reading fails
     */
    public static ModelMetadata getMetadata(String filePath) {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty");
        }
        
        File file = new File(filePath);
        if (!file.exists()) {
            throw new ModelPersistenceException("Model file not found: " + filePath);
        }
        
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(filePath)))) {
            String header = ois.readUTF();
            if (!MAGIC_HEADER.equals(header)) {
                throw new ModelPersistenceException(
                    "Invalid model file format. Expected MindForge model file.");
            }
            String className = ois.readUTF();
            long timestamp = ois.readLong();
            
            return new ModelMetadata(className, timestamp, file.length());
        } catch (IOException e) {
            throw new ModelPersistenceException("Failed to read model metadata from: " + filePath, e);
        }
    }
    
    /**
     * Checks if a file is a valid MindForge model file.
     * 
     * @param filePath Path to check
     * @return true if the file is a valid MindForge model file
     */
    public static boolean isValidModelFile(String filePath) {
        if (filePath == null || filePath.trim().isEmpty()) {
            return false;
        }
        
        File file = new File(filePath);
        if (!file.exists() || !file.isFile()) {
            return false;
        }
        
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(filePath)))) {
            String header = ois.readUTF();
            return MAGIC_HEADER.equals(header);
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Saves a model to a byte array.
     * Useful for sending models over network or storing in databases.
     * 
     * @param model The model to serialize
     * @return Byte array containing the serialized model
     * @throws ModelPersistenceException if serialization fails
     */
    public static byte[] toBytes(Serializable model) {
        if (model == null) {
            throw new IllegalArgumentException("Model cannot be null");
        }
        
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeUTF(MAGIC_HEADER);
            oos.writeUTF(model.getClass().getName());
            oos.writeLong(System.currentTimeMillis());
            oos.writeObject(model);
            oos.flush();
            return baos.toByteArray();
        } catch (IOException e) {
            throw new ModelPersistenceException("Failed to serialize model to bytes", e);
        }
    }
    
    /**
     * Loads a model from a byte array.
     * 
     * @param <T> The type of the model
     * @param bytes Byte array containing the serialized model
     * @return The loaded model
     * @throws ModelPersistenceException if deserialization fails
     */
    @SuppressWarnings("unchecked")
    public static <T> T fromBytes(byte[] bytes) {
        if (bytes == null || bytes.length == 0) {
            throw new IllegalArgumentException("Bytes cannot be null or empty");
        }
        
        try (ObjectInputStream ois = new ObjectInputStream(
                new ByteArrayInputStream(bytes))) {
            String header = ois.readUTF();
            if (!MAGIC_HEADER.equals(header)) {
                throw new ModelPersistenceException(
                    "Invalid model data. Expected MindForge model format.");
            }
            String className = ois.readUTF();
            long timestamp = ois.readLong();
            return (T) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new ModelPersistenceException("Failed to deserialize model from bytes", e);
        }
    }
    
    /**
     * Metadata about a saved model.
     */
    public static class ModelMetadata {
        private final String className;
        private final long savedTimestamp;
        private final long fileSize;
        
        public ModelMetadata(String className, long savedTimestamp, long fileSize) {
            this.className = className;
            this.savedTimestamp = savedTimestamp;
            this.fileSize = fileSize;
        }
        
        /**
         * Gets the class name of the saved model.
         */
        public String getClassName() {
            return className;
        }
        
        /**
         * Gets the simple class name (without package).
         */
        public String getSimpleClassName() {
            int lastDot = className.lastIndexOf('.');
            return lastDot >= 0 ? className.substring(lastDot + 1) : className;
        }
        
        /**
         * Gets the timestamp when the model was saved.
         */
        public long getSavedTimestamp() {
            return savedTimestamp;
        }
        
        /**
         * Gets the file size in bytes.
         */
        public long getFileSize() {
            return fileSize;
        }
        
        @Override
        public String toString() {
            return String.format("ModelMetadata(class=%s, saved=%d, size=%d bytes)",
                getSimpleClassName(), savedTimestamp, fileSize);
        }
    }
    
    // Private constructor to prevent instantiation
    private ModelPersistence() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }
}
