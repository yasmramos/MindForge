package com.mindforge.persistence;

/**
 * Exception thrown when model persistence operations fail.
 * 
 * This exception wraps underlying I/O and serialization exceptions
 * to provide a consistent error handling interface for model
 * save/load operations.
 * 
 * @author MindForge
 * @version 1.0.8-alpha
 */
public class ModelPersistenceException extends RuntimeException {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * Creates a new ModelPersistenceException with a message.
     * 
     * @param message The error message
     */
    public ModelPersistenceException(String message) {
        super(message);
    }
    
    /**
     * Creates a new ModelPersistenceException with a message and cause.
     * 
     * @param message The error message
     * @param cause The underlying cause of the exception
     */
    public ModelPersistenceException(String message, Throwable cause) {
        super(message, cause);
    }
}
