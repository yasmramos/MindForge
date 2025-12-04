package com.mindforge.util;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Simple logging utility for MindForge.
 */
public class MindForgeLogger {
    
    public enum Level {
        DEBUG(0), INFO(1), WARN(2), ERROR(3), OFF(4);
        
        private final int value;
        
        Level(int value) {
            this.value = value;
        }
        
        public int getValue() {
            return value;
        }
    }
    
    private static MindForgeLogger instance;
    private Level level;
    private boolean consoleOutput;
    private PrintWriter fileWriter;
    private SimpleDateFormat dateFormat;
    private String name;
    
    /**
     * Private constructor for singleton pattern.
     */
    private MindForgeLogger() {
        this.level = Level.INFO;
        this.consoleOutput = true;
        this.dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
        this.name = "MindForge";
    }
    
    /**
     * Get the singleton logger instance.
     * 
     * @return logger instance
     */
    public static synchronized MindForgeLogger getLogger() {
        if (instance == null) {
            instance = new MindForgeLogger();
        }
        return instance;
    }
    
    /**
     * Get a named logger.
     * 
     * @param name logger name
     * @return logger instance
     */
    public static MindForgeLogger getLogger(String name) {
        MindForgeLogger logger = getLogger();
        logger.name = name;
        return logger;
    }
    
    /**
     * Set the logging level.
     * 
     * @param level minimum level to log
     */
    public void setLevel(Level level) {
        this.level = level;
    }
    
    /**
     * Get the current logging level.
     * 
     * @return current level
     */
    public Level getLevel() {
        return level;
    }
    
    /**
     * Enable or disable console output.
     * 
     * @param enabled true to enable console output
     */
    public void setConsoleOutput(boolean enabled) {
        this.consoleOutput = enabled;
    }
    
    /**
     * Set a file for logging output.
     * 
     * @param filePath path to log file
     * @throws IOException if file cannot be opened
     */
    public void setLogFile(String filePath) throws IOException {
        if (fileWriter != null) {
            fileWriter.close();
        }
        fileWriter = new PrintWriter(new FileWriter(filePath, true));
    }
    
    /**
     * Close the log file.
     */
    public void closeLogFile() {
        if (fileWriter != null) {
            fileWriter.close();
            fileWriter = null;
        }
    }
    
    /**
     * Log a message at the specified level.
     * 
     * @param level log level
     * @param message message to log
     */
    public void log(Level level, String message) {
        if (level.getValue() >= this.level.getValue()) {
            String formattedMessage = formatMessage(level, message);
            
            if (consoleOutput) {
                if (level == Level.ERROR) {
                    System.err.println(formattedMessage);
                } else {
                    System.out.println(formattedMessage);
                }
            }
            
            if (fileWriter != null) {
                fileWriter.println(formattedMessage);
                fileWriter.flush();
            }
        }
    }
    
    /**
     * Log a message with exception at the specified level.
     * 
     * @param level log level
     * @param message message to log
     * @param throwable exception to log
     */
    public void log(Level level, String message, Throwable throwable) {
        log(level, message);
        if (level.getValue() >= this.level.getValue()) {
            if (consoleOutput) {
                throwable.printStackTrace(System.err);
            }
            if (fileWriter != null) {
                throwable.printStackTrace(fileWriter);
                fileWriter.flush();
            }
        }
    }
    
    /**
     * Log a debug message.
     * 
     * @param message message to log
     */
    public void debug(String message) {
        log(Level.DEBUG, message);
    }
    
    /**
     * Log an info message.
     * 
     * @param message message to log
     */
    public void info(String message) {
        log(Level.INFO, message);
    }
    
    /**
     * Log a warning message.
     * 
     * @param message message to log
     */
    public void warn(String message) {
        log(Level.WARN, message);
    }
    
    /**
     * Log an error message.
     * 
     * @param message message to log
     */
    public void error(String message) {
        log(Level.ERROR, message);
    }
    
    /**
     * Log an error message with exception.
     * 
     * @param message message to log
     * @param throwable exception to log
     */
    public void error(String message, Throwable throwable) {
        log(Level.ERROR, message, throwable);
    }
    
    /**
     * Log formatted debug message.
     * 
     * @param format format string
     * @param args arguments
     */
    public void debug(String format, Object... args) {
        debug(String.format(format, args));
    }
    
    /**
     * Log formatted info message.
     * 
     * @param format format string
     * @param args arguments
     */
    public void info(String format, Object... args) {
        info(String.format(format, args));
    }
    
    /**
     * Log formatted warning message.
     * 
     * @param format format string
     * @param args arguments
     */
    public void warn(String format, Object... args) {
        warn(String.format(format, args));
    }
    
    /**
     * Log formatted error message.
     * 
     * @param format format string
     * @param args arguments
     */
    public void error(String format, Object... args) {
        error(String.format(format, args));
    }
    
    private String formatMessage(Level level, String message) {
        String timestamp = dateFormat.format(new Date());
        return String.format("[%s] [%s] [%s] %s", timestamp, level.name(), name, message);
    }
    
    /**
     * Check if debug level is enabled.
     * 
     * @return true if debug is enabled
     */
    public boolean isDebugEnabled() {
        return level.getValue() <= Level.DEBUG.getValue();
    }
    
    /**
     * Check if info level is enabled.
     * 
     * @return true if info is enabled
     */
    public boolean isInfoEnabled() {
        return level.getValue() <= Level.INFO.getValue();
    }
}
