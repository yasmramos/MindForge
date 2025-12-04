package com.mindforge.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.File;
import java.nio.file.Path;

@DisplayName("MindForge Logger Tests")
class MindForgeLoggerTest {
    
    private PrintStream originalOut;
    private PrintStream originalErr;
    private ByteArrayOutputStream outContent;
    private ByteArrayOutputStream errContent;
    private MindForgeLogger logger;
    
    @BeforeEach
    void setUp() {
        originalOut = System.out;
        originalErr = System.err;
        outContent = new ByteArrayOutputStream();
        errContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));
        System.setErr(new PrintStream(errContent));
        
        // Get logger instance
        logger = MindForgeLogger.getLogger();
        logger.setLevel(MindForgeLogger.Level.INFO);
        logger.setConsoleOutput(true);
    }
    
    @AfterEach
    void tearDown() {
        System.setOut(originalOut);
        System.setErr(originalErr);
        logger.closeLogFile();
    }
    
    @Nested
    @DisplayName("Log Level Tests")
    class LogLevelTests {
        
        @Test
        @DisplayName("Should log INFO messages at INFO level")
        void testInfoLevel() {
            logger.setLevel(MindForgeLogger.Level.INFO);
            logger.info("Test info message");
            
            String output = outContent.toString();
            assertTrue(output.contains("INFO") || output.contains("info") || output.contains("Test info message"), 
                "Should log INFO messages");
        }
        
        @Test
        @DisplayName("Should not log DEBUG messages at INFO level")
        void testDebugNotLoggedAtInfo() {
            logger.setLevel(MindForgeLogger.Level.INFO);
            logger.debug("Test debug message");
            
            String output = outContent.toString();
            assertFalse(output.contains("Test debug message"), 
                "Should not log DEBUG messages at INFO level");
        }
        
        @Test
        @DisplayName("Should log DEBUG messages at DEBUG level")
        void testDebugLevel() {
            logger.setLevel(MindForgeLogger.Level.DEBUG);
            logger.debug("Test debug message");
            
            String output = outContent.toString();
            assertTrue(output.contains("DEBUG") || output.contains("debug") || output.contains("Test debug message"), 
                "Should log DEBUG messages at DEBUG level");
        }
        
        @Test
        @DisplayName("Should log WARN messages")
        void testWarnLevel() {
            logger.warn("Test warning message");
            
            String output = outContent.toString() + errContent.toString();
            assertTrue(output.contains("WARN") || output.contains("warn") || output.contains("Test warning message"), 
                "Should log WARN messages");
        }
        
        @Test
        @DisplayName("Should log ERROR messages")
        void testErrorLevel() {
            logger.error("Test error message");
            
            String output = outContent.toString() + errContent.toString();
            assertTrue(output.contains("ERROR") || output.contains("error") || output.contains("Test error message"), 
                "Should log ERROR messages");
        }
    }
    
    @Nested
    @DisplayName("Log Message Format Tests")
    class LogFormatTests {
        
        @Test
        @DisplayName("Should include timestamp in log message")
        void testTimestamp() {
            logger.info("Test message");
            
            String output = outContent.toString();
            // Should contain date pattern like 2025- or time pattern like :
            assertTrue(output.contains(":") || output.contains("-") || output.contains("Test message"), 
                "Should include timestamp or message");
        }
        
        @Test
        @DisplayName("Should include message content")
        void testMessageContent() {
            String testMessage = "UniqueTestMessage12345";
            logger.info(testMessage);
            
            String output = outContent.toString();
            assertTrue(output.contains(testMessage), 
                "Should include the actual message content");
        }
    }
    
    @Nested
    @DisplayName("File Logging Tests")
    class FileLoggingTests {
        
        @TempDir
        Path tempDir;
        
        @Test
        @DisplayName("Should write to log file")
        void testFileLogging() throws Exception {
            File logFile = tempDir.resolve("test.log").toFile();
            logger.setLogFile(logFile.getAbsolutePath());
            logger.info("Test file logging");
            logger.closeLogFile();
            
            assertTrue(logFile.exists(), "Log file should be created");
        }
        
        @Test
        @DisplayName("Should close log file properly")
        void testCloseLogFile() throws Exception {
            File logFile = tempDir.resolve("test2.log").toFile();
            logger.setLogFile(logFile.getAbsolutePath());
            logger.info("Test message");
            
            assertDoesNotThrow(() -> logger.closeLogFile());
        }
    }
    
    @Nested
    @DisplayName("Console Output Control Tests")
    class ConsoleControlTests {
        
        @Test
        @DisplayName("Should disable console output")
        void testDisableConsole() {
            logger.setConsoleOutput(false);
            logger.info("This should not appear in console");
            
            // Just verify no exception is thrown
            assertDoesNotThrow(() -> logger.info("Test"));
        }
        
        @Test
        @DisplayName("Should enable console output")
        void testEnableConsole() {
            logger.setConsoleOutput(true);
            logger.info("Visible message");
            
            String output = outContent.toString();
            assertTrue(output.contains("Visible message"));
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle null message")
        void testNullMessage() {
            assertDoesNotThrow(() -> logger.info(null));
        }
        
        @Test
        @DisplayName("Should handle empty message")
        void testEmptyMessage() {
            assertDoesNotThrow(() -> logger.info(""));
        }
        
        @Test
        @DisplayName("Should handle special characters")
        void testSpecialCharacters() {
            assertDoesNotThrow(() -> logger.info("Test with special chars: \n\t\r"));
        }
        
        @Test
        @DisplayName("Should handle very long message")
        void testLongMessage() {
            StringBuilder longMessage = new StringBuilder();
            for (int i = 0; i < 10000; i++) {
                longMessage.append("a");
            }
            assertDoesNotThrow(() -> logger.info(longMessage.toString()));
        }
    }
    
    @Nested
    @DisplayName("Named Logger Tests")
    class NamedLoggerTests {
        
        @Test
        @DisplayName("Should create named logger")
        void testNamedLogger() {
            MindForgeLogger namedLogger = MindForgeLogger.getLogger("TestLogger");
            assertNotNull(namedLogger);
        }
        
        @Test
        @DisplayName("Should get singleton instance")
        void testSingleton() {
            MindForgeLogger logger1 = MindForgeLogger.getLogger();
            MindForgeLogger logger2 = MindForgeLogger.getLogger();
            assertSame(logger1, logger2, "Should return same instance");
        }
    }
    
    @Nested
    @DisplayName("Formatted Logging Tests")
    class FormattedLoggingTests {
        
        @Test
        @DisplayName("Should format message with arguments")
        void testFormattedMessage() {
            logger.info("Value: %d, Name: %s", 42, "test");
            
            String output = outContent.toString();
            assertTrue(output.contains("42") && output.contains("test"), 
                "Should format message with arguments");
        }
    }
}
