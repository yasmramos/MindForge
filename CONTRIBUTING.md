# Contributing to MindForge

Thank you for your interest in contributing to MindForge! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MindForge.git
   cd MindForge
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/yasmramos/MindForge.git
   ```

## Development Setup

### Requirements

- Java 11 or higher
- Maven 3.6+
- Git

### Building the Project

```bash
# Build the project
mvn clean compile

# Run all tests
mvn test

# Build JAR without tests
mvn clean package -DskipTests

# Install to local Maven repository
mvn clean install -DskipTests
```

### Running Examples

```bash
cd examples
mvn compile exec:java -Dexec.mainClass="com.mindforge.examples.QuickStart"
```

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Java version and OS
   - Minimal code example if possible

### Suggesting Features

1. Check existing issues/discussions
2. Create a feature request with:
   - Clear description of the feature
   - Use case / motivation
   - Proposed API (if applicable)

### Contributing Code

1. **Find or create an issue** for what you want to work on
2. **Fork and branch**: Create a feature branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes**: Write code following our standards
4. **Test**: Add/update tests and ensure all pass
5. **Commit**: Use clear, descriptive commit messages
6. **Push and PR**: Push to your fork and create a Pull Request

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass: `mvn test`
4. Update CHANGELOG.md if applicable
5. Request review from maintainers

### PR Title Format

```
[TYPE] Brief description

Types:
- [FEAT] New feature
- [FIX] Bug fix
- [DOCS] Documentation
- [REFACTOR] Code refactoring
- [TEST] Test additions/changes
- [PERF] Performance improvements
```

## Coding Standards

### Java Style

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 120 characters
- Follow standard Java naming conventions:
  - Classes: `PascalCase`
  - Methods/variables: `camelCase`
  - Constants: `UPPER_SNAKE_CASE`

### Code Organization

```java
package com.mindforge.category;

import java.util.*;  // Standard library imports first
import com.mindforge.*;  // Project imports second

/**
 * Class description.
 * 
 * @author Your Name
 */
public class ClassName {
    // Constants first
    private static final int CONSTANT = 10;
    
    // Instance variables
    private int field;
    
    // Constructors
    public ClassName() { }
    
    // Public methods
    public void publicMethod() { }
    
    // Private methods
    private void helperMethod() { }
}
```

### Documentation

- All public classes and methods must have Javadoc
- Include `@param`, `@return`, `@throws` as appropriate
- Add usage examples in Javadoc when helpful

```java
/**
 * Trains the model with the given data.
 * 
 * Example:
 * <pre>
 * KNearestNeighbors knn = new KNearestNeighbors(5);
 * knn.train(X, y);
 * </pre>
 * 
 * @param X feature matrix (n_samples x n_features)
 * @param y target labels (n_samples)
 * @throws IllegalArgumentException if X and y have different lengths
 */
public void train(double[][] X, int[] y) {
    // implementation
}
```

## Testing Guidelines

### Test Structure

- Place tests in `src/test/java` mirroring the main source structure
- Use JUnit 5 for testing
- Name test classes as `ClassNameTest`
- Name test methods descriptively: `testMethodName_scenario_expectedResult`

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Test with various input sizes

```java
@Test
void testPredict_withValidInput_returnsCorrectLabel() {
    // Arrange
    KNearestNeighbors knn = new KNearestNeighbors(3);
    double[][] X = {{1.0, 2.0}, {3.0, 4.0}};
    int[] y = {0, 1};
    knn.train(X, y);
    
    // Act
    int prediction = knn.predict(new double[]{1.5, 2.5});
    
    // Assert
    assertEquals(0, prediction);
}
```

### Running Tests

```bash
# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=KNearestNeighborsTest

# Run with coverage report
mvn verify jacoco:report
```

## Documentation

### README Updates

- Update README.md when adding new features
- Include usage examples

### Examples

- Add examples to `examples/` directory for new features
- Update `examples/README.md` with new examples

### Javadoc

- Generate Javadoc: `mvn javadoc:javadoc`
- View at `target/site/apidocs/index.html`

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for contributing to MindForge!
