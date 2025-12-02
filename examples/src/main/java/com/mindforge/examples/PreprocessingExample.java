package com.mindforge.examples;

import com.mindforge.preprocessing.StandardScaler;
import com.mindforge.preprocessing.MinMaxScaler;
import com.mindforge.preprocessing.LabelEncoder;
import java.util.Arrays;

/**
 * Preprocessing Example with MindForge
 * 
 * Demonstrates data preprocessing techniques:
 * - StandardScaler (z-score normalization)
 * - MinMaxScaler (range scaling)
 * - LabelEncoder (categorical to numeric)
 */
public class PreprocessingExample {

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("MindForge Preprocessing Example");
        System.out.println("=".repeat(60));

        // Sample data with different scales
        double[][] data = {
            {25, 50000, 1.75},   // age, salary, height(m)
            {30, 60000, 1.80},
            {35, 75000, 1.65},
            {40, 90000, 1.90},
            {45, 120000, 1.70}
        };
        
        System.out.println("\n1. Original Data:");
        System.out.println("   " + "-".repeat(50));
        System.out.println("   Age    | Salary   | Height (m)");
        System.out.println("   " + "-".repeat(35));
        for (double[] row : data) {
            System.out.printf("   %.0f     | $%.0f  | %.2f%n", row[0], row[1], row[2]);
        }

        // StandardScaler - zero mean, unit variance
        System.out.println("\n2. StandardScaler (Z-Score Normalization):");
        System.out.println("   " + "-".repeat(50));
        System.out.println("   Formula: z = (x - mean) / std");
        System.out.println("   Result: mean=0, std=1 for each feature");
        
        StandardScaler stdScaler = new StandardScaler();
        stdScaler.fit(data);
        double[][] stdScaled = stdScaler.transform(data);
        
        System.out.println("\n   Scaled values:");
        System.out.println("   Age      | Salary   | Height");
        System.out.println("   " + "-".repeat(35));
        for (double[] row : stdScaled) {
            System.out.printf("   %+.4f  | %+.4f  | %+.4f%n", row[0], row[1], row[2]);
        }
        
        // Verify mean and std
        double[] means = new double[3];
        double[] stds = new double[3];
        for (int j = 0; j < 3; j++) {
            for (double[] row : stdScaled) {
                means[j] += row[j];
            }
            means[j] /= stdScaled.length;
            for (double[] row : stdScaled) {
                stds[j] += Math.pow(row[j] - means[j], 2);
            }
            stds[j] = Math.sqrt(stds[j] / stdScaled.length);
        }
        System.out.printf("\n   Verification - Mean: [%.4f, %.4f, %.4f]%n", means[0], means[1], means[2]);
        System.out.printf("   Verification - Std:  [%.4f, %.4f, %.4f]%n", stds[0], stds[1], stds[2]);

        // MinMaxScaler - scale to [0, 1]
        System.out.println("\n3. MinMaxScaler (Range Scaling to [0, 1]):");
        System.out.println("   " + "-".repeat(50));
        System.out.println("   Formula: x_scaled = (x - min) / (max - min)");
        System.out.println("   Result: all values in range [0, 1]");
        
        MinMaxScaler minMaxScaler = new MinMaxScaler();
        minMaxScaler.fit(data);
        double[][] minMaxScaled = minMaxScaler.transform(data);
        
        System.out.println("\n   Scaled values:");
        System.out.println("   Age     | Salary  | Height");
        System.out.println("   " + "-".repeat(35));
        for (double[] row : minMaxScaled) {
            System.out.printf("   %.4f  | %.4f  | %.4f%n", row[0], row[1], row[2]);
        }

        // Custom range MinMaxScaler
        System.out.println("\n4. MinMaxScaler with custom range [-1, 1]:");
        System.out.println("   " + "-".repeat(50));
        
        MinMaxScaler customScaler = new MinMaxScaler(-1, 1);
        customScaler.fit(data);
        double[][] customScaled = customScaler.transform(data);
        
        System.out.println("   Scaled values:");
        System.out.println("   Age      | Salary   | Height");
        System.out.println("   " + "-".repeat(35));
        for (double[] row : customScaled) {
            System.out.printf("   %+.4f  | %+.4f  | %+.4f%n", row[0], row[1], row[2]);
        }

        // LabelEncoder - categorical to numeric
        System.out.println("\n5. LabelEncoder (Categorical to Numeric):");
        System.out.println("   " + "-".repeat(50));
        
        String[] categories = {"Red", "Green", "Blue", "Red", "Blue", "Green", "Red"};
        
        System.out.println("   Original labels: " + Arrays.toString(categories));
        
        LabelEncoder encoder = new LabelEncoder();
        int[] encoded = encoder.fitTransform(categories);
        
        System.out.println("   Encoded values:  " + Arrays.toString(encoded));
        System.out.println("   Classes: " + Arrays.toString(encoder.getClasses()));
        
        // Inverse transform
        String[] decoded = encoder.inverseTransform(new int[]{0, 1, 2});
        System.out.println("   Inverse [0,1,2]: " + Arrays.toString(decoded));

        // Practical Example: Employee Data Preprocessing
        System.out.println("\n6. PRACTICAL EXAMPLE: Employee Data Preprocessing");
        System.out.println("   " + "-".repeat(50));
        
        // Employee data
        String[] departments = {"Engineering", "Sales", "Marketing", "Engineering", "Sales"};
        double[][] employeeData = {
            {28, 65000},  // age, salary
            {35, 55000},
            {42, 48000},
            {31, 72000},
            {38, 58000}
        };
        
        System.out.println("\n   Raw employee data:");
        System.out.println("   Dept        | Age | Salary");
        System.out.println("   " + "-".repeat(35));
        for (int i = 0; i < departments.length; i++) {
            System.out.printf("   %-11s | %d  | $%,.0f%n", 
                              departments[i], (int)employeeData[i][0], employeeData[i][1]);
        }
        
        // Encode departments
        LabelEncoder deptEncoder = new LabelEncoder();
        int[] deptEncoded = deptEncoder.fitTransform(departments);
        
        // Scale numeric features
        StandardScaler empScaler = new StandardScaler();
        empScaler.fit(employeeData);
        double[][] empScaled = empScaler.transform(employeeData);
        
        System.out.println("\n   Preprocessed data (ready for ML):");
        System.out.println("   Dept (enc) | Age (scaled) | Salary (scaled)");
        System.out.println("   " + "-".repeat(45));
        for (int i = 0; i < departments.length; i++) {
            System.out.printf("   %d          | %+.4f       | %+.4f%n", 
                              deptEncoded[i], empScaled[i][0], empScaled[i][1]);
        }

        // Summary
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Preprocessing Summary:");
        System.out.println("=".repeat(60));
        System.out.println("- StandardScaler: Best for algorithms assuming normal distribution");
        System.out.println("- MinMaxScaler: Best when you need bounded values [0,1] or [-1,1]");
        System.out.println("- LabelEncoder: Converts categorical strings to integers");
        System.out.println("- Always fit on training data, transform both train and test!");
    }
}
