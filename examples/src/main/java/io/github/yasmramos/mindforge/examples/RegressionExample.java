package io.github.yasmramos.mindforge.examples;

import io.github.yasmramos.mindforge.regression.LinearRegression;
import io.github.yasmramos.mindforge.regression.RidgeRegression;
import io.github.yasmramos.mindforge.preprocessing.StandardScaler;
import io.github.yasmramos.mindforge.validation.Metrics;
import java.util.Random;

/**
 * Regression Example with MindForge
 * 
 * Demonstrates regression algorithms:
 * - Linear Regression
 * - Ridge Regression (L2 regularization)
 * - Model evaluation metrics (MSE, RMSE, MAE, R²)
 */
public class RegressionExample {

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("MindForge Regression Example");
        System.out.println("=".repeat(60));

        // Generate synthetic regression data: y = 2*x1 + 3*x2 + noise
        System.out.println("\n1. Generating synthetic data...");
        double[][] X = new double[100][2];
        double[] y = new double[100];
        Random rand = new Random(42);
        
        for (int i = 0; i < 100; i++) {
            X[i][0] = rand.nextDouble() * 10;  // x1: 0-10
            X[i][1] = rand.nextDouble() * 10;  // x2: 0-10
            y[i] = 2 * X[i][0] + 3 * X[i][1] + rand.nextGaussian() * 0.5;  // y = 2*x1 + 3*x2 + noise
        }
        System.out.printf("   Created %d samples with 2 features%n", X.length);
        System.out.println("   True relationship: y = 2*x1 + 3*x2 + noise");

        // Split data manually (80/20)
        int trainSize = 80;
        double[][] XTrain = new double[trainSize][2];
        double[][] XTest = new double[100 - trainSize][2];
        double[] yTrain = new double[trainSize];
        double[] yTest = new double[100 - trainSize];
        
        System.arraycopy(X, 0, XTrain, 0, trainSize);
        System.arraycopy(X, trainSize, XTest, 0, 100 - trainSize);
        System.arraycopy(y, 0, yTrain, 0, trainSize);
        System.arraycopy(y, trainSize, yTest, 0, 100 - trainSize);
        
        System.out.printf("   Training samples: %d, Test samples: %d%n", trainSize, 100 - trainSize);

        // Scale features
        System.out.println("\n2. Scaling features...");
        StandardScaler scaler = new StandardScaler();
        scaler.fit(XTrain);
        XTrain = scaler.transform(XTrain);
        XTest = scaler.transform(XTest);

        // Linear Regression
        System.out.println("\n3. Training Linear Regression...");
        System.out.println("   " + "-".repeat(50));
        LinearRegression lr = new LinearRegression();
        lr.train(XTrain, yTrain);
        
        double[] lrPred = new double[XTest.length];
        for (int i = 0; i < XTest.length; i++) {
            lrPred[i] = lr.predict(XTest[i]);
        }
        
        double lrMse = Metrics.mse(yTest, lrPred);
        double lrRmse = Metrics.rmse(yTest, lrPred);
        double lrMae = Metrics.mae(yTest, lrPred);
        double lrR2 = Metrics.r2Score(yTest, lrPred);
        
        System.out.printf("   MSE:  %.4f%n", lrMse);
        System.out.printf("   RMSE: %.4f%n", lrRmse);
        System.out.printf("   MAE:  %.4f%n", lrMae);
        System.out.printf("   R²:   %.4f%n", lrR2);
        
        double[] lrWeights = lr.getWeights();
        System.out.printf("   Learned weights: [%.4f, %.4f]%n", lrWeights[0], lrWeights[1]);
        System.out.printf("   Bias: %.4f%n", lr.getBias());

        // Ridge Regression with different alpha values
        System.out.println("\n4. Ridge Regression Comparison (varying alpha):");
        System.out.println("   " + "-".repeat(50));
        
        double[] alphas = {0.01, 0.1, 1.0, 10.0};
        for (double alpha : alphas) {
            RidgeRegression ridge = new RidgeRegression(alpha);
            ridge.train(XTrain, yTrain);
            
            double[] ridgePred = ridge.predict(XTest);
            double ridgeR2 = Metrics.r2Score(yTest, ridgePred);
            double ridgeRmse = Metrics.rmse(yTest, ridgePred);
            
            double[] coefs = ridge.getCoefficients();
            System.out.printf("   alpha=%.2f: R²=%.4f, RMSE=%.4f, coefs=[%.3f, %.3f]%n", 
                              alpha, ridgeR2, ridgeRmse, coefs[0], coefs[1]);
        }

        // Practical Example: House Price Prediction
        System.out.println("\n5. PRACTICAL EXAMPLE: House Price Prediction");
        System.out.println("   " + "-".repeat(50));
        
        // Simulate house data: [size_sqft, bedrooms] -> price
        double[][] houses = {
            {1400, 3}, {1600, 3}, {1700, 3}, {1875, 2}, {1100, 3},
            {1550, 2}, {2350, 4}, {2450, 4}, {1425, 3}, {1700, 3},
            {1800, 3}, {2050, 4}, {1650, 3}, {1200, 2}, {2150, 4}
        };
        double[] prices = {
            245, 312, 279, 308, 199,
            219, 405, 324, 319, 255,
            289, 356, 290, 185, 398
        };
        
        // Train model
        StandardScaler houseScaler = new StandardScaler();
        houseScaler.fit(houses);
        double[][] housesScaled = houseScaler.transform(houses);
        
        LinearRegression houseLr = new LinearRegression();
        houseLr.train(housesScaled, prices);
        
        // Predict new houses
        double[][] newHouses = {{1500, 3}, {2000, 4}, {1800, 2}};
        double[][] newHousesScaled = houseScaler.transform(newHouses);
        
        System.out.println("\n   Price predictions for new houses:");
        for (int i = 0; i < newHouses.length; i++) {
            double pred = houseLr.predict(newHousesScaled[i]);
            System.out.printf("     House %d: %.0f sqft, %d beds -> $%.0fk%n",
                              i + 1, newHouses[i][0], (int) newHouses[i][1], pred);
        }

        // Summary
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Regression Summary:");
        System.out.println("=".repeat(60));
        System.out.println("- Linear Regression: Simple, interpretable baseline");
        System.out.println("- Ridge Regression: Adds L2 regularization to prevent overfitting");
        System.out.println("- Key metrics: MSE, RMSE (error magnitude), R² (explained variance)");
        System.out.println("- Higher R² = better model (max 1.0)");
    }
}
