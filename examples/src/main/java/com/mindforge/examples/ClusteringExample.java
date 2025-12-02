package com.mindforge.examples;

import com.mindforge.clustering.KMeans;
import com.mindforge.preprocessing.StandardScaler;
import java.util.Random;

/**
 * Clustering Example with MindForge
 * 
 * Demonstrates K-Means clustering:
 * - Grouping similar data points
 * - Finding cluster centers
 * - Customer segmentation use case
 */
public class ClusteringExample {

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("MindForge Clustering Example - K-Means");
        System.out.println("=".repeat(60));

        // Generate synthetic clustered data (3 clusters)
        System.out.println("\n1. Generating synthetic data with 3 clusters...");
        double[][] X = generateClusteredData(150, 3);
        System.out.printf("   Created %d samples with 2 features%n", X.length);
        
        // Scale features
        System.out.println("\n2. Scaling features...");
        StandardScaler scaler = new StandardScaler();
        scaler.fit(X);
        X = scaler.transform(X);

        // K-Means Clustering with different k values
        System.out.println("\n3. K-Means Clustering Comparison:");
        System.out.println("   " + "-".repeat(50));
        
        for (int k = 2; k <= 5; k++) {
            KMeans kmeans = new KMeans(k, 100, new Random(42));
            int[] labels = kmeans.fitPredict(X);
            double inertia = kmeans.getInertia();
            
            // Count samples per cluster
            int[] counts = new int[k];
            for (int label : labels) {
                counts[label]++;
            }
            
            StringBuilder clusterInfo = new StringBuilder("[");
            for (int i = 0; i < k; i++) {
                clusterInfo.append(counts[i]);
                if (i < k - 1) clusterInfo.append(", ");
            }
            clusterInfo.append("]");
            
            System.out.printf("   k=%d: Inertia=%.2f, Cluster sizes=%s%n", 
                              k, inertia, clusterInfo);
        }

        // Best model (k=3) details
        System.out.println("\n4. K-Means (k=3) Details:");
        System.out.println("   " + "-".repeat(50));
        
        KMeans kmeans3 = new KMeans(3, 100, new Random(42));
        int[] labels = kmeans3.fitPredict(X);
        double[][] centroids = kmeans3.getCentroids();
        
        System.out.println("   Cluster centroids (scaled coordinates):");
        for (int i = 0; i < centroids.length; i++) {
            System.out.printf("     Cluster %d: (%.4f, %.4f)%n", 
                              i, centroids[i][0], centroids[i][1]);
        }

        // Practical Example: Customer Segmentation
        System.out.println("\n5. PRACTICAL EXAMPLE: Customer Segmentation");
        System.out.println("   " + "-".repeat(50));
        
        // Simulate customer data: [spending_score (0-100), annual_income (k$)]
        double[][] customers = {
            {15, 39}, {16, 81}, {17, 6}, {18, 77}, {19, 40},
            {39, 75}, {40, 35}, {42, 92}, {43, 26}, {44, 71},
            {69, 88}, {71, 61}, {73, 97}, {74, 23}, {75, 62},
            {87, 18}, {88, 95}, {89, 11}, {90, 82}, {91, 28}
        };
        
        StandardScaler custScaler = new StandardScaler();
        custScaler.fit(customers);
        double[][] custScaled = custScaler.transform(customers);
        
        KMeans custKmeans = new KMeans(3, 100, new Random(42));
        int[] custLabels = custKmeans.fitPredict(custScaled);
        
        String[] segmentNames = {"Budget Conscious", "Average Spenders", "Premium Customers"};
        
        // Assign segment names based on cluster characteristics
        System.out.println("\n   Customer Segmentation Results:");
        for (int i = 0; i < customers.length; i++) {
            int segment = custLabels[i] % 3;
            System.out.printf("     Customer %2d: Spending=%.0f, Income=$%.0fk -> %s%n",
                              i + 1, customers[i][0], customers[i][1], 
                              segmentNames[segment]);
        }

        // Summary
        System.out.println("\n" + "=".repeat(60));
        System.out.println("K-Means Clustering Summary:");
        System.out.println("=".repeat(60));
        System.out.println("- K-Means groups similar data points into k clusters");
        System.out.println("- Lower inertia = tighter clusters");
        System.out.println("- Use 'elbow method' to find optimal k");
        System.out.println("- Common use cases: customer segmentation, image compression");
    }

    private static double[][] generateClusteredData(int n, int nClusters) {
        double[][] X = new double[n][2];
        Random rand = new Random(42);
        
        // Cluster centers
        double[][] centers = {{0, 0}, {5, 5}, {-5, 5}};
        
        for (int i = 0; i < n; i++) {
            int cluster = i % nClusters;
            X[i][0] = centers[cluster][0] + rand.nextGaussian();
            X[i][1] = centers[cluster][1] + rand.nextGaussian();
        }
        return X;
    }
}
