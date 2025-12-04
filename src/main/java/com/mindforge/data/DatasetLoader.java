package com.mindforge.data;

import java.util.Random;

/**
 * Loader for popular machine learning datasets.
 */
public class DatasetLoader {
    
    /**
     * Load the Iris dataset.
     * 150 samples, 4 features, 3 classes.
     * 
     * @return Iris dataset
     */
    public static Dataset loadIris() {
        double[][] features = {
            {5.1,3.5,1.4,0.2}, {4.9,3.0,1.4,0.2}, {4.7,3.2,1.3,0.2}, {4.6,3.1,1.5,0.2},
            {5.0,3.6,1.4,0.2}, {5.4,3.9,1.7,0.4}, {4.6,3.4,1.4,0.3}, {5.0,3.4,1.5,0.2},
            {4.4,2.9,1.4,0.2}, {4.9,3.1,1.5,0.1}, {5.4,3.7,1.5,0.2}, {4.8,3.4,1.6,0.2},
            {4.8,3.0,1.4,0.1}, {4.3,3.0,1.1,0.1}, {5.8,4.0,1.2,0.2}, {5.7,4.4,1.5,0.4},
            {5.4,3.9,1.3,0.4}, {5.1,3.5,1.4,0.3}, {5.7,3.8,1.7,0.3}, {5.1,3.8,1.5,0.3},
            {5.4,3.4,1.7,0.2}, {5.1,3.7,1.5,0.4}, {4.6,3.6,1.0,0.2}, {5.1,3.3,1.7,0.5},
            {4.8,3.4,1.9,0.2}, {5.0,3.0,1.6,0.2}, {5.0,3.4,1.6,0.4}, {5.2,3.5,1.5,0.2},
            {5.2,3.4,1.4,0.2}, {4.7,3.2,1.6,0.2}, {4.8,3.1,1.6,0.2}, {5.4,3.4,1.5,0.4},
            {5.2,4.1,1.5,0.1}, {5.5,4.2,1.4,0.2}, {4.9,3.1,1.5,0.2}, {5.0,3.2,1.2,0.2},
            {5.5,3.5,1.3,0.2}, {4.9,3.6,1.4,0.1}, {4.4,3.0,1.3,0.2}, {5.1,3.4,1.5,0.2},
            {5.0,3.5,1.3,0.3}, {4.5,2.3,1.3,0.3}, {4.4,3.2,1.3,0.2}, {5.0,3.5,1.6,0.6},
            {5.1,3.8,1.9,0.4}, {4.8,3.0,1.4,0.3}, {5.1,3.8,1.6,0.2}, {4.6,3.2,1.4,0.2},
            {5.3,3.7,1.5,0.2}, {5.0,3.3,1.4,0.2}, {7.0,3.2,4.7,1.4}, {6.4,3.2,4.5,1.5},
            {6.9,3.1,4.9,1.5}, {5.5,2.3,4.0,1.3}, {6.5,2.8,4.6,1.5}, {5.7,2.8,4.5,1.3},
            {6.3,3.3,4.7,1.6}, {4.9,2.4,3.3,1.0}, {6.6,2.9,4.6,1.3}, {5.2,2.7,3.9,1.4},
            {5.0,2.0,3.5,1.0}, {5.9,3.0,4.2,1.5}, {6.0,2.2,4.0,1.0}, {6.1,2.9,4.7,1.4},
            {5.6,2.9,3.6,1.3}, {6.7,3.1,4.4,1.4}, {5.6,3.0,4.5,1.5}, {5.8,2.7,4.1,1.0},
            {6.2,2.2,4.5,1.5}, {5.6,2.5,3.9,1.1}, {5.9,3.2,4.8,1.8}, {6.1,2.8,4.0,1.3},
            {6.3,2.5,4.9,1.5}, {6.1,2.8,4.7,1.2}, {6.4,2.9,4.3,1.3}, {6.6,3.0,4.4,1.4},
            {6.8,2.8,4.8,1.4}, {6.7,3.0,5.0,1.7}, {6.0,2.9,4.5,1.5}, {5.7,2.6,3.5,1.0},
            {5.5,2.4,3.8,1.1}, {5.5,2.4,3.7,1.0}, {5.8,2.7,3.9,1.2}, {6.0,2.7,5.1,1.6},
            {5.4,3.0,4.5,1.5}, {6.0,3.4,4.5,1.6}, {6.7,3.1,4.7,1.5}, {6.3,2.3,4.4,1.3},
            {5.6,3.0,4.1,1.3}, {5.5,2.5,4.0,1.3}, {5.5,2.6,4.4,1.2}, {6.1,3.0,4.6,1.4},
            {5.8,2.6,4.0,1.2}, {5.0,2.3,3.3,1.0}, {5.6,2.7,4.2,1.3}, {5.7,3.0,4.2,1.2},
            {5.7,2.9,4.2,1.3}, {6.2,2.9,4.3,1.3}, {5.1,2.5,3.0,1.1}, {5.7,2.8,4.1,1.3},
            {6.3,3.3,6.0,2.5}, {5.8,2.7,5.1,1.9}, {7.1,3.0,5.9,2.1}, {6.3,2.9,5.6,1.8},
            {6.5,3.0,5.8,2.2}, {7.6,3.0,6.6,2.1}, {4.9,2.5,4.5,1.7}, {7.3,2.9,6.3,1.8},
            {6.7,2.5,5.8,1.8}, {7.2,3.6,6.1,2.5}, {6.5,3.2,5.1,2.0}, {6.4,2.7,5.3,1.9},
            {6.8,3.0,5.5,2.1}, {5.7,2.5,5.0,2.0}, {5.8,2.8,5.1,2.4}, {6.4,3.2,5.3,2.3},
            {6.5,3.0,5.5,1.8}, {7.7,3.8,6.7,2.2}, {7.7,2.6,6.9,2.3}, {6.0,2.2,5.0,1.5},
            {6.9,3.2,5.7,2.3}, {5.6,2.8,4.9,2.0}, {7.7,2.8,6.7,2.0}, {6.3,2.7,4.9,1.8},
            {6.7,3.3,5.7,2.1}, {7.2,3.2,6.0,1.8}, {6.2,2.8,4.8,1.8}, {6.1,3.0,4.9,1.8},
            {6.4,2.8,5.6,2.1}, {7.2,3.0,5.8,1.6}, {7.4,2.8,6.1,1.9}, {7.9,3.8,6.4,2.0},
            {6.4,2.8,5.6,2.2}, {6.3,2.8,5.1,1.5}, {6.1,2.6,5.6,1.4}, {7.7,3.0,6.1,2.3},
            {6.3,3.4,5.6,2.4}, {6.4,3.1,5.5,1.8}, {6.0,3.0,4.8,1.8}, {6.9,3.1,5.4,2.1},
            {6.7,3.1,5.6,2.4}, {6.9,3.1,5.1,2.3}, {5.8,2.7,5.1,1.9}, {6.8,3.2,5.9,2.3},
            {6.7,3.3,5.7,2.5}, {6.7,3.0,5.2,2.3}, {6.3,2.5,5.0,1.9}, {6.5,3.0,5.2,2.0},
            {6.2,3.4,5.4,2.3}, {5.9,3.0,5.1,1.8}
        };
        
        int[] labels = {
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
        };
        
        Dataset dataset = new Dataset(features, labels);
        dataset.setName("Iris");
        dataset.setDescription("Fisher's Iris dataset - 3 classes of iris plants");
        dataset.setFeatureNames(new String[]{"sepal length", "sepal width", "petal length", "petal width"});
        dataset.setTargetNames(new String[]{"setosa", "versicolor", "virginica"});
        
        return dataset;
    }
    
    /**
     * Load the Wine dataset.
     * 178 samples, 13 features, 3 classes.
     * 
     * @return Wine dataset
     */
    public static Dataset loadWine() {
        double[][] features = {
            {14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065},
            {13.2,1.78,2.14,11.2,100,2.65,2.76,0.26,1.28,4.38,1.05,3.4,1050},
            {13.16,2.36,2.67,18.6,101,2.8,3.24,0.3,2.81,5.68,1.03,3.17,1185},
            {14.37,1.95,2.5,16.8,113,3.85,3.49,0.24,2.18,7.8,0.86,3.45,1480},
            {13.24,2.59,2.87,21,118,2.8,2.69,0.39,1.82,4.32,1.04,2.93,735},
            {14.2,1.76,2.45,15.2,112,3.27,3.39,0.34,1.97,6.75,1.05,2.85,1450},
            {14.39,1.87,2.45,14.6,96,2.5,2.52,0.3,1.98,5.25,1.02,3.58,1290},
            {14.06,2.15,2.61,17.6,121,2.6,2.51,0.31,1.25,5.05,1.06,3.58,1295},
            {14.83,1.64,2.17,14,97,2.8,2.98,0.29,1.98,5.2,1.08,2.85,1045},
            {13.86,1.35,2.27,16,98,2.98,3.15,0.22,1.85,7.22,1.01,3.55,1045},
            {14.1,2.16,2.3,18,105,2.95,3.32,0.22,2.38,5.75,1.25,3.17,1510},
            {14.12,1.48,2.32,16.8,95,2.2,2.43,0.26,1.57,5,1.17,2.82,1280},
            {13.75,1.73,2.41,16,89,2.6,2.76,0.29,1.81,5.6,1.15,2.9,1320},
            {14.75,1.73,2.39,11.4,91,3.1,3.69,0.43,2.81,5.4,1.25,2.73,1150},
            {14.38,1.87,2.38,12,102,3.3,3.64,0.29,2.96,7.5,1.2,3,1547},
            {13.63,1.81,2.7,17.2,112,2.85,2.91,0.3,1.46,7.3,1.28,2.88,1310},
            {14.3,1.92,2.72,20,120,2.8,3.14,0.33,1.97,6.2,1.07,2.65,1280},
            {13.83,1.57,2.62,20,115,2.95,3.4,0.4,1.72,6.6,1.13,2.57,1130},
            {14.19,1.59,2.48,16.5,108,3.3,3.93,0.32,1.86,8.7,1.23,2.82,1680},
            {13.64,3.1,2.56,15.2,116,2.7,3.03,0.17,1.66,5.1,0.96,3.36,845},
            {12.37,0.94,1.36,10.6,88,1.98,0.57,0.28,0.42,1.95,1.05,1.82,520},
            {12.33,1.1,2.28,16,101,2.05,1.09,0.63,0.41,3.27,1.25,1.67,680},
            {12.64,1.36,2.02,16.8,100,2.02,1.41,0.53,0.62,5.75,0.98,1.59,450},
            {13.67,1.25,1.92,18,94,2.1,1.79,0.32,0.73,3.8,1.23,2.46,630},
            {12.37,1.13,2.16,19,87,3.5,3.1,0.19,1.87,4.45,1.22,2.87,420},
            {12.17,1.45,2.53,19,104,1.89,1.75,0.45,1.03,2.95,1.45,2.23,355},
            {12.37,1.21,2.56,18.1,98,2.42,2.65,0.37,2.08,4.6,1.19,2.3,678},
            {13.11,1.01,1.7,15,78,2.98,3.18,0.26,2.28,5.3,1.12,3.18,502},
            {12.37,1.17,1.92,19.6,78,2.11,2,0.27,1.04,4.68,1.12,3.48,510},
            {13.34,0.94,2.36,17,110,2.53,1.3,0.55,0.42,3.17,1.02,1.93,750}
        };
        
        int[] labels = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1};
        
        Dataset dataset = new Dataset(features, labels);
        dataset.setName("Wine (subset)");
        dataset.setDescription("Wine recognition dataset - 3 cultivars subset");
        dataset.setFeatureNames(new String[]{
            "alcohol", "malic_acid", "ash", "alcalinity", "magnesium",
            "phenols", "flavanoids", "nonflavanoid", "proanthocyanins",
            "color_intensity", "hue", "od280/od315", "proline"
        });
        dataset.setTargetNames(new String[]{"class_0", "class_1", "class_2"});
        
        return dataset;
    }
    
    /**
     * Load the Breast Cancer Wisconsin dataset (subset).
     * Binary classification for tumor diagnosis.
     * 
     * @return Breast Cancer dataset
     */
    public static Dataset loadBreastCancer() {
        double[][] features = {
            {17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871},
            {20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667},
            {19.69,21.25,130,1203,0.1096,0.1599,0.1974,0.1279,0.2069,0.05999},
            {11.42,20.38,77.58,386.1,0.1425,0.2839,0.2414,0.1052,0.2597,0.09744},
            {20.29,14.34,135.1,1297,0.1003,0.1328,0.198,0.1043,0.1809,0.05883},
            {12.45,15.7,82.57,477.1,0.1278,0.17,0.1578,0.08089,0.2087,0.07613},
            {18.25,19.98,119.6,1040,0.09463,0.109,0.1127,0.074,0.1794,0.05742},
            {13.71,20.83,90.2,577.9,0.1189,0.1645,0.09366,0.05985,0.2196,0.07451},
            {13,21.82,87.5,519.8,0.1273,0.1932,0.1859,0.09353,0.235,0.07389},
            {12.46,24.04,83.97,475.9,0.1186,0.2396,0.2273,0.08543,0.203,0.08243},
            {16.02,23.24,102.7,797.8,0.08206,0.06669,0.03299,0.03323,0.1528,0.05697},
            {15.78,17.89,103.6,781,0.0971,0.1292,0.09954,0.06606,0.1842,0.06082},
            {19.17,24.8,132.4,1123,0.0974,0.2458,0.2065,0.1118,0.2397,0.078},
            {15.85,23.95,103.7,782.7,0.08401,0.1002,0.09938,0.05364,0.1847,0.05338},
            {13.73,22.61,93.6,578.3,0.1131,0.2293,0.2128,0.08025,0.2069,0.07682},
            {14.54,27.54,96.73,658.8,0.1139,0.1595,0.1639,0.07364,0.2303,0.07077},
            {14.68,20.13,94.74,684.5,0.09867,0.072,0.07395,0.05259,0.1586,0.05922},
            {16.13,20.68,108.1,798.8,0.117,0.2022,0.1722,0.1028,0.2164,0.07356},
            {19.81,22.15,130,1260,0.09831,0.1027,0.1479,0.09498,0.1582,0.05395},
            {13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766}
        };
        
        int[] labels = {1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,1,0,1,1,0};
        
        Dataset dataset = new Dataset(features, labels);
        dataset.setName("Breast Cancer (subset)");
        dataset.setDescription("Breast Cancer Wisconsin - binary classification");
        dataset.setFeatureNames(new String[]{
            "radius", "texture", "perimeter", "area", "smoothness",
            "compactness", "concavity", "concave_points", "symmetry", "fractal_dim"
        });
        dataset.setTargetNames(new String[]{"malignant", "benign"});
        
        return dataset;
    }
    
    /**
     * Generate a random classification dataset.
     * 
     * @param nSamples number of samples
     * @param nFeatures number of features
     * @param nClasses number of classes
     * @param seed random seed
     * @return generated dataset
     */
    public static Dataset makeClassification(int nSamples, int nFeatures, int nClasses, long seed) {
        Random random = new Random(seed);
        
        double[][] features = new double[nSamples][nFeatures];
        int[] labels = new int[nSamples];
        
        // Generate cluster centers
        double[][] centers = new double[nClasses][nFeatures];
        for (int c = 0; c < nClasses; c++) {
            for (int f = 0; f < nFeatures; f++) {
                centers[c][f] = random.nextGaussian() * 3;
            }
        }
        
        // Generate samples around centers
        for (int i = 0; i < nSamples; i++) {
            int classIdx = i % nClasses;
            labels[i] = classIdx;
            for (int f = 0; f < nFeatures; f++) {
                features[i][f] = centers[classIdx][f] + random.nextGaussian();
            }
        }
        
        Dataset dataset = new Dataset(features, labels);
        dataset.setName("Synthetic Classification");
        dataset.setDescription("Randomly generated classification dataset");
        
        return dataset;
    }
    
    /**
     * Generate a random regression dataset.
     * 
     * @param nSamples number of samples
     * @param nFeatures number of features
     * @param noise noise level
     * @param seed random seed
     * @return generated dataset
     */
    public static Dataset makeRegression(int nSamples, int nFeatures, double noise, long seed) {
        Random random = new Random(seed);
        
        double[][] features = new double[nSamples][nFeatures];
        double[] targets = new double[nSamples];
        
        // Generate random coefficients
        double[] coefs = new double[nFeatures];
        for (int f = 0; f < nFeatures; f++) {
            coefs[f] = random.nextGaussian() * 10;
        }
        
        // Generate samples
        for (int i = 0; i < nSamples; i++) {
            double target = 0;
            for (int f = 0; f < nFeatures; f++) {
                features[i][f] = random.nextGaussian();
                target += coefs[f] * features[i][f];
            }
            targets[i] = target + random.nextGaussian() * noise;
        }
        
        Dataset dataset = new Dataset(features, targets);
        dataset.setName("Synthetic Regression");
        dataset.setDescription("Randomly generated regression dataset");
        
        return dataset;
    }
    
    /**
     * Generate the XOR dataset (non-linearly separable).
     * 
     * @param nSamples samples per quadrant
     * @param noise noise level
     * @param seed random seed
     * @return XOR dataset
     */
    public static Dataset makeXOR(int nSamples, double noise, long seed) {
        Random random = new Random(seed);
        int total = nSamples * 4;
        
        double[][] features = new double[total][2];
        int[] labels = new int[total];
        
        int idx = 0;
        for (int q = 0; q < 4; q++) {
            double cx = (q % 2 == 0) ? -1 : 1;
            double cy = (q < 2) ? -1 : 1;
            int label = (q == 0 || q == 3) ? 0 : 1;
            
            for (int i = 0; i < nSamples; i++) {
                features[idx][0] = cx + random.nextGaussian() * noise;
                features[idx][1] = cy + random.nextGaussian() * noise;
                labels[idx] = label;
                idx++;
            }
        }
        
        Dataset dataset = new Dataset(features, labels);
        dataset.setName("XOR Dataset");
        dataset.setDescription("Non-linearly separable XOR pattern");
        dataset.setFeatureNames(new String[]{"x", "y"});
        dataset.setTargetNames(new String[]{"class_0", "class_1"});
        
        return dataset;
    }
    
    /**
     * Generate concentric circles dataset.
     * 
     * @param nSamples samples per circle
     * @param noise noise level
     * @param seed random seed
     * @return circles dataset
     */
    public static Dataset makeCircles(int nSamples, double noise, long seed) {
        Random random = new Random(seed);
        int total = nSamples * 2;
        
        double[][] features = new double[total][2];
        int[] labels = new int[total];
        
        for (int i = 0; i < nSamples; i++) {
            // Inner circle
            double angle = random.nextDouble() * 2 * Math.PI;
            features[i][0] = 0.5 * Math.cos(angle) + random.nextGaussian() * noise;
            features[i][1] = 0.5 * Math.sin(angle) + random.nextGaussian() * noise;
            labels[i] = 0;
            
            // Outer circle
            angle = random.nextDouble() * 2 * Math.PI;
            features[nSamples + i][0] = Math.cos(angle) + random.nextGaussian() * noise;
            features[nSamples + i][1] = Math.sin(angle) + random.nextGaussian() * noise;
            labels[nSamples + i] = 1;
        }
        
        Dataset dataset = new Dataset(features, labels);
        dataset.setName("Circles Dataset");
        dataset.setDescription("Concentric circles - non-linearly separable");
        dataset.setFeatureNames(new String[]{"x", "y"});
        dataset.setTargetNames(new String[]{"inner", "outer"});
        
        return dataset;
    }
}
