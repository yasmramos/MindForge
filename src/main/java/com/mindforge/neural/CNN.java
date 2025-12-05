package com.mindforge.neural;

import java.io.Serializable;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Convolutional Neural Network (CNN) for image classification.
 * 
 * Supports building networks with:
 * - 2D Convolutional layers
 * - Max and Average Pooling layers
 * - Batch Normalization
 * - Dropout regularization
 * - Fully connected (Dense) layers
 * 
 * Example usage:
 * <pre>
 * CNN cnn = new CNN.Builder()
 *     .inputShape(1, 28, 28)  // Grayscale 28x28 images
 *     .addConv2D(32, 3, 1, 1, ActivationFunction.RELU)
 *     .addMaxPooling(2)
 *     .addConv2D(64, 3, 1, 1, ActivationFunction.RELU)
 *     .addMaxPooling(2)
 *     .addFlatten()
 *     .addDense(128, ActivationFunction.RELU)
 *     .addDense(10, ActivationFunction.SOFTMAX)
 *     .learningRate(0.001)
 *     .epochs(10)
 *     .batchSize(32)
 *     .build();
 * 
 * cnn.fit(trainImages, trainLabels);
 * int[] predictions = cnn.predict(testImages);
 * </pre>
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
public class CNN implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private List<Layer> layers;
    private double learningRate;
    private int epochs;
    private int batchSize;
    private boolean verbose;
    private List<Double> trainingLoss;
    private List<Double> validationLoss;
    private List<Double> trainingAccuracy;
    private Long randomSeed;
    
    // Input shape
    private int inputChannels;
    private int inputHeight;
    private int inputWidth;
    
    private CNN() {
        this.layers = new ArrayList<>();
        this.trainingLoss = new ArrayList<>();
        this.validationLoss = new ArrayList<>();
        this.trainingAccuracy = new ArrayList<>();
    }
    
    /**
     * Builder for constructing CNN architectures.
     */
    public static class Builder {
        private int inputChannels = 1;
        private int inputHeight = 28;
        private int inputWidth = 28;
        private double learningRate = 0.001;
        private int epochs = 10;
        private int batchSize = 32;
        private boolean verbose = true;
        private Long randomSeed = null;
        
        // Track current layer output shape
        private int currentChannels;
        private int currentHeight;
        private int currentWidth;
        private boolean isFlattened = false;
        private int flattenedSize;
        
        private List<LayerConfig> layerConfigs = new ArrayList<>();
        
        // Internal class to hold layer configuration
        private static class LayerConfig {
            String type;
            Object[] params;
            
            LayerConfig(String type, Object... params) {
                this.type = type;
                this.params = params;
            }
        }
        
        /**
         * Set the input shape.
         *
         * @param channels number of channels
         * @param height image height
         * @param width image width
         * @return this builder
         */
        public Builder inputShape(int channels, int height, int width) {
            this.inputChannels = channels;
            this.inputHeight = height;
            this.inputWidth = width;
            this.currentChannels = channels;
            this.currentHeight = height;
            this.currentWidth = width;
            return this;
        }
        
        /**
         * Add a 2D convolutional layer.
         *
         * @param filters number of filters
         * @param kernelSize size of the kernel
         * @param stride stride of convolution
         * @param padding zero-padding
         * @param activation activation function
         * @return this builder
         */
        public Builder addConv2D(int filters, int kernelSize, int stride, int padding,
                                 ActivationFunction activation) {
            if (isFlattened) {
                throw new IllegalStateException("Cannot add Conv2D after Flatten layer");
            }
            layerConfigs.add(new LayerConfig("conv2d", currentChannels, currentHeight, 
                currentWidth, filters, kernelSize, stride, padding, activation));
            
            // Update current shape
            currentHeight = (currentHeight + 2 * padding - kernelSize) / stride + 1;
            currentWidth = (currentWidth + 2 * padding - kernelSize) / stride + 1;
            currentChannels = filters;
            return this;
        }
        
        /**
         * Add a 2D convolutional layer with default stride=1, padding=0.
         *
         * @param filters number of filters
         * @param kernelSize size of the kernel
         * @param activation activation function
         * @return this builder
         */
        public Builder addConv2D(int filters, int kernelSize, ActivationFunction activation) {
            return addConv2D(filters, kernelSize, 1, 0, activation);
        }
        
        /**
         * Add a max pooling layer.
         *
         * @param poolSize size of pooling window
         * @param stride stride (defaults to poolSize)
         * @return this builder
         */
        public Builder addMaxPooling(int poolSize, int stride) {
            if (isFlattened) {
                throw new IllegalStateException("Cannot add MaxPooling after Flatten layer");
            }
            layerConfigs.add(new LayerConfig("maxpool", currentChannels, currentHeight,
                currentWidth, poolSize, stride));
            
            // Update current shape
            currentHeight = (currentHeight - poolSize) / stride + 1;
            currentWidth = (currentWidth - poolSize) / stride + 1;
            return this;
        }
        
        /**
         * Add a max pooling layer with stride equal to pool size.
         *
         * @param poolSize size of pooling window
         * @return this builder
         */
        public Builder addMaxPooling(int poolSize) {
            return addMaxPooling(poolSize, poolSize);
        }
        
        /**
         * Add an average pooling layer.
         *
         * @param poolSize size of pooling window
         * @param stride stride (defaults to poolSize)
         * @return this builder
         */
        public Builder addAveragePooling(int poolSize, int stride) {
            if (isFlattened) {
                throw new IllegalStateException("Cannot add AveragePooling after Flatten layer");
            }
            layerConfigs.add(new LayerConfig("avgpool", currentChannels, currentHeight,
                currentWidth, poolSize, stride));
            
            // Update current shape
            currentHeight = (currentHeight - poolSize) / stride + 1;
            currentWidth = (currentWidth - poolSize) / stride + 1;
            return this;
        }
        
        /**
         * Add an average pooling layer with stride equal to pool size.
         *
         * @param poolSize size of pooling window
         * @return this builder
         */
        public Builder addAveragePooling(int poolSize) {
            return addAveragePooling(poolSize, poolSize);
        }
        
        /**
         * Add a flatten layer to transition to dense layers.
         *
         * @return this builder
         */
        public Builder addFlatten() {
            if (isFlattened) {
                throw new IllegalStateException("Already flattened");
            }
            layerConfigs.add(new LayerConfig("flatten", currentChannels, currentHeight, currentWidth));
            flattenedSize = currentChannels * currentHeight * currentWidth;
            isFlattened = true;
            return this;
        }
        
        /**
         * Add a dense (fully connected) layer.
         *
         * @param units number of neurons
         * @param activation activation function
         * @return this builder
         */
        public Builder addDense(int units, ActivationFunction activation) {
            if (!isFlattened) {
                // Auto-flatten
                addFlatten();
            }
            layerConfigs.add(new LayerConfig("dense", flattenedSize, units, activation));
            flattenedSize = units;
            return this;
        }
        
        /**
         * Add a dropout layer.
         *
         * @param rate dropout rate (0-1)
         * @return this builder
         */
        public Builder addDropout(double rate) {
            if (!isFlattened) {
                layerConfigs.add(new LayerConfig("dropout2d", 
                    currentChannels * currentHeight * currentWidth, rate));
            } else {
                layerConfigs.add(new LayerConfig("dropout", flattenedSize, rate));
            }
            return this;
        }
        
        /**
         * Add a batch normalization layer.
         *
         * @return this builder
         */
        public Builder addBatchNorm() {
            if (!isFlattened) {
                layerConfigs.add(new LayerConfig("batchnorm2d", 
                    currentChannels * currentHeight * currentWidth));
            } else {
                layerConfigs.add(new LayerConfig("batchnorm", flattenedSize));
            }
            return this;
        }
        
        /**
         * Set learning rate.
         *
         * @param lr learning rate
         * @return this builder
         */
        public Builder learningRate(double lr) {
            this.learningRate = lr;
            return this;
        }
        
        /**
         * Set number of training epochs.
         *
         * @param epochs number of epochs
         * @return this builder
         */
        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }
        
        /**
         * Set mini-batch size.
         *
         * @param batchSize batch size
         * @return this builder
         */
        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }
        
        /**
         * Set verbose mode.
         *
         * @param verbose whether to print progress
         * @return this builder
         */
        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }
        
        /**
         * Set random seed for reproducibility.
         *
         * @param seed random seed
         * @return this builder
         */
        public Builder randomSeed(long seed) {
            this.randomSeed = seed;
            return this;
        }
        
        /**
         * Build the CNN.
         *
         * @return the constructed CNN
         */
        public CNN build() {
            CNN cnn = new CNN();
            cnn.inputChannels = inputChannels;
            cnn.inputHeight = inputHeight;
            cnn.inputWidth = inputWidth;
            cnn.learningRate = learningRate;
            cnn.epochs = epochs;
            cnn.batchSize = batchSize;
            cnn.verbose = verbose;
            cnn.randomSeed = randomSeed;
            
            long seed = randomSeed != null ? randomSeed : System.currentTimeMillis();
            
            // Build layers from configurations
            for (LayerConfig config : layerConfigs) {
                Layer layer = createLayer(config, seed++);
                cnn.layers.add(layer);
            }
            
            return cnn;
        }
        
        private Layer createLayer(LayerConfig config, long seed) {
            switch (config.type) {
                case "conv2d":
                    return new Conv2DLayer(
                        (int) config.params[0],  // inputChannels
                        (int) config.params[1],  // inputHeight
                        (int) config.params[2],  // inputWidth
                        (int) config.params[3],  // numFilters
                        (int) config.params[4],  // kernelSize
                        (int) config.params[5],  // stride
                        (int) config.params[6],  // padding
                        (ActivationFunction) config.params[7],  // activation
                        seed
                    );
                    
                case "maxpool":
                    return new MaxPooling2DLayer(
                        (int) config.params[0],  // inputChannels
                        (int) config.params[1],  // inputHeight
                        (int) config.params[2],  // inputWidth
                        (int) config.params[3],  // poolSize
                        (int) config.params[4]   // stride
                    );
                    
                case "avgpool":
                    return new AveragePooling2DLayer(
                        (int) config.params[0],  // inputChannels
                        (int) config.params[1],  // inputHeight
                        (int) config.params[2],  // inputWidth
                        (int) config.params[3],  // poolSize
                        (int) config.params[4]   // stride
                    );
                    
                case "flatten":
                    return new FlattenLayer(
                        (int) config.params[0],  // inputChannels
                        (int) config.params[1],  // inputHeight
                        (int) config.params[2]   // inputWidth
                    );
                    
                case "dense":
                    return new DenseLayer(
                        (int) config.params[0],  // inputSize
                        (int) config.params[1],  // outputSize
                        (ActivationFunction) config.params[2],  // activation
                        seed
                    );
                    
                case "dropout":
                case "dropout2d":
                    return new DropoutLayer(
                        (int) config.params[0],  // inputSize
                        (double) config.params[1] // rate
                    );
                    
                case "batchnorm":
                case "batchnorm2d":
                    return new BatchNormLayer(
                        (int) config.params[0]   // inputSize
                    );
                    
                default:
                    throw new IllegalArgumentException("Unknown layer type: " + config.type);
            }
        }
    }
    
    /**
     * Forward pass through the network.
     *
     * @param input input image flattened (channels * height * width)
     * @return output predictions
     */
    public double[] forward(double[] input) {
        double[] current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }
    
    /**
     * Backward pass through the network.
     *
     * @param gradOutput gradient from loss function
     */
    private void backward(double[] gradOutput) {
        double[] grad = gradOutput;
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers.get(i).backward(grad, learningRate);
        }
    }
    
    /**
     * Train the CNN for classification.
     *
     * @param X training images (each row is a flattened image)
     * @param y training labels (class indices)
     */
    public void fit(double[][] X, int[] y) {
        fit(X, y, null, null);
    }
    
    /**
     * Train the CNN for classification with validation data.
     *
     * @param X training images
     * @param y training labels
     * @param XVal validation images
     * @param yVal validation labels
     */
    public void fit(double[][] X, int[] y, double[][] XVal, int[] yVal) {
        int nSamples = X.length;
        int nClasses = 0;
        for (int label : y) {
            if (label >= nClasses) nClasses = label + 1;
        }
        
        // Convert labels to one-hot encoding
        double[][] yOneHot = new double[nSamples][nClasses];
        for (int i = 0; i < nSamples; i++) {
            yOneHot[i][y[i]] = 1.0;
        }
        
        Random random = randomSeed != null ? new Random(randomSeed) : new Random();
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) indices[i] = i;
        
        // Set training mode for dropout layers
        setTrainingMode(true);
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle indices
            for (int i = nSamples - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
            
            double epochLoss = 0.0;
            int correct = 0;
            
            // Mini-batch training
            for (int batch = 0; batch < nSamples; batch += batchSize) {
                int batchEnd = Math.min(batch + batchSize, nSamples);
                
                for (int i = batch; i < batchEnd; i++) {
                    int idx = indices[i];
                    
                    // Forward pass
                    double[] output = forward(X[idx]);
                    
                    // Calculate cross-entropy loss gradient
                    double[] gradOutput = new double[output.length];
                    for (int j = 0; j < output.length; j++) {
                        gradOutput[j] = output[j] - yOneHot[idx][j];
                        epochLoss -= yOneHot[idx][j] * Math.log(Math.max(output[j], 1e-15));
                    }
                    
                    // Calculate accuracy
                    if (argmax(output) == y[idx]) {
                        correct++;
                    }
                    
                    // Backward pass
                    backward(gradOutput);
                }
            }
            
            epochLoss /= nSamples;
            double accuracy = (double) correct / nSamples;
            trainingLoss.add(epochLoss);
            trainingAccuracy.add(accuracy);
            
            // Validation loss
            if (XVal != null && yVal != null) {
                setTrainingMode(false);
                double valLoss = calculateLoss(XVal, yVal);
                double valAccuracy = score(XVal, yVal);
                validationLoss.add(valLoss);
                setTrainingMode(true);
                
                if (verbose) {
                    System.out.printf("Epoch %d/%d - Loss: %.4f - Acc: %.4f - Val Loss: %.4f - Val Acc: %.4f%n",
                            epoch + 1, epochs, epochLoss, accuracy, valLoss, valAccuracy);
                }
            } else if (verbose) {
                System.out.printf("Epoch %d/%d - Loss: %.4f - Acc: %.4f%n", 
                    epoch + 1, epochs, epochLoss, accuracy);
            }
        }
        
        // Set to inference mode
        setTrainingMode(false);
    }
    
    /**
     * Predict class labels.
     *
     * @param X images
     * @return predicted class labels
     */
    public int[] predict(double[][] X) {
        setTrainingMode(false);
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            double[] output = forward(X[i]);
            predictions[i] = argmax(output);
        }
        return predictions;
    }
    
    /**
     * Predict class probabilities.
     *
     * @param X images
     * @return predicted probabilities for each class
     */
    public double[][] predictProba(double[][] X) {
        setTrainingMode(false);
        double[][] probabilities = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            probabilities[i] = forward(X[i]);
        }
        return probabilities;
    }
    
    /**
     * Calculate the classification accuracy.
     *
     * @param X images
     * @param y true labels
     * @return accuracy
     */
    public double score(double[][] X, int[] y) {
        int[] predictions = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
    }
    
    private double calculateLoss(double[][] X, int[] y) {
        double loss = 0.0;
        for (int i = 0; i < X.length; i++) {
            double[] output = forward(X[i]);
            loss -= Math.log(Math.max(output[y[i]], 1e-15));
        }
        return loss / X.length;
    }
    
    private int argmax(double[] array) {
        int maxIdx = 0;
        double maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    private void setTrainingMode(boolean training) {
        for (Layer layer : layers) {
            if (layer instanceof DropoutLayer) {
                ((DropoutLayer) layer).setTraining(training);
            }
        }
    }
    
    /**
     * Get training loss history.
     *
     * @return list of training losses per epoch
     */
    public List<Double> getTrainingLoss() {
        return trainingLoss;
    }
    
    /**
     * Get validation loss history.
     *
     * @return list of validation losses per epoch
     */
    public List<Double> getValidationLoss() {
        return validationLoss;
    }
    
    /**
     * Get training accuracy history.
     *
     * @return list of training accuracies per epoch
     */
    public List<Double> getTrainingAccuracy() {
        return trainingAccuracy;
    }
    
    /**
     * Get the layers.
     *
     * @return list of layers
     */
    public List<Layer> getLayers() {
        return layers;
    }
    
    /**
     * Get input shape as an array [channels, height, width].
     *
     * @return input shape
     */
    public int[] getInputShape() {
        return new int[]{inputChannels, inputHeight, inputWidth};
    }
    
    /**
     * Set verbose mode.
     *
     * @param verbose whether to print progress
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
    
    /**
     * Set learning rate.
     *
     * @param learningRate new learning rate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    /**
     * Get total number of trainable parameters.
     *
     * @return number of parameters
     */
    public int getNumParameters() {
        int total = 0;
        for (Layer layer : layers) {
            if (layer instanceof Conv2DLayer) {
                total += ((Conv2DLayer) layer).getNumParameters();
            } else if (layer instanceof DenseLayer) {
                DenseLayer dense = (DenseLayer) layer;
                total += dense.getInputSize() * dense.getOutputSize() + dense.getOutputSize();
            }
        }
        return total;
    }
    
    /**
     * Print network architecture summary.
     */
    public void summary() {
        System.out.println("CNN Architecture Summary");
        System.out.println("========================");
        System.out.printf("Input shape: [%d, %d, %d]%n", inputChannels, inputHeight, inputWidth);
        System.out.println();
        
        for (int i = 0; i < layers.size(); i++) {
            System.out.printf("Layer %d: %s%n", i + 1, layers.get(i).toString());
        }
        
        System.out.println();
        System.out.printf("Total parameters: %,d%n", getNumParameters());
    }
    
    /**
     * Save the model to a file.
     *
     * @param filename path to save file
     * @throws IOException if save fails
     */
    public void save(String filename) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(this);
        }
    }
    
    /**
     * Load a model from a file.
     *
     * @param filename path to saved file
     * @return loaded CNN
     * @throws IOException if load fails
     * @throws ClassNotFoundException if class not found
     */
    public static CNN load(String filename) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            return (CNN) ois.readObject();
        }
    }
    
    @Override
    public String toString() {
        return String.format("CNN(layers=%d, params=%,d)", layers.size(), getNumParameters());
    }
}
