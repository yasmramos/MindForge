package io.github.yasmramos.mindforge.neural;

import java.io.Serializable;

/**
 * 2D Max Pooling layer for Convolutional Neural Networks.
 * 
 * Performs max pooling operation over spatial dimensions to reduce 
 * the spatial size of feature maps while retaining important features.
 * 
 * Input shape: [channels, height, width] flattened to 1D array
 * Output shape: [channels, outHeight, outWidth] flattened to 1D array
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
public class MaxPooling2DLayer extends Layer implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // Layer configuration
    private final int inputChannels;
    private final int inputHeight;
    private final int inputWidth;
    private final int poolSize;
    private final int stride;
    
    // Output dimensions
    private final int outputHeight;
    private final int outputWidth;
    
    // Cache for backpropagation (stores indices of max values)
    private int[][][] maxIndicesH;  // [channels][outHeight][outWidth]
    private int[][][] maxIndicesW;  // [channels][outHeight][outWidth]
    private double[][][] lastInput3D;
    
    /**
     * Creates a 2D max pooling layer.
     *
     * @param inputChannels number of input channels
     * @param inputHeight height of input feature maps
     * @param inputWidth width of input feature maps
     * @param poolSize size of the pooling window (square)
     */
    public MaxPooling2DLayer(int inputChannels, int inputHeight, int inputWidth, int poolSize) {
        this(inputChannels, inputHeight, inputWidth, poolSize, poolSize);
    }
    
    /**
     * Creates a 2D max pooling layer with custom stride.
     *
     * @param inputChannels number of input channels
     * @param inputHeight height of input feature maps
     * @param inputWidth width of input feature maps
     * @param poolSize size of the pooling window (square)
     * @param stride stride of the pooling operation
     */
    public MaxPooling2DLayer(int inputChannels, int inputHeight, int inputWidth,
                             int poolSize, int stride) {
        validateParameters(inputChannels, inputHeight, inputWidth, poolSize, stride);
        
        this.inputChannels = inputChannels;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.poolSize = poolSize;
        this.stride = stride;
        
        // Calculate output dimensions
        this.outputHeight = (inputHeight - poolSize) / stride + 1;
        this.outputWidth = (inputWidth - poolSize) / stride + 1;
        
        // Set input/output sizes for Layer interface
        this.inputSize = inputChannels * inputHeight * inputWidth;
        this.outputSize = inputChannels * outputHeight * outputWidth;
    }
    
    private void validateParameters(int inputChannels, int inputHeight, int inputWidth,
                                   int poolSize, int stride) {
        if (inputChannels <= 0) {
            throw new IllegalArgumentException("inputChannels must be positive");
        }
        if (inputHeight <= 0 || inputWidth <= 0) {
            throw new IllegalArgumentException("Input dimensions must be positive");
        }
        if (poolSize <= 0) {
            throw new IllegalArgumentException("poolSize must be positive");
        }
        if (stride <= 0) {
            throw new IllegalArgumentException("stride must be positive");
        }
        if (poolSize > inputHeight || poolSize > inputWidth) {
            throw new IllegalArgumentException("poolSize cannot be larger than input dimensions");
        }
    }
    
    @Override
    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException(
                "Input size mismatch: expected " + inputSize + ", got " + input.length);
        }
        
        // Reshape input to 3D
        lastInput3D = reshape1Dto3D(input, inputChannels, inputHeight, inputWidth);
        
        // Initialize max indices arrays for backpropagation
        maxIndicesH = new int[inputChannels][outputHeight][outputWidth];
        maxIndicesW = new int[inputChannels][outputHeight][outputWidth];
        
        // Perform max pooling
        double[][][] output3D = new double[inputChannels][outputHeight][outputWidth];
        
        for (int c = 0; c < inputChannels; c++) {
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    double maxVal = Double.NEGATIVE_INFINITY;
                    int maxH = 0, maxW = 0;
                    
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            
                            if (lastInput3D[c][ih][iw] > maxVal) {
                                maxVal = lastInput3D[c][ih][iw];
                                maxH = ih;
                                maxW = iw;
                            }
                        }
                    }
                    
                    output3D[c][oh][ow] = maxVal;
                    maxIndicesH[c][oh][ow] = maxH;
                    maxIndicesW[c][oh][ow] = maxW;
                }
            }
        }
        
        // Flatten output to 1D
        lastOutput = reshape3Dto1D(output3D);
        return lastOutput;
    }
    
    @Override
    public double[] backward(double[] gradOutput, double learningRate) {
        // Reshape gradient to 3D
        double[][][] gradOutput3D = reshape1Dto3D(gradOutput, inputChannels, outputHeight, outputWidth);
        
        // Initialize gradient for input
        double[][][] gradInput3D = new double[inputChannels][inputHeight][inputWidth];
        
        // Propagate gradients only to max positions
        for (int c = 0; c < inputChannels; c++) {
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    int maxH = maxIndicesH[c][oh][ow];
                    int maxW = maxIndicesW[c][oh][ow];
                    gradInput3D[c][maxH][maxW] += gradOutput3D[c][oh][ow];
                }
            }
        }
        
        return reshape3Dto1D(gradInput3D);
    }
    
    // Helper methods for tensor operations
    
    private double[][][] reshape1Dto3D(double[] array, int channels, int height, int width) {
        double[][][] result = new double[channels][height][width];
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result[c][h][w] = array[idx++];
                }
            }
        }
        return result;
    }
    
    private double[] reshape3Dto1D(double[][][] tensor) {
        int channels = tensor.length;
        int height = tensor[0].length;
        int width = tensor[0][0].length;
        double[] result = new double[channels * height * width];
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result[idx++] = tensor[c][h][w];
                }
            }
        }
        return result;
    }
    
    // Getters
    
    public int getInputChannels() { return inputChannels; }
    public int getInputHeight() { return inputHeight; }
    public int getInputWidth() { return inputWidth; }
    public int getPoolSize() { return poolSize; }
    public int getStride() { return stride; }
    public int getOutputHeight() { return outputHeight; }
    public int getOutputWidth() { return outputWidth; }
    
    @Override
    public String toString() {
        return String.format("MaxPooling2D(pool=%dx%d, stride=%d) -> [%d, %d, %d]",
            poolSize, poolSize, stride, inputChannels, outputHeight, outputWidth);
    }
}
