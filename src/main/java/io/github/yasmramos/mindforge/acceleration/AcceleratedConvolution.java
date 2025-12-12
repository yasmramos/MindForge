package io.github.yasmramos.mindforge.acceleration;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.stream.IntStream;

/**
 * High-performance convolution operations for CNNs.
 * 
 * Provides optimized implementations of 2D convolution and related operations
 * using parallelization, im2col transformation, and cache-efficient algorithms.
 * 
 * <p>Supports both standard convolution and transposed convolution for
 * forward and backward passes in convolutional neural networks.</p>
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
public class AcceleratedConvolution {
    
    private static final AccelerationConfig config = AccelerationConfig.getInstance();
    private static final MemoryPool pool = MemoryPool.getInstance();
    
    // Private constructor - utility class
    private AcceleratedConvolution() {}
    
    // ==================== Forward Convolution ====================
    
    /**
     * Perform 2D convolution: output = input * filters + biases
     * 
     * Uses im2col transformation for efficient matrix multiplication-based convolution.
     * 
     * @param input input feature maps [channels][height][width]
     * @param filters convolutional filters [numFilters][channels][kH][kW]
     * @param biases filter biases [numFilters]
     * @param stride convolution stride
     * @param padding zero-padding amount
     * @return output feature maps [numFilters][outHeight][outWidth]
     */
    public static double[][][] conv2d(double[][][] input, double[][][][] filters, 
                                       double[] biases, int stride, int padding) {
        int channels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = filters.length;
        int kernelHeight = filters[0][0].length;
        int kernelWidth = filters[0][0][0].length;
        
        int outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;
        
        double[][][] output = new double[numFilters][outputHeight][outputWidth];
        
        // Determine execution strategy based on output size
        int totalOutputElements = numFilters * outputHeight * outputWidth;
        
        if (config.isParallelizationEnabled() && 
            totalOutputElements >= config.getConvolutionThreshold() * config.getConvolutionThreshold()) {
            conv2dParallel(input, filters, biases, output, stride, padding);
        } else {
            conv2dSequential(input, filters, biases, output, stride, padding);
        }
        
        return output;
    }
    
    /**
     * Sequential convolution implementation.
     */
    private static void conv2dSequential(double[][][] input, double[][][][] filters,
                                          double[] biases, double[][][] output,
                                          int stride, int padding) {
        int channels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = filters.length;
        int kernelHeight = filters[0][0].length;
        int kernelWidth = filters[0][0][0].length;
        int outputHeight = output[0].length;
        int outputWidth = output[0][0].length;
        
        // Apply padding if needed
        double[][][] paddedInput = padding > 0 ? pad(input, padding) : input;
        
        // Perform convolution
        for (int f = 0; f < numFilters; f++) {
            double[][][] filter = filters[f];
            double bias = biases[f];
            
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    double sum = bias;
                    
                    // Compute convolution for this output position
                    for (int c = 0; c < channels; c++) {
                        for (int kh = 0; kh < kernelHeight; kh++) {
                            int ih = oh * stride + kh;
                            double[] inputRow = paddedInput[c][ih];
                            double[] filterRow = filter[c][kh];
                            
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                int iw = ow * stride + kw;
                                sum += inputRow[iw] * filterRow[kw];
                            }
                        }
                    }
                    
                    output[f][oh][ow] = sum;
                }
            }
        }
    }
    
    /**
     * Parallel convolution implementation.
     */
    private static void conv2dParallel(double[][][] input, double[][][][] filters,
                                        double[] biases, double[][][] output,
                                        int stride, int padding) {
        int channels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        int numFilters = filters.length;
        int kernelHeight = filters[0][0].length;
        int kernelWidth = filters[0][0][0].length;
        int outputHeight = output[0].length;
        int outputWidth = output[0][0].length;
        
        // Apply padding if needed
        final double[][][] paddedInput = padding > 0 ? pad(input, padding) : input;
        
        ForkJoinPool executor = config.getExecutor();
        
        // Parallelize over filters
        executor.submit(() ->
            IntStream.range(0, numFilters).parallel().forEach(f -> {
                double[][][] filter = filters[f];
                double bias = biases[f];
                
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        double sum = bias;
                        
                        for (int c = 0; c < channels; c++) {
                            for (int kh = 0; kh < kernelHeight; kh++) {
                                int ih = oh * stride + kh;
                                double[] inputRow = paddedInput[c][ih];
                                double[] filterRow = filter[c][kh];
                                
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    int iw = ow * stride + kw;
                                    sum += inputRow[iw] * filterRow[kw];
                                }
                            }
                        }
                        
                        output[f][oh][ow] = sum;
                    }
                }
            })
        ).join();
    }
    
    // ==================== Im2Col Transformation ====================
    
    /**
     * Transform input into column matrix for efficient convolution via matrix multiplication.
     * 
     * This is the im2col algorithm that reshapes local patches of the input
     * into columns of a matrix, allowing convolution to be performed as matrix multiplication.
     * 
     * @param input input feature maps [channels][height][width]
     * @param kernelHeight kernel height
     * @param kernelWidth kernel width
     * @param stride convolution stride
     * @param padding zero-padding amount
     * @return column matrix [kernelHeight * kernelWidth * channels][outputHeight * outputWidth]
     */
    public static double[][] im2col(double[][][] input, int kernelHeight, int kernelWidth,
                                    int stride, int padding) {
        int channels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        
        int outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;
        
        int colRows = channels * kernelHeight * kernelWidth;
        int colCols = outputHeight * outputWidth;
        
        double[][] cols = pool.acquire2D(colRows, colCols);
        
        double[][][] paddedInput = padding > 0 ? pad(input, padding) : input;
        
        if (config.shouldParallelize(colRows * colCols)) {
            im2colParallel(paddedInput, cols, kernelHeight, kernelWidth, stride, 
                          outputHeight, outputWidth, channels);
        } else {
            im2colSequential(paddedInput, cols, kernelHeight, kernelWidth, stride,
                            outputHeight, outputWidth, channels);
        }
        
        return cols;
    }
    
    private static void im2colSequential(double[][][] input, double[][] cols,
                                          int kH, int kW, int stride,
                                          int outH, int outW, int channels) {
        int colIdx = 0;
        
        for (int c = 0; c < channels; c++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int outIdx = 0;
                    
                    for (int oh = 0; oh < outH; oh++) {
                        int ih = oh * stride + kh;
                        for (int ow = 0; ow < outW; ow++) {
                            int iw = ow * stride + kw;
                            cols[colIdx][outIdx++] = input[c][ih][iw];
                        }
                    }
                    
                    colIdx++;
                }
            }
        }
    }
    
    private static void im2colParallel(double[][][] input, double[][] cols,
                                        int kH, int kW, int stride,
                                        int outH, int outW, int channels) {
        ForkJoinPool executor = config.getExecutor();
        
        executor.submit(() ->
            IntStream.range(0, channels).parallel().forEach(c -> {
                int baseIdx = c * kH * kW;
                
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int colIdx = baseIdx + kh * kW + kw;
                        int outIdx = 0;
                        
                        for (int oh = 0; oh < outH; oh++) {
                            int ih = oh * stride + kh;
                            for (int ow = 0; ow < outW; ow++) {
                                int iw = ow * stride + kw;
                                cols[colIdx][outIdx++] = input[c][ih][iw];
                            }
                        }
                    }
                }
            })
        ).join();
    }
    
    /**
     * Transform column matrix back to image format (col2im).
     * Used in backpropagation through convolution layers.
     * 
     * @param cols column matrix
     * @param channels number of channels
     * @param height output height
     * @param width output width
     * @param kernelHeight kernel height
     * @param kernelWidth kernel width
     * @param stride convolution stride
     * @param padding zero-padding amount
     * @return reconstructed feature maps [channels][height][width]
     */
    public static double[][][] col2im(double[][] cols, int channels, int height, int width,
                                       int kernelHeight, int kernelWidth,
                                       int stride, int padding) {
        int paddedHeight = height + 2 * padding;
        int paddedWidth = width + 2 * padding;
        int outH = (paddedHeight - kernelHeight) / stride + 1;
        int outW = (paddedWidth - kernelWidth) / stride + 1;
        
        double[][][] result = new double[channels][paddedHeight][paddedWidth];
        
        int colIdx = 0;
        for (int c = 0; c < channels; c++) {
            for (int kh = 0; kh < kernelHeight; kh++) {
                for (int kw = 0; kw < kernelWidth; kw++) {
                    int outIdx = 0;
                    
                    for (int oh = 0; oh < outH; oh++) {
                        int ih = oh * stride + kh;
                        for (int ow = 0; ow < outW; ow++) {
                            int iw = ow * stride + kw;
                            result[c][ih][iw] += cols[colIdx][outIdx++];
                        }
                    }
                    
                    colIdx++;
                }
            }
        }
        
        // Remove padding
        if (padding > 0) {
            return unpad(result, padding);
        }
        
        return result;
    }
    
    // ==================== Backward Convolution ====================
    
    /**
     * Compute gradient with respect to input for convolution layer.
     * 
     * @param gradOutput gradient of loss with respect to output [numFilters][outH][outW]
     * @param filters convolutional filters [numFilters][channels][kH][kW]
     * @param inputShape original input shape [channels, height, width]
     * @param stride convolution stride
     * @param padding zero-padding amount
     * @return gradient with respect to input [channels][height][width]
     */
    public static double[][][] conv2dBackwardInput(double[][][] gradOutput, double[][][][] filters,
                                                    int[] inputShape, int stride, int padding) {
        int channels = inputShape[0];
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];
        int numFilters = filters.length;
        int kernelHeight = filters[0][0].length;
        int kernelWidth = filters[0][0][0].length;
        int outH = gradOutput[0].length;
        int outW = gradOutput[0][0].length;
        
        // Initialize padded gradient
        int paddedHeight = inputHeight + 2 * padding;
        int paddedWidth = inputWidth + 2 * padding;
        double[][][] gradPadded = new double[channels][paddedHeight][paddedWidth];
        
        // Compute gradient (transposed convolution)
        if (config.isParallelizationEnabled() && numFilters * outH * outW >= 1000) {
            ForkJoinPool executor = config.getExecutor();
            
            // Use thread-local accumulators to avoid synchronization
            double[][][][] localGrads = new double[config.getNumThreads()][channels][paddedHeight][paddedWidth];
            
            executor.submit(() ->
                IntStream.range(0, numFilters).parallel().forEach(f -> {
                    int threadId = (int) (Thread.currentThread().getId() % config.getNumThreads());
                    double[][][] localGrad = localGrads[threadId];
                    double[][][] filter = filters[f];
                    
                    for (int oh = 0; oh < outH; oh++) {
                        for (int ow = 0; ow < outW; ow++) {
                            double grad = gradOutput[f][oh][ow];
                            
                            for (int c = 0; c < channels; c++) {
                                for (int kh = 0; kh < kernelHeight; kh++) {
                                    int ih = oh * stride + kh;
                                    for (int kw = 0; kw < kernelWidth; kw++) {
                                        int iw = ow * stride + kw;
                                        localGrad[c][ih][iw] += grad * filter[c][kh][kw];
                                    }
                                }
                            }
                        }
                    }
                })
            ).join();
            
            // Merge local gradients
            for (int t = 0; t < config.getNumThreads(); t++) {
                for (int c = 0; c < channels; c++) {
                    for (int h = 0; h < paddedHeight; h++) {
                        for (int w = 0; w < paddedWidth; w++) {
                            gradPadded[c][h][w] += localGrads[t][c][h][w];
                        }
                    }
                }
            }
        } else {
            for (int f = 0; f < numFilters; f++) {
                double[][][] filter = filters[f];
                
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        double grad = gradOutput[f][oh][ow];
                        
                        for (int c = 0; c < channels; c++) {
                            for (int kh = 0; kh < kernelHeight; kh++) {
                                int ih = oh * stride + kh;
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    int iw = ow * stride + kw;
                                    gradPadded[c][ih][iw] += grad * filter[c][kh][kw];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Remove padding
        if (padding > 0) {
            return unpad(gradPadded, padding);
        }
        
        return gradPadded;
    }
    
    /**
     * Compute gradient with respect to filters for convolution layer.
     * 
     * @param gradOutput gradient of loss with respect to output [numFilters][outH][outW]
     * @param input original input [channels][height][width]
     * @param filterShape filter shape [numFilters, channels, kH, kW]
     * @param stride convolution stride
     * @param padding zero-padding amount
     * @return gradient with respect to filters [numFilters][channels][kH][kW]
     */
    public static double[][][][] conv2dBackwardFilters(double[][][] gradOutput, double[][][] input,
                                                        int[] filterShape, int stride, int padding) {
        int numFilters = filterShape[0];
        int channels = filterShape[1];
        int kernelHeight = filterShape[2];
        int kernelWidth = filterShape[3];
        int outH = gradOutput[0].length;
        int outW = gradOutput[0][0].length;
        
        double[][][][] gradFilters = new double[numFilters][channels][kernelHeight][kernelWidth];
        
        double[][][] paddedInput = padding > 0 ? pad(input, padding) : input;
        
        if (config.isParallelizationEnabled() && numFilters >= 4) {
            ForkJoinPool executor = config.getExecutor();
            
            executor.submit(() ->
                IntStream.range(0, numFilters).parallel().forEach(f -> {
                    double[][] gradF = gradOutput[f];
                    double[][][] gf = gradFilters[f];
                    
                    for (int c = 0; c < channels; c++) {
                        for (int kh = 0; kh < kernelHeight; kh++) {
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                double sum = 0.0;
                                
                                for (int oh = 0; oh < outH; oh++) {
                                    int ih = oh * stride + kh;
                                    for (int ow = 0; ow < outW; ow++) {
                                        int iw = ow * stride + kw;
                                        sum += gradF[oh][ow] * paddedInput[c][ih][iw];
                                    }
                                }
                                
                                gf[c][kh][kw] = sum;
                            }
                        }
                    }
                })
            ).join();
        } else {
            for (int f = 0; f < numFilters; f++) {
                double[][] gradF = gradOutput[f];
                
                for (int c = 0; c < channels; c++) {
                    for (int kh = 0; kh < kernelHeight; kh++) {
                        for (int kw = 0; kw < kernelWidth; kw++) {
                            double sum = 0.0;
                            
                            for (int oh = 0; oh < outH; oh++) {
                                int ih = oh * stride + kh;
                                for (int ow = 0; ow < outW; ow++) {
                                    int iw = ow * stride + kw;
                                    sum += gradF[oh][ow] * paddedInput[c][ih][iw];
                                }
                            }
                            
                            gradFilters[f][c][kh][kw] = sum;
                        }
                    }
                }
            }
        }
        
        return gradFilters;
    }
    
    // ==================== Pooling Operations ====================
    
    /**
     * Max pooling operation.
     * 
     * @param input input feature maps [channels][height][width]
     * @param poolSize pooling window size
     * @param stride pooling stride
     * @return pooled output and max indices
     */
    public static double[][][] maxPool2d(double[][][] input, int poolSize, int stride) {
        int channels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        
        int outputHeight = (inputHeight - poolSize) / stride + 1;
        int outputWidth = (inputWidth - poolSize) / stride + 1;
        
        double[][][] output = new double[channels][outputHeight][outputWidth];
        
        if (config.isParallelizationEnabled() && channels >= 4) {
            ForkJoinPool executor = config.getExecutor();
            
            executor.submit(() ->
                IntStream.range(0, channels).parallel().forEach(c -> {
                    double[][] inChannel = input[c];
                    double[][] outChannel = output[c];
                    
                    for (int oh = 0; oh < outputHeight; oh++) {
                        int hStart = oh * stride;
                        for (int ow = 0; ow < outputWidth; ow++) {
                            int wStart = ow * stride;
                            
                            double maxVal = Double.NEGATIVE_INFINITY;
                            for (int ph = 0; ph < poolSize; ph++) {
                                for (int pw = 0; pw < poolSize; pw++) {
                                    double val = inChannel[hStart + ph][wStart + pw];
                                    if (val > maxVal) {
                                        maxVal = val;
                                    }
                                }
                            }
                            
                            outChannel[oh][ow] = maxVal;
                        }
                    }
                })
            ).join();
        } else {
            for (int c = 0; c < channels; c++) {
                double[][] inChannel = input[c];
                double[][] outChannel = output[c];
                
                for (int oh = 0; oh < outputHeight; oh++) {
                    int hStart = oh * stride;
                    for (int ow = 0; ow < outputWidth; ow++) {
                        int wStart = ow * stride;
                        
                        double maxVal = Double.NEGATIVE_INFINITY;
                        for (int ph = 0; ph < poolSize; ph++) {
                            for (int pw = 0; pw < poolSize; pw++) {
                                double val = inChannel[hStart + ph][wStart + pw];
                                if (val > maxVal) {
                                    maxVal = val;
                                }
                            }
                        }
                        
                        outChannel[oh][ow] = maxVal;
                    }
                }
            }
        }
        
        return output;
    }
    
    /**
     * Average pooling operation.
     * 
     * @param input input feature maps [channels][height][width]
     * @param poolSize pooling window size
     * @param stride pooling stride
     * @return pooled output
     */
    public static double[][][] avgPool2d(double[][][] input, int poolSize, int stride) {
        int channels = input.length;
        int inputHeight = input[0].length;
        int inputWidth = input[0][0].length;
        
        int outputHeight = (inputHeight - poolSize) / stride + 1;
        int outputWidth = (inputWidth - poolSize) / stride + 1;
        
        double[][][] output = new double[channels][outputHeight][outputWidth];
        double invPoolArea = 1.0 / (poolSize * poolSize);
        
        for (int c = 0; c < channels; c++) {
            double[][] inChannel = input[c];
            double[][] outChannel = output[c];
            
            for (int oh = 0; oh < outputHeight; oh++) {
                int hStart = oh * stride;
                for (int ow = 0; ow < outputWidth; ow++) {
                    int wStart = ow * stride;
                    
                    double sum = 0.0;
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            sum += inChannel[hStart + ph][wStart + pw];
                        }
                    }
                    
                    outChannel[oh][ow] = sum * invPoolArea;
                }
            }
        }
        
        return output;
    }
    
    // ==================== Utility Methods ====================
    
    /**
     * Apply zero-padding to input.
     * 
     * @param input input [channels][height][width]
     * @param padding padding amount
     * @return padded input [channels][height + 2*padding][width + 2*padding]
     */
    public static double[][][] pad(double[][][] input, int padding) {
        if (padding == 0) {
            return input;
        }
        
        int channels = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        
        double[][][] padded = new double[channels][height + 2 * padding][width + 2 * padding];
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                System.arraycopy(input[c][h], 0, padded[c][h + padding], padding, width);
            }
        }
        
        return padded;
    }
    
    /**
     * Remove padding from input.
     * 
     * @param padded padded input [channels][height + 2*padding][width + 2*padding]
     * @param padding padding amount
     * @return unpadded input [channels][height][width]
     */
    public static double[][][] unpad(double[][][] padded, int padding) {
        if (padding == 0) {
            return padded;
        }
        
        int channels = padded.length;
        int height = padded[0].length - 2 * padding;
        int width = padded[0][0].length - 2 * padding;
        
        double[][][] result = new double[channels][height][width];
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                System.arraycopy(padded[c][h + padding], padding, result[c][h], 0, width);
            }
        }
        
        return result;
    }
    
    /**
     * Flatten 3D tensor to 1D array.
     * 
     * @param tensor input tensor [channels][height][width]
     * @return flattened array [channels * height * width]
     */
    public static double[] flatten(double[][][] tensor) {
        int channels = tensor.length;
        int height = tensor[0].length;
        int width = tensor[0][0].length;
        
        // Use exact size to ensure correct length for downstream operations
        double[] result = new double[channels * height * width];
        
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                System.arraycopy(tensor[c][h], 0, result, idx, width);
                idx += width;
            }
        }
        
        return result;
    }
    
    /**
     * Reshape 1D array to 3D tensor.
     * 
     * @param array input array
     * @param channels number of channels
     * @param height height
     * @param width width
     * @return reshaped tensor [channels][height][width]
     */
    public static double[][][] reshape(double[] array, int channels, int height, int width) {
        double[][][] result = new double[channels][height][width];
        
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                System.arraycopy(array, idx, result[c][h], 0, width);
                idx += width;
            }
        }
        
        return result;
    }
}
