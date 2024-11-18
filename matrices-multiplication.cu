#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK_CUDA_CALL(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


/**
* Multiplies two matrices A and B on CPU and stores result in matrix C.
* 
* @param A matrix A
* @param B matrix B
* @param C resulting matrix C
* @param n size of square matrices
*/
void 
multiplyMatricesCPU(
    float* A, 
    float* B, 
    float* C, 
    int n
) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

/**
* Kernel-function which multiplies two matrices A and B on GPU and stores 
* result in matrix C.
* 
* @param A matrix A
* @param B matrix B
* @param C resulting matrix C
* @param n size of square matrices
*/
__global__ 
void 
multiplyMatricesGPU(
    float* A, 
    float* B, 
    float* C, 
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

/**
* Transposes provided matrix B and stores result in matrix B_T.
*
* @param B matrix B
* @param B_T resulting matrix B_T
* @param n size of square matrices
*/
void 
transposeMatrix(
    float* B, 
    float* B_T, 
    int n
) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            B_T[j * n + i] = B[i * n + j];
        }
    }
}

/**
* Multiplies two matrices A and B transposed on CPU and stores result in matrix C.
* The function implements optimisation approach for matrix multiplication, which
* concludes in transposing matrix B and then multiplying it with matrix A, so
* it is possible to access elements of matrix B in a row-major order, which is
* less memory intensive.
* 
* @param A matrix A
* @param B_T matrix B
* @param C resulting matrix C
* @param n size of square matrices
*/
void 
multiplyMatricesWithTransposeCPU(
    float* A, 
    float* B_T, 
    float* C, 
    int n
) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B_T[j * n + k];
            }
        }
    }
}

/**
* Prints provided matrix.
*
* @param matrix matrix to be printed
* @param n size of square matrix
*/
void 
printMatrix(float* matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

/**
* Main function which tests matrix multiplication on CPU and GPU.
*/
int 
main() {
    int 
        sizes[] = {3, 512, 102, 2048 }; // sizes of matrices to be tested 
        // (3x3 is for demonstration of calculus correctness)

    for (int i = 0; i < 4; ++i) {

        // INITIALIZATION
        int 
            N = sizes[i];
        float* 
            A = new float[N * N], 
            B = new float[N * N],
            C_CPU = new float[N * N],
            C_GPU = new float[N * N],
            C_Trans = new float[N * N];

        for (int i = 0; i < N * N; ++i) {
            A[i] = static_cast<float>(rand()) / RAND_MAX;
            B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        if (N == 3) {
            printMatrix(A, N);
            printMatrix(B, N);
        }

        // CPU TEST
        auto startCPU = std::chrono::high_resolution_clock::now();
        multiplyMatricesCPU(A, B, C_CPU, N);
        auto endCPU = std::chrono::high_resolution_clock::now();
        double timeCPU = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();
        
        if (N == 3) {
            std::cout << "Calculated on CPU:\n";
            printMatrix(C_CPU, N);
        }

        // GPU TEST
        float* d_A, * d_B, * d_C;
        CHECK_CUDA_CALL(cudaMalloc(&d_A, N * N * sizeof(float)));
        CHECK_CUDA_CALL(cudaMalloc(&d_B, N * N * sizeof(float)));
        CHECK_CUDA_CALL(cudaMalloc(&d_C, N * N * sizeof(float)));
        CHECK_CUDA_CALL(cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_CALL(cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

        cudaEvent_t startGPU, endGPU;
        CHECK_CUDA_CALL(cudaEventCreate(&startGPU));
        CHECK_CUDA_CALL(cudaEventCreate(&endGPU));

        CHECK_CUDA_CALL(cudaEventRecord(startGPU));
        multiplyMatricesGPU << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
        CHECK_CUDA_CALL(cudaDeviceSynchronize());
        CHECK_CUDA_CALL(cudaEventRecord(endGPU));

        CHECK_CUDA_CALL(cudaEventSynchronize(endGPU));

        float elapsedGPU;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsedGPU, startGPU, endGPU));
        double timeGPU = elapsedGPU;

        CHECK_CUDA_CALL(cudaMemcpy(C_GPU, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

        if (N == 3) {
            std::cout << "Calculated on GPU:\n";
            printMatrix(C_GPU, N);
        }

        // CPU TEST WITH TRANSPOSE
        float* B_T = new float[N * N];
        transposeMatrix(B, B_T, N);
        auto startTrans = std::chrono::high_resolution_clock::now();
        multiplyMatricesWithTransposeCPU(A, B_T, C_Trans, N);
        auto endTrans = std::chrono::high_resolution_clock::now();
        double timeTrans = std::chrono::duration<double, std::milli>(endTrans - startTrans).count();

        if (N == 3) {
            std::cout << "Calculated on CPU with transpose:\n";
            printMatrix(C_Trans, N);
        }


        // RESULTS PRINT
        std::cout << "Matrix size: " << N << "x" << N << "\n";
        std::cout << "Time (CPU): " << timeCPU << " ms\n";
        std::cout << "Time (GPU): " << timeGPU << " ms\n";
        std::cout << "Time (CPU with Transpose): " << timeTrans << " ms\n";

        double slowest = std::max({ timeCPU, timeGPU, timeTrans });
        std::cout << "Speedup (CPU vs GPU): " << slowest / timeGPU << "x\n";
        std::cout << "Speedup (CPU vs Transpose): " << slowest / timeTrans << "x\n\n";


        // CLEANUP
        delete[] A;
        delete[] B;
        delete[] C_CPU;
        delete[] C_GPU;
        delete[] C_Trans;
        delete[] B_T;
        CHECK_CUDA_CALL(cudaFree(d_A));
        CHECK_CUDA_CALL(cudaFree(d_B));
        CHECK_CUDA_CALL(cudaFree(d_C));

        CHECK_CUDA_CALL(cudaEventDestroy(startGPU));
        CHECK_CUDA_CALL(cudaEventDestroy(endGPU));
    }

    return 0;
}
