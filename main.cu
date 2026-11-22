#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err) << "\n"; \
    } \
} while(0)

// 5 simple interdependent kernels
__global__ void kernelA(float *data, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) data[i]+=1.0f; }
__global__ void kernelB(float *data, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) data[i]+=2.0f; }
__global__ void kernelC(float *data, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) data[i]+=3.0f; }
__global__ void kernelD(float *data, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) data[i]+=4.0f; }
__global__ void kernelE(float *data, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) data[i]+=5.0f; }

int main() {
    const int N = 1024;
    const int ITERATIONS = 1000;
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;   // <-- FIXED: was 1
    size_t size = N * sizeof(float);

    // Confirm a CUDA-capable GPU is visible
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    std::cout << "CUDA device count: " << devCount << "\n";
    if (devCount == 0) {
        std::cerr << "No CUDA-capable device detected. Check your NVIDIA driver.\n";
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << "\n\n";

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Sequential Launch ---
    CUDA_CHECK(cudaMemset(d_data, 0, size));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cpu_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < ITERATIONS; i++) {
        kernelA<<<BLOCKS, THREADS, 0, stream>>>(d_data, N);
        kernelB<<<BLOCKS, THREADS, 0, stream>>>(d_data, N);
        kernelC<<<BLOCKS, THREADS, 0, stream>>>(d_data, N);
        kernelD<<<BLOCKS, THREADS, 0, stream>>>(d_data, N);
        kernelE<<<BLOCKS, THREADS, 0, stream>>>(d_data, N);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    auto cpu_end = std::chrono::high_resolution_clock::now();

    float seq_gpu_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&seq_gpu_time, start, stop));
    double seq_cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // --- Graph Capture ---
    CUDA_CHECK(cudaMemset(d_data, 0, size));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    { // capture ONE iteration = 5 kernels A-E
        kernelA<<<BLOCKS, THREADS, 0, stream>>>(d_data, N);
        kernelB<<<BLOCKS, THREADS, 0, stream>>>(d_data, N);
        kernelC<<<BLOCKS, THREADS, 0, stream>>>(d_data, N);
        kernelD<<<BLOCKS, THREADS, 0, stream>>>(d_data, N);
        kernelE<<<BLOCKS, THREADS, 0, stream>>>(d_data, N);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));

    CUDA_CHECK(cudaMemset(d_data, 0, size));
    CUDA_CHECK(cudaDeviceSynchronize());

    cpu_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < ITERATIONS; i++) CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    cpu_end = std::chrono::high_resolution_clock::now();

    float graph_gpu_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&graph_gpu_time, start, stop));
    double graph_cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // --- Validation ---
    float *h_data = new float[N];
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    float expected = 15.0f * ITERATIONS;   // (1+2+3+4+5) * 1000 = 15000
    bool valid = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_data[i] - expected) > 1e-3f) { valid = false; break; }
    }

    std::cout << "Validation: " << (valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "Sample h_data[0]=" << h_data[0]
              << ", h_data[N-1]=" << h_data[N-1]
              << " (expected " << expected << ")\n";
    std::cout << "Sequential - CPU: " << seq_cpu_time << " ms, GPU: " << seq_gpu_time << " ms\n";
    std::cout << "Graph      - CPU: " << graph_cpu_time << " ms, GPU: " << graph_gpu_time << " ms\n";

    delete[] h_data;
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    return 0;
}