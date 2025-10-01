#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vec_add(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20; // ~1M elements
    size_t bytes = N * sizeof(int);
    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(bytes); h_b = (int*)malloc(bytes); h_c = (int*)malloc(bytes);
    for (int i=0;i<N;++i){ h_a[i]=i; h_b[i]=2*i; }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;
    vec_add<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // validate
    for (int i=0;i<10;++i){
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    printf("Done.\\n");
    return 0;
}
