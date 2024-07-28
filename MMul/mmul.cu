
#include <stdio.h>
#include <cuda.h>


__global__ void multMatrixVector(float* B, float* C, float* A, unsigned int size) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  float sum = 0.0f;
  if (x < size)
  {
    for (int i = 0; i < size; i++)
    {
      sum += B[x*size+i] * C[i];
    }
  A[x] = sum;
  }
}
//

int main(int argc, char **argv)
{
	float h_B[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	float h_C[] = {1, 2, 3};
	unsigned int length = 3;

	float *d_B;
  float *d_C;
  float *d_A;
  cudaMalloc((void **)&d_B, length * length * sizeof(float));
  cudaMalloc((void **)&d_C, length * sizeof(float));
  cudaMalloc((void **)&d_A, length * sizeof(float));

  cudaMemcpy(d_B, h_B, length * length * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, length * sizeof(float), cudaMemcpyHostToDevice);
    
  multMatrixVector<<<1, 32>>>(d_B, d_C, d_A, length);
    
  float *h_A = (float*) malloc (length*sizeof(float));
  cudaMemcpy(h_A, d_A, length * sizeof(float), cudaMemcpyDeviceToHost);
  
  for(int i=0; i<length; i++)
  {
    printf("%f, ", h_A[i]);
  }

  free(h_A);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

	return 0;
}
