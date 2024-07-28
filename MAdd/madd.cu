
#include <stdio.h>
#include <cuda.h>

#define TILE 16


// 1 thread 1 element
__global__ void addMMv1(float* B, float* C, float* A, unsigned int size) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if (i < size)
	{
		C[i] = A[i] + B[i];
	}
}

// 1 thread 1 col
__global__ void addMMv2(float* B, float* C, float* A, unsigned int size) {
  int y = blockDim.x*blockIdx.x + threadIdx.x;

  if (y < size)
	{
		for (int row_idx = 0; row_idx < size; ++row_idx)
		{
			int idx = row_idx * size + y;
			C[idx] = A[idx] + B[idx];
		}
	}

}

// 1 thread 1 row
__global__ void addMMv3(float* B, float* C, float* A, unsigned int size) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  
  if (x < size)
	{
		int row_idx = x * size; 

		for (int col_idx = 0; col_idx < size; ++col_idx)
		{
			int idx = row_idx + col_idx;
			C[idx] = A[idx] + B[idx];
		}
	}
}
//

void generateRandomFlattenMatrix(float *M, unsigned int size) {
	for (int i = 0; i < size; ++i) {
	   M[i] = (rand() % 20) + 50;
	}
}



int main(int argc, char **argv)
{
	
	unsigned int length = 32;
  float *h_A = (float *) malloc(length*length);
  float *h_B = (float *) malloc(length*length);
	generateRandomFlattenMatrix(h_A, length);
	generateRandomFlattenMatrix(h_B, length);
  
  float *d_B;
  float *d_C;
  float *d_A;
  cudaMalloc((void **)&d_B, length * length * sizeof(float));
  cudaMalloc((void **)&d_C, length * length * sizeof(float));
  cudaMalloc((void **)&d_A, length * length * sizeof(float));

  cudaMemcpy(d_B, h_B, length * length * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, h_A, length * length * sizeof(float), cudaMemcpyHostToDevice);

  // Uncomment the chosen one of the three below lines to get results form particular function
  //addMMv1<<<1, 64>>>(d_B, d_C, d_A, length);
  //addMMv2<<<1, 32>>>(d_B, d_C, d_A, length);
  addMMv3<<<1, 32>>>(d_B, d_C, d_A, length);
    
  float *h_C = (float*) malloc (length* length*sizeof(float));
  cudaMemcpy(h_C, d_C, length * length * sizeof(float), cudaMemcpyDeviceToHost);
  
  for(int i=0; i<length; i++)
  {
    printf("%f, ", h_C[i]);
  }

  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

	return 0;
}
