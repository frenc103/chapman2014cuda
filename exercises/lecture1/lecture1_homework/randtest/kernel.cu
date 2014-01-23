#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void print5(unsigned int *a)
{
	int i;

	curand_init(&a[blockIdx.x], 0, 0, 
    /* finish this code to calculate c element-wise from a and b where each block calculates one element */
	for (i = 0; i < 5; i++)
	{
		printf("Block %i Rand %i - %i", blockIdx.x, i, rand_r(&a[blockIdx.x]));
	}
}


/* experiment with different values of N.  */
/* how large can it be? */
#define N 3

int main()
{
	unsigned int *a;
	unsigned int *d_a;
	int size = N * sizeof( int );

	/* allocate space for device copies of a, b, c */
	
	cudaMalloc( (void **) &d_a, size );

	/* allocate space for host copies of a, b, c and setup input values */

	a = (unsigned int *)malloc( size );

	/* intializing a, b, c on host */
	
	for( int i = 0; i < N; i++ )
	{
		a[i] = rand();
	}

	/* copy inputs to device */
	
	cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );

	/* launch the kernel on the GPU */
	/* finish this kernel launch with N blocks and 1 thread per block */
	print5<<<N,1>>>(d_a);


	free(a);
	cudaFree( d_a );
	
	return 0;
} /* end main */
