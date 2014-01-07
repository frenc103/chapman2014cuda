#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void add(int *a, int *b, int *c)
{
    /* sum of a and b stored in c */
	...
}

int main()
{
    int a, b, c;
	int *d_a, *d_b, *d_c;
	int size = sizeof( int );

	/* allocate space for device copies of a, b, c */

	cudaMalloc( (void **) &d_a, size );
	cudaMalloc( (void **) &d_b, size );
	cudaMalloc( (void **) &d_c, size );

	/* setup initial values */

	a = 2;
	b = 7;
	c = -99;

	/* copy inputs to device */

	cudaMemcpy( d_a, &a, size, ... );
	cudaMemcpy( d_b, &b, size, ... );

	/* launch the kernel on the GPU */
	/* use 1 block per grid and 1 thread per block*/
	...

	/* copy result back to host */

	cudaMemcpy( &c, d_c, size, ... );

	printf("value of c after kernel is %d\n",c);

	/* clean up */

	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	
	return 0;
} /* end main */
