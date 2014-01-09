#include "cuda_runtime.h"
#include <stdio.h>

__global__ void mykernel(){
	printf("Hello world from device block %i !\n",blockIdx.x);
} /* end kernel */

int main(void) 
{
        /* launch this kernel 10 times*/
	mykernel<<< 2 , 5 >>>();
 	cudaDeviceSynchronize();
	printf("Hello World from Host\n");
	return 0;
} /* end main */
