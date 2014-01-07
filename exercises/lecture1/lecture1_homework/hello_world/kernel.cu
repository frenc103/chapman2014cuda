#include "cuda_runtime.h"
#include <stdio.h>

__global__ void mykernel(){
	printf("Hello world from device!\n");
} /* end kernel */

int main(void) 
{
        /* launch this kernel 10 times*/
	mykernel<<< ... , ...>>>();
 	cudaDeviceSynchronize();
	printf("Hello World from Host\n");
	return 0;
} /* end main */
