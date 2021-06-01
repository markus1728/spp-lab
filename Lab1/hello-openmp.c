#include <omp.h>
#include <stdio.h>

int main ( int argc, char *argv[] )
{
	printf("Maximale Thread-Anzahl: %d\n", omp_get_max_threads());
	omp_set_num_threads(5);
#pragma omp parallel
	printf("Hello %d\n", omp_get_thread_num());
	return 0;
}