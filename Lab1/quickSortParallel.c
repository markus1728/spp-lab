#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// This function partitions array A into two subarrays A_1 and A_2
// Input:
//     *l is the index of the first element in array A
//     *r is the index of the last element in array A
//
//     A
//     [  |  |  |  |  ...  |  |  |  |  ]
//      *l                           *r
//
// Output:
//     *l is now the index of the first element of array A_2, which still needs to be sorted
//     *r is now the index of the last  element of array A_1, which still needs to be sorted
//
//     A_1              A_2
//     [  |  | ... |  ] [  |  | ... |  ]
//                  *r   *l
void partition ( int* A, int* l, int* r )
{
	int highest, lowest, pivot, i;

	highest = *r;
	lowest = *l;
	pivot = A[highest];
        i = (lowest - 1);

    for (int j = lowest; j <= highest-1; j++)
    {
        if (A[j] <= pivot)
        {
            i++;

			if (A[i] != A[j])
			{
				// Swap
				int tmp = A[i];
				A[i] = A[j];
				A[j] = tmp;
			}
        }
    }

	// Swap pivot
	int tmp = A[i+1];
	A[i+1] = A[highest];
	A[highest] = tmp;

	// New pivot
    *r = (i+1);
}

// Input:
//     l is the index of the first element in array A
//     r is the index of the last element in array A
//
//     A
//     [  |  |  |  |  ...  |  |  |  |  ]
//      l                            r
void quicksort( int* A, int l, int r )
{
	int numberOfElements=r;
	if (l < r)
    {
                int oldR = r;
		int pivot;

		partition(A, &l, &r);

		pivot = r;
		r = oldR;

				#pragma omp task default(none) firstprivate(A,l,pivot) final(r + 1 <100)
				//firstprivate: A,l,pivot sind bereits initialisiert und sollen als private übergeben werden
        {quicksort(A, l, pivot-1);}

				#pragma omp task default(none) firstprivate(A,r,pivot) final(r + 1 <100)
				//firstprivate: A,r,pivot sind bereits initialisiert und sollen als private übergeben werden
        {quicksort(A, pivot+1, r);}
     }
}
int main( int argc, char** argv )
{
	unsigned int numberOfElements;
	unsigned int* numbers;

    if( argc < 2 )
    {
        printf( "Usage: %s <array length>\n", argv[0] );
        return 1;
    }
    // Read in number of elements
	numberOfElements = atoi(argv[1]);

    srand( 14811 );

    // Allocate array
	numbers = malloc(numberOfElements * sizeof(unsigned int));

    // Initialize array
	for (int i = 0; i < numberOfElements; i++)
	{
		numbers[i] = rand();
	}

    // Time the execution
	#pragma omp parallel default(none) shared(numbers,numberOfElements)
	{
		#pragma omp single  //single: nur ein thread soll quicksort ausführen
		{quicksort(numbers, 0, numberOfElements-1);}
	}

    // Verify sorted order
	int  sortedcorrectly = 1;
        for (int i = 1; i < numberOfElements-1; i++)
        {
                if (numbers[i] < numbers[i-1])
                {
                        sortedcorrectly = 0;
                }
        }
	if (sortedcorrectly ==1)
                printf("This is sorted.\n");
        else
            	printf("This isn't sorted.\n");



	free(numbers);

    return 0;
}
