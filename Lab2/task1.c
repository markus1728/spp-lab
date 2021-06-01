#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>

#include "timing.h"


#define MAX_NUM_LOCAL_ELEMS   3


/**
 * Checks whether arr is sorted locally
 **/
int is_arr_sorted( int* arr, int len ) {

    int i;
    for( i = 0; i < len - 1; ++i )
        if( arr[ i ] > arr[ i + 1 ] )
            return 0;
    return 1;
}


/**
 * Checks whether arr is sorted globally.
 **/
int verify_results( int* arr, int len, int myrank, int nprocs ) {

    int is_sorted_global = 0;

    //Aufgabe 1.7	
	
	MPI_Status status;
	int zahl = -1;

	if (myrank == 0)
	{
		MPI_Sendrecv(arr, len, MPI_INT, myrank+1, 0, &zahl, 0, MPI_INT, MPI_PROC_NULL, 0, MPI_COMM_WORLD, &status); 
	}
	else if(myrank == nprocs -1)
	{
		MPI_Sendrecv(arr, 0, MPI_INT, MPI_PROC_NULL, 0, &zahl, 1, MPI_INT, myrank-1, 0, MPI_COMM_WORLD, &status); 
	}
	else
	{
		MPI_Sendrecv(arr, len, MPI_INT, myrank+1, 0, &zahl, 1, MPI_INT, myrank-1, 0, MPI_COMM_WORLD, &status );
	}

	if (len > 0 && zahl != -1 && arr[0] > zahl)
	{
		is_sorted_global = 1;
	}
	
	if (len == 0 || myrank == 0) is_sorted_global = 1;	
	
	int is_sorted_arr[nprocs];
	MPI_Gather(&is_sorted_global, 1, MPI_INT, is_sorted_arr, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (myrank == 0)
	{
		int i;
		for (i = 0; i < nprocs; i++)
		{
			if (is_sorted_arr[i] == 0)
			{
				is_sorted_global = 0;
				return 0;
			}
		} 
	}

    return is_sorted_global;
}


/**
 * This function compares two integers.
 */
int comp_func( const void* a, const void* b ) {
	return ( *(int*)a - *(int*)b );
}


/**
 * Returns unique random integer.
 */
int get_unique_rand_elem( unsigned int total_count_bits, unsigned int index, double scale ) {

    int random_elem = (int)( scale * drand48() );
    int unique_random_element = ( random_elem << total_count_bits ) | index;

    return unique_random_element;
}


/**
 * Initilizes the input. Each process will have a random local array of numbers. The
 * length of this array is anywhere between 0 to MAX_NUM_LOCAL_ELEMS
 */
void init_input( int w_myrank, int w_nprocs, int* input_arr,
                 int* input_len, int* total_elems ) {

    
	// Initialize random seed
    srand48( w_nprocs );  

    // Total number of elements is 65% of the number of processes
    *total_elems = (int)( w_nprocs * 0.65 );
    int* global_arr = NULL;
    int* sendcnts = NULL;
    int* displs = NULL;

    if( w_myrank == 0 ) {
        printf( "Total number of input elements: %d\n", *total_elems );

        global_arr = malloc( *total_elems * sizeof(int) );

        double scale = *total_elems * 5;
        int total_count_bits = (int)ceil( log( *total_elems ) / log( 2.0 ) );

        // Init global array with random elements
        for( int i = 0; i < *total_elems; ++i )
            global_arr[i] = get_unique_rand_elem( total_count_bits, i, scale );

        // Randomly decide how much elements each rank will get
        sendcnts = malloc( w_nprocs * sizeof(int) );
        memset( sendcnts, 0, w_nprocs * sizeof(int) );
        int total_elem_cnt = *total_elems;
        for( int i = 0; i < w_nprocs; ++i ) {
            double coin_flip = drand48();
            if( coin_flip < 0.45 ) {
                sendcnts[i]++;
                total_elem_cnt--;
                if( total_elem_cnt == 0 ) break;
                coin_flip = drand48();
                if( coin_flip < 0.35 ) {
                    sendcnts[i]++;
                    total_elem_cnt--;
                    if( total_elem_cnt == 0 ) break;
                    if( coin_flip < 0.15 ) {
                        sendcnts[i]++;
                        total_elem_cnt--;
                        if( total_elem_cnt == 0 ) break;
                    }
                }
            }
        }

        // Redistribute remaining counts
        int curr_rank = 0;
        while( total_elem_cnt > 0 ) {
            while( sendcnts[curr_rank] >= MAX_NUM_LOCAL_ELEMS )
                ++curr_rank;
            sendcnts[curr_rank]++;
            total_elem_cnt--;
        }

        displs = malloc( w_nprocs * sizeof(int) );
        displs[0] = 0;
        
	for( int i = 1; i < w_nprocs; ++i )
            displs[i] = displs[i - 1] + sendcnts[i - 1];
    }

    // Redistribute the input length
    MPI_Scatter( sendcnts, 1, MPI_INT, input_len, 1, MPI_INT, 0, MPI_COMM_WORLD );

    // Redistribute the input
    MPI_Scatterv( global_arr, sendcnts, displs, MPI_INT, input_arr, *input_len,
                  MPI_INT, 0, MPI_COMM_WORLD );

    free( global_arr );
    free( sendcnts );
    free( displs );
}

int main( int argc, char** argv ) {

	int w_myrank, w_nprocs;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &w_myrank );
    MPI_Comm_size( MPI_COMM_WORLD, &w_nprocs );

    init_clock_time();

    //
    // Initialization phase
    //

    int n = 0;
    int total_n;
    int elem_arr[MAX_NUM_LOCAL_ELEMS];

    init_input( w_myrank, w_nprocs, elem_arr, &n, &total_n );

    double start = get_clock_time();


    // Aufgabe 1.1

	// Rows
	int r_color = w_myrank / sqrt(w_nprocs);
		
	MPI_Comm row_comm;
	MPI_Comm_split(MPI_COMM_WORLD, r_color, w_myrank, &row_comm);

	int r_myrank, r_nprocs;
	MPI_Comm_rank(row_comm, &r_myrank);
	MPI_Comm_size(row_comm, &r_nprocs);

	// Columns	
	int c_color = w_myrank % (int)sqrt(w_nprocs);
		
	MPI_Comm col_comm;
	MPI_Comm_split(MPI_COMM_WORLD, c_color, w_myrank, &col_comm);

	int c_myrank, c_nprocs;
	MPI_Comm_rank(col_comm, &c_myrank);
	MPI_Comm_size(col_comm, &c_nprocs);


    // Aufgabe 1.2
   
   	//ermittelt für jeden Prozess die Anzahl der Elemente in seinem elem_arr
   	int sendcount = n; 

   	int sumRow = 0;    
    	int sumCol = 0;

    	//Arrays fürs Abspeichern der Arraylängen der anderen Prozesse (pro index eine Arraylänge)
    	int recvcountsRowBuf[r_nprocs];
    	int recvcountsColBuf[c_nprocs];

    	//mit Allgather recvcountsRow/-Col für Allgatherv berechnen
    	//Prozess teilt jedem anderen Prozess in Comm mit, wie viele Elemente in seinem elem_array sind und empfängt gleichzeitig 
    	//die Arraylängen aller anderen Prozesse in diesem comm und legt sie seinem receiveCount ab!
    	MPI_Allgather(&sendcount, 1, MPI_INT, &recvcountsRowBuf, 1, MPI_INT, row_comm);
    	MPI_Allgather(&sendcount, 1, MPI_INT, &recvcountsColBuf, 1, MPI_INT, col_comm);

 
    	//berechne Menge an Elementen pro Zeile und Spalte 
   	int k;
   	for (k = 0; k < (sizeof(recvcountsRowBuf) / sizeof(int)); k++)
    	{
                 sumRow += recvcountsRowBuf[k];
    	}

   	int u;
    	for (u = 0; u < (sizeof(recvcountsColBuf) / sizeof(int)); u++)
    	{
                 sumCol += recvcountsColBuf[u];
    	}

    	//jeweils Array rowA und rowB mit Länge der Summe aus Elementen aus recvountsRowBuf initialisieren
    	int rowA[sumRow];
    	int colA[sumCol];

    	int displsRow[r_nprocs];
    	int displsCol[c_nprocs];

    	displsCol[0]=0;    //das jeweils erste Element muss nicht verrückt werden
    	displsRow[0]=0;

    	//für Row displacement berechnen
    	int q;
    	for (q = 1; q < r_nprocs; q++)
    	{
     	 	displsRow[q]=displsRow[q-1]+recvcountsRowBuf[q-1];
    	}	

    	//für Col displacement berechnen
    	int v; 
    	for (v = 1; v < c_nprocs; v++)
    	{
      	 	displsCol[v]=displsCol[v-1]+recvcountsColBuf[v-1];
    	}

    	//alle benötigten Inputdaten für Allgatherv vorhanden
    	MPI_Allgatherv(&elem_arr, sendcount, MPI_INT, &rowA, recvcountsRowBuf, displsRow, MPI_INT, row_comm);
    	MPI_Allgatherv(&elem_arr, sendcount, MPI_INT, &colA, recvcountsColBuf, displsCol, MPI_INT, col_comm);


    // Aufgabe 1.3
    	qsort(rowA, ( sizeof(rowA) / sizeof(rowA[0]) ), sizeof(int), comp_func);   
    	qsort(colA, ( sizeof(colA) / sizeof(colA[0]) ), sizeof(int), comp_func);  


    // Aufgabe 1.4

	int colAsize = sizeof(colA) / sizeof(colA[0]);
	int rowAsize = sizeof(rowA) / sizeof(rowA[0]);

	int localRankArray[colAsize];
	int index = 0;
	
	if (rowAsize == 0)
	{
		for (int i = 0; i < colAsize; i++)
			localRankArray[i] = 0;
	}
	else
	{
		for (int i = 0; i < colAsize; i++)
		{
			localRankArray[i] = rowAsize;
			int j = index;
			while (j < rowAsize)
			{
				if (colA[i] <= rowA[j])
				{
					localRankArray[i] = j;
					index = j;
					break;
				}
				else
					j++;	
			}

		}
	}


    // Aufgabe 1.5
   
   	 int arr_global_ranks[colAsize]; 
    
   	 for (int i = 0; i < colAsize; i++) 
   	 {
         	MPI_Allreduce(&localRankArray[i], &arr_global_ranks[i], 1, MPI_INT, MPI_SUM, col_comm);
    	}
	

    // Aufgabe 1.6

	MPI_Request req_arr[MAX_NUM_LOCAL_ELEMS];
	MPI_Status stat_arr[MAX_NUM_LOCAL_ELEMS];
	int n_req = 0;

	int i;
	for (i = 0; i < n; i++)
	{
		int j;
		for (j = 0; j < colAsize; j++)
		{
			if (colA[j] == elem_arr[i])  //nur die schicken, die im eigenen elem_arr sind!
			{
				MPI_Isend( &(elem_arr[i]), 1, MPI_INT, arr_global_ranks[j], arr_global_ranks[j], MPI_COMM_WORLD,
						req_arr+n_req);
				n_req++;
				break;

			}
		}
	}

	// Receive elements
	MPI_Waitall(n_req, req_arr, stat_arr);
	if(w_myrank < total_n)
	{
		MPI_Request recvRequest;
		MPI_Status recvStatus;

		MPI_Irecv( &(elem_arr[0]), 1, MPI_INT, MPI_ANY_SOURCE, w_myrank, MPI_COMM_WORLD, &recvRequest);
		MPI_Wait(&recvRequest, &recvStatus);
		n = 1;
	}
	else 
		n = 0;
	
	// End-Überprüfung
	if (n == 1) printf("Ich bin Prozess %d und ich besitze Zahl %d\n", w_myrank, elem_arr[0]);
	else if (n == 0) printf("Ich bin Prozess %d und besitze keine Zahl\n", w_myrank);
	else printf("Ich bin Prozess %d und hier ist etwas schief gelaufen\n", w_myrank);

	
    //
    // Measure the execution time after all the steps are finished,
    // but before verifying the results
    //
    double elapsed = get_clock_time() - start;

    //
    // Verify the data is sorted globally
    //
    int res = verify_results( elem_arr, n, w_myrank, w_nprocs );
    if( w_myrank == 0 ) {
        if( res ) {
            printf( "Results correct!\n" );
        }
        else {
            printf( "Results incorrect!\n" );
        }
    }

    // Get timing - max across all ranks
    double elapsed_global;
    MPI_Reduce( &elapsed, &elapsed_global, 1, MPI_DOUBLE,
                MPI_MAX, 0, MPI_COMM_WORLD );

    if( w_myrank == 0 ) {
        printf( "Elapsed time (ms): %f\n", elapsed_global );
    }

    MPI_Finalize();

    return 0;
}
