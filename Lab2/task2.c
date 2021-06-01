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

    //Aufgabe 1.7 übernommen
	
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
 * Merges arr_from into arr.
 */
void merge_arr( int* arr1, int len1, int* arr2, int len2, int* arr_out, int* len_out ) {

   int idx1 = 0, idx2 = 0, idx3 = 0;

    while( idx1 < len1 ) {
        while( idx2 < len2 && 
               arr2[ idx2 ] < arr1[ idx1 ] ) {
            arr_out[ idx3++ ] = arr2[ idx2++ ];
        }
        arr_out[ idx3++ ] = arr1[ idx1++ ];
    }
    while( idx2 < len2 ) {
        arr_out[ idx3++ ] = arr2[ idx2++ ];
    }
    *len_out = idx3;

}


/**
 * All-gather-merge using hypercube approach.
 */
void all_gather_merge( int* arr, int len, int** out_arr, int* out_len, 
                       int nprocs, MPI_Comm comm ) {

    // Aufgabe 2.2 bis 2.4

    	//bestimme k für Interationen und Dimensionen des Hypercubes 
    	int k = log10(nprocs) / log10(2);

    	int l_myrank;
    	MPI_Comm_rank( comm, &l_myrank );

    	int i; 
    	int rankPartner;
    	int* main_arr;
    	int main_len;

    	for (i = 0; i <= k-1; i++) {

        	//bei erster Iteration nehme erhaltenes Array und übertrage es in main_arr
        	//zur weiteren Bearbeitung im nachfolgenden Code
        	if ( i== 0) {
            		main_arr = (int*) malloc(len * sizeof(int));
            		int z;
            		for (z = 0; z < len; z++) {
               			 main_arr[z] = arr[z];
            		}
            		main_len = len;
        	}

        	//bestimme Tauschpartner für die jeweilige Iteration/Dimension
        	rankPartner = l_myrank ^ (int) pow(2,i);
	

	   	//erstelle für Tauschpartner und sich einen eigenen comm (immer 2 Prozesse drin)
        	int color = rankPartner + l_myrank;
        	MPI_Comm Partner_comm;
        	MPI_Comm_split(comm, color, l_myrank, &Partner_comm);
        	int parRank;
        	MPI_Comm_rank(Partner_comm, &parRank);


        	//bestimme, wie viele Elemente vom Tauschpartner zu erwarten sind, damit die
        	//Länge des Arrays, welches die Elemente empfangen wird, gesetzt werden kann.
        	//in recvcountsBuf[0] liegt immer die Elementanzahl von Partner 0 und in
        	// recvcountsBuf[1] liegt immer die Elementanzahl von Partner 1
        	int recvcountsBuf[2];
        	MPI_Allgather(&main_len, 1, MPI_INT, &recvcountsBuf, 1, MPI_INT, Partner_comm);


        	//wenn ein Prozess Partner 0 ist, muss er sein recv_arr mit Länge vom Array des
        	//Partners 1 einstellen. Partner 1 stellt sein recv_arr mit Länge vom Array
        	// des Partners 0 ein.
        	int recvcount;
        	if (parRank==0) {
            		recvcount = recvcountsBuf[1];
        	}
        	else {
            		recvcount = recvcountsBuf[0];
        	}
        	int recv_arr[recvcount];
        

        	//Sende Elemente an Partner und empfange Elemente vom Partner
        	MPI_Status status;
        	MPI_Sendrecv(main_arr, main_len, MPI_INT, rankPartner, i, recv_arr, recvcount,
                     MPI_INT, rankPartner, i, comm, &status);


        	//bereite ein Array fürs mergen von eigenen und empfangenen Elementen vor
        	int mer_arr_length = main_len + recvcount;
        	int mer_arr[mer_arr_length];
	
	
        	//merge eigene und empfangene Elemente und lege Ergebnis in mer_arr ab
        	merge_arr( main_arr, main_len, recv_arr , recvcount , mer_arr , &mer_arr_length );

	
        	//übertrage das gemergte Array mer_arr in das main_arr, um damit die
        	//nächste Iteration durchführen zu können
        	main_len = mer_arr_length;
        	main_arr = (int*) malloc(mer_arr_length * sizeof(int));
           	int z;
            	for (z = 0; z < mer_arr_length; z++) {
               	 	main_arr[z] = mer_arr[z];
            	}

    	} 

	//nach dem Durchlauf aller Iterationen/Dimensionen wird das Ergebnis ins
	//out_arr zur Darstellung in der main-Funktion übertragen 
	int e;
	for (e = 0; e < main_len; e++) {
		out_arr[e] = &main_arr[e];
	}
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
    MPI_Scatterv( global_arr, sendcnts, displs, MPI_INT, input_arr, *input_len, MPI_INT, 
                  0, MPI_COMM_WORLD );

    free( global_arr );
    free( sendcnts );
    free( displs );
}

int main( int argc, char** argv ) {

    int w_myrank, w_nprocs;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &w_myrank );
    MPI_Comm_size( MPI_COMM_WORLD, &w_nprocs );

    //
    // Initialization phase
    //
    int n = 0;
    int total_n;
    int elem_arr[MAX_NUM_LOCAL_ELEMS];
    
    init_input( w_myrank, w_nprocs, elem_arr, &n, &total_n );

    
    //
    // Aufgabe 2
    //
    

    //Aufgabe 1.1 

    	// Comm für alle Elemente in jeweiliger Zeile
	int r_color = w_myrank / sqrt(w_nprocs);
	int r_myrank, r_nprocs;
	MPI_Comm row_comm;
	MPI_Comm_split(MPI_COMM_WORLD, r_color, w_myrank, &row_comm);
	MPI_Comm_rank(row_comm, &r_myrank);
	MPI_Comm_size(row_comm, &r_nprocs);

	// Comm für alle Elemente in jeweiliger Spalte 
	int c_color = w_myrank % (int)sqrt(w_nprocs);
	int c_myrank, c_nprocs;		
	MPI_Comm col_comm;
	MPI_Comm_split(MPI_COMM_WORLD, c_color, w_myrank, &col_comm);
	MPI_Comm_rank(col_comm, &c_myrank);
	MPI_Comm_size(col_comm, &c_nprocs);


    //Aufgabe 1.2
	
	//Ermittlung der Anzahl aller Elemente in einer Zeile bzw. Spalte 
   	int sumRow = 0;
    	int sumCol = 0;
    	int recvcountsRowBuf[r_nprocs];
    	int recvcountsColBuf[c_nprocs];
    	MPI_Allgather(&n, 1, MPI_INT, &recvcountsRowBuf, 1, MPI_INT, row_comm);
    	MPI_Allgather(&n, 1, MPI_INT, &recvcountsColBuf, 1, MPI_INT, col_comm);

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

   	//Arrays, welche am Ende die Zeilen- und Spaltenelemente enthalten 
    	int* row_arr[sumRow];
    	int* col_arr[sumCol];
   

    //Aufgabe 2.1
	
    	qsort(elem_arr, n , sizeof(int), comp_func);

	
    //Aufgabe 2.5

   	//Aufruf der Funktionen, um Zeilen- und Spaltenelemente mithilfe des Hypercubes zu sammeln und zu sortieren
   	all_gather_merge( &(elem_arr[0]), n, row_arr, &sumRow, r_nprocs, row_comm );
   	all_gather_merge( &(elem_arr[0]), n, col_arr, &sumCol, c_nprocs, col_comm );



    //Aufgabe 1.4

	int col_arrsize = sizeof(col_arr) / sizeof(col_arr[0]);
	int row_arrsize = sizeof(row_arr) / sizeof(row_arr[0]);

	int localRankArray[col_arrsize];
	int index = 0;
	
	if (row_arrsize == 0)
	{
		for (int i = 0; i < col_arrsize; i++)
			localRankArray[i] = 0;
	}
	else
	{
		for (int i = 0; i < col_arrsize; i++)
		{
			localRankArray[i] = row_arrsize;
			int j = index;
			while (j < row_arrsize)
			{
				if (*col_arr[i] <= *row_arr[j])
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


    //Aufgabe 1.5
   
    	int arr_global_ranks[col_arrsize]; 
    
    	for (int i = 0; i < col_arrsize; i++) 
    	{
         	MPI_Allreduce(&localRankArray[i], &arr_global_ranks[i], 1, MPI_INT, MPI_SUM, col_comm);
    	}
	

    //Aufgabe 1.6

	MPI_Request req_arr[MAX_NUM_LOCAL_ELEMS];
	MPI_Status stat_arr[MAX_NUM_LOCAL_ELEMS];
	int n_req = 0;

	int i;
	for (i = 0; i < n; i++)
	{
		int j;
		for (j = 0; j < col_arrsize; j++)
		{
			if (*col_arr[j] == elem_arr[i])
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
 
    MPI_Finalize();

    return 0;
}
