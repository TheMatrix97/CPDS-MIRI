/*
 * Iterative solver for heat distribution
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

void usage( char *s )
{
    fprintf(stderr, 
	    "Usage: %s <input file> [result file]\n\n", s);
}

int main( int argc, char *argv[] )
{
    unsigned iter;
    FILE *infile, *resfile;
    char *resfilename;
    int myid, numprocs;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
      printf("I am the master (%d) and going to distribute work to %d additional workers ...\n", myid, numprocs-1);

    // algorithmic parameters
    algoparam_t param;
    int np;

    double runtime, flop;
    double residual=0.0;

    // check arguments
    if( argc < 2 )
    {
	usage( argv[0] );
	return 1;
    }

    // check input file
    if( !(infile=fopen(argv[1], "r"))  ) 
    {
	fprintf(stderr, 
		"\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
      
	usage(argv[0]);
	return 1;
    }

    // check result file
    resfilename= (argc>=3) ? argv[2]:"heat.ppm";

    if( !(resfile=fopen(resfilename, "w")) )
    {
	fprintf(stderr, 
		"\nError: Cannot open \"%s\" for writing.\n\n", 
		resfilename);
	usage(argv[0]);
	return 1;
    }

    // check input
    if( !read_input(infile, &param) )
    {
	fprintf(stderr, "\nError: Error parsing input file.\n\n");
	usage(argv[0]);
	return 1;
    }
    print_params(&param);

    // set the visualization resolution
    
    param.u     = 0;
    param.uhelp = 0;
    param.uvis  = 0;
    param.visres = param.resolution;
   
    if( !initialize(&param) )
	{
	    fprintf(stderr, "Error in Solver initialization.\n\n");
	    usage(argv[0]);
            return 1;
	}

    // full size (param.resolution are only the inner points)
    np = param.resolution + 2;
	
	//distribute work among users
	int filas = param.resolution / numprocs;
    
    // starting time
    runtime = wtime();

    // send to workers the necessary data to perform computation
    for (int i=0; i<numprocs; i++) {
        if (i>0) {
                MPI_Send(&param.maxiter, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.resolution, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.algorithm, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				//Send U and Uhelp assigned to each thread (i = num thread * rows (rows assign to each thread) * np = column by row)
                MPI_Send(&param.u[i*filas*np], (filas+2)*np, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.uhelp[i*filas*np], (filas+2)*np, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }

    iter = 0;
    while(1) {
	switch( param.algorithm ) {
	    case 0: // JACOBI
	            residual = relax_jacobi(param.u, param.uhelp, filas+2, np);
		    // Copy uhelp into u
		    for (int i=0; i<filas+2; i++)
    		        for (int j=0; j<np; j++)
	    		    param.u[ i*np+j ] = param.uhelp[ i*np+j ];
				
			//Share boundaries to adjacent procs (if more than one proc assigned)
			if(numprocs > 1){	
				MPI_Send(&param.u[filas*np], np, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
				MPI_Recv(&param.u[(filas+1)*np], np, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
		    }
			break;
	    case 1: // RED-BLACK
		    residual = relax_redblack(param.u, np, np);
		    break;
	    case 2: // GAUSS
		    residual = relax_gauss(param.u, filas+2, np);
			//Share boundaries to adjacent procs (if more than one proc assigned)
			if(numprocs > 1){	
				MPI_Send(&param.u[filas*np], np, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
				MPI_Recv(&param.u[(filas+1)*np], np, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
		    }
		    break;
	    }

        iter++;
		double temp_residual = residual;
		//with all reduce all processes get the correct residual, finishing at the same time
		MPI_Allreduce(&temp_residual, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
        // solution good enough ?
        if (residual < 0.00005) break;

        // max. iteration reached ? (no limit with maxiter=0)
        if (param.maxiter>0 && iter>=param.maxiter) break;
    }
	
	//Get Work from slaves
	for(int nproc = 1; nproc < numprocs; nproc++){ //Get result from workers
		MPI_Recv(&param.u[nproc*filas*np], (filas+2)*(np), MPI_DOUBLE, nproc, 1, MPI_COMM_WORLD, &status); //get processed info from thread assigned
	}

    // Flop count after iter iterations
    flop = iter * 11.0 * param.resolution * param.resolution;
    // stopping time
    runtime = wtime() - runtime;

    fprintf(stdout, "Time: %04.3f ", runtime);
    fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", 
	    flop/1000000000.0,
	    flop/runtime/1000000);
    fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

    // for plot...
    coarsen( param.u, np, np,
	     param.uvis, param.visres+2, param.visres+2 );
  
    write_image( resfile, param.uvis,  
		 param.visres+2, 
		 param.visres+2 );

    finalize( &param );

    fprintf(stdout, "Process %d finished computing with residual value = %f\n", myid, residual);

    MPI_Finalize();

    return 0;

} else {

    printf("I am worker %d and ready to receive work to do ...\n", myid);

    // receive information from master to perform computation locally

    int columns, rows, np;
    int iter, maxiter;
    int algorithm;
    double residual;

    MPI_Recv(&maxiter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&columns, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&algorithm, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    rows = columns;
    np = columns + 2;
	
	//number of rows depend of nprocs
	rows = rows / numprocs;

    // allocate memory for worker
    double * u = calloc( sizeof(double),(rows+2)*(columns+2) );
    double * uhelp = calloc( sizeof(double),(rows+2)*(columns+2) );
    if( (!u) || (!uhelp) )
    {
        fprintf(stderr, "Error: Cannot allocate memory\n");
        return 0;
    }
    
    // fill initial values for matrix with values received from master
    MPI_Recv(&u[0], (rows+2)*(columns+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&uhelp[0], (rows+2)*(columns+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

    iter = 0;
    while(1) {
	switch( algorithm ) {
	    case 0: // JACOBI
	            residual = relax_jacobi(u, uhelp, rows+2, np);
		    // Copy uhelp into u
		    for (int i=0; i<rows+2; i++)
    		        for (int j=0; j<np; j++)
	    		    u[ i*np+j ] = uhelp[ i*np+j ];
				
			//Share boundaries to adjacent procs, send/receive upper bound
			MPI_Send(&u[np], np, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD);
			MPI_Recv(&u[0], np, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD, &status);
			
			if(myid != numprocs-1){ //intermediate block, send/receive also lower bound
				MPI_Send(&u[rows*np], np, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD);
				MPI_Recv(&u[(rows+1)*np], np, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD, &status);
			}
			
		    break;
	    case 1: // RED-BLACK
		    residual = relax_redblack(u, np, np);
		    break;
	    case 2: // GAUSS
			//printf("Iter");
		    residual = relax_gauss(u, rows+2, np);
			//printf("Residual -> %f", residual);
			//Share boundaries to adjacent procs, send/receive upper bound
			
			MPI_Send(&u[np], np, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD);
			MPI_Recv(&u[0], np, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD, &status);
			
			if(myid != numprocs-1){ //intermediate block, send/receive also lower bound
				MPI_Send(&u[rows*np], np, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD);
				MPI_Recv(&u[(rows+1)*np], np, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD, &status);
			}
		    break;
	    }

        iter++;
		
		double temp_residual = residual;
		//with all reduce all processes get the correct residual, finishing at the same time
		MPI_Allreduce(&temp_residual, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
		// solution good enough ?
        if (residual < 0.00005) break;

        // max. iteration reached ? (no limit with maxiter=0)
        if (maxiter>0 && iter>=maxiter) break;
    }
	MPI_Send(&u[0], (rows+2)*(np), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

    if( u ) free(u); if( uhelp ) free(uhelp);

    fprintf(stdout, "Process %d finished computing %d iterations with residual value = %f\n", myid, iter, residual);

    MPI_Finalize();
    exit(0);
  }
}
