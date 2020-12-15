#include "heat.h"
#include <omp.h>
#define NB 8

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;
  
    nbx = NB;
    //using this it gets worse... high latency function?
    //nbx = omp_get_num_threads();//Use num threads in order to get better performance
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby; 
    #pragma omp parallel for private(diff) reduction(+:sum)
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
					     u[ i*sizey     + (j+1) ]+  // right
				             u[ (i-1)*sizey + j     ]+  // top
				             u[ (i+1)*sizey + j     ]); // bottom
	            diff = utmp[i*sizey+j] - u[i*sizey + j];
	            sum += diff * diff; 
	        }

    return sum;
}

/*
 * Blocked Red-Black solver: one iteration step
 */
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    int ii;
    #pragma omp parallel
    {
    // Computing "Red" blocks
    #pragma omp for schedule(static, 1) private(diff, lsw, unew) reduction(+:sum)
    for (int ii=0; ii<nbx; ii++) {
        lsw = ii%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey + j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    // Computing "Black" blocks
    #pragma omp for schedule(static,1) private(diff, lsw, unew) reduction(+:sum)
    for (ii=0; ii<nbx; ii++) { 
        lsw = (ii+1)%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }
    }
    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    int items[nbx][nby];
    #pragma omp parallel
    #pragma omp single
    {
    for (int ii=0; ii<nbx; ii++){
		//printf("Esta es mi ii %i\n", ii);
        for (int jj=0; jj<nby; jj++){
		#pragma omp task firstprivate(ii, jj) private(diff, unew) depend(out: items[ii][jj]) depend(in: items[ii-1][jj], items[ii][jj-1])
		{
			//printf("running task %i, %i | depend %i, %i \n", ii,jj, ii-1, jj-1);
			for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++){
				for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
					unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
						  u[ i*sizey	+ (j+1) ]+  // right
						  u[ (i-1)*sizey	+ j     ]+  // top
						  u[ (i+1)*sizey	+ j     ]); // bottom
					diff = unew - u[i*sizey+ j];
					#pragma omp atomic
					sum += diff * diff; 
					u[i*sizey+j]=unew;
				}
			}
			//printf("Block %i / %i done\n", ii,jj);
		}
		}
    }
    #pragma omp taskwait
    }
	//printf("finish loop\n");
    return sum;
}

