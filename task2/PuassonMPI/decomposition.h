#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include <mpi.h>

#include "accessories.h"

// Structure which contains process location info, region and neighbors
typedef struct procinfo {
	// Location of process in process grid
	int coords[2];

	// Neighbors of the process
	int left, right, up, down;

	// Process rank
	int rank;

	// Process local domain size
	int n0, n1;
	// Process domain start and end point
	int st[2], en[2];

	// Process real kept domain
	int rn0, rn1;

} ProcInfo;

// Returns log_{2}(Number) if it is integer. If not it returns (-1). 
int IsPower(int Number);

// Splitting procedure of proc. number p. The integer p0 is calculated such that abs(N0/p0 - N1/(p-p0)) --> min.
int SplitFunction(int N0, int N1, int p);

// Performs domain decomposition on processes
void DomainDecomp(int N0, int N1, MPI_Comm *Grid_Comm, ProcInfo *info);

#endif