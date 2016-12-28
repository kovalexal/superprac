#include <stdio.h>
#include "decomposition.h"
#include "task.h"
#include "solution.h"

int main(int argc, char **argv) {
	// The number of internal points on axes (ox) and (oy).
	int N0 = atoi(argv[1]), N1 = atoi(argv[2]);
	// Number of steep descent and CGM iterations.
	int SDINum = atoi(argv[3]), CGMNum = atoi(argv[4]);

	// A handler of a new grid communicator
	MPI_Comm Grid_Comm;
	// Information of current process
	ProcInfo info;

	// Initialize MPI environment
	MPI_Init(&argc, &argv);

	// Perform MPI decomposition
	DomainDecomp(N0, N1, &Grid_Comm, &info);
	
	// Solve task
	Solve(N0, N1, A0, A1, B0, B1, SDINum, CGMNum, &Grid_Comm, &info);

	MPI_Finalize();
}