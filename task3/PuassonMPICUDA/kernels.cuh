#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "accessories.h"
#include "decomposition.h"
#include "task.h"

typedef struct cudainfo {
	int device;
	cudaDeviceProp prop;
	int bytesFull, bytesReduce;
	double *reduceVect;
	double *dtmpVect, *dreduceVect;
	dim3 blocks, threads;
	int elemsPerThread;
} CUDAInfo;

void CUDAInit(ProcInfo *info);
void CUDAFinalize();

void CalculateResidualVector(double *dlResVect, double *dlSolVect, double *dlRHS_Vect, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

double CalculateProduct(double *dlVect1, double *dlVect2, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

double CalculateProductLaplasian(double *dlVect1, double *dlVect2, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

void CalculateBasisVect(double *dlBasisVect, double *dlResVect, double alpha, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

double CalculateNextSolution(double *dlSolVect, double *dlVect, double tau, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

double CalculateResidual(double *dlSolVect, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

#endif