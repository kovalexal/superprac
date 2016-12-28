#ifndef SOLUTION_H
#define SOLUTION_H

#include "accessories.h"
#include "decomposition.h"
#include "task.h"

// Calculates residual vector r(k) = Ax(k)-f
void CalculateResidualVector(double *lResVect, double *lSolVect, double *lRHS_Vect, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

// Calculates dot product (v1(k), v2(k))
double CalculateProduct(double *lVect1, double *lVect2, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

// Calculates dot product (Av1(k), v2(k))
double CalculateProductLaplasian(double *lVect1, double *lVect2, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

// Recalculates next g(k)
void CalculateBasisVect(double *lBasisVect, double *lResVect, double alpha, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

// Calculates next solution x(k+1)
double CalculateNextSolution(double *lSolVect, double *lVect, double tau, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

// Calculates residual value between Solution and SolVect
double CalculateResidual(double *lSolVect, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info);

// Performs process values exchanging in grid
void Exchange(double *M, MPI_Comm *Grid_Comm, ProcInfo *info);

// Gathers submatrices to process with rank 0
void Gather(double *M, double *D, int N0, int N1, MPI_Comm *Grid_comm, ProcInfo *info);

// Solves task
void Solve(int N0, int N1, double A0, double A1, double B0, double B1, int SDINum, int CGMNum, MPI_Comm *Grid_Comm, ProcInfo *info);

#endif