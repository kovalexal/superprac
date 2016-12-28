#ifndef TASK_H
#define TASK_H

#include "accessories.h"
#include "decomposition.h"

// Considered region.
// extern const double A0, A1;
// extern const double B0, B1;
const double A0 = -2.0, A1 = 2.0;
const double B0 = -2.0, B1 = 2.0;

// Returns analytical solution at (x, y)
double Solution(double x, double y);

// Returns boundary value at (x, y)
double BoundaryValue(double x, double y);

// Fills array with task right part
void RightPart(double * rhs, int N0, int N1, double h0, double h1, double A0, double B0, ProcInfo *info);

#endif