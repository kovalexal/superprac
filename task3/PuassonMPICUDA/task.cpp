#include "task.h"

double Solution(double x, double y)
{
	return 1 + sin(x*y);
}

double BoundaryValue(double x, double y)
{
	return 1 + sin(x*y);
}

int RightPartOLD(double * rhs, int N0, int N1, double h0, double h1, double A0, double B0)
{
	int i, j;
	memset(rhs, 0, N0*N1 * sizeof(double));

	for (i = 0; i < N0; ++i)
		for (j = 0; j < N1; ++j)
			rhs[i*N1 + j] = R2(x(i, h0, A0), y(j, h1, B0)) * sin(x(i, h0, A0) * y(j, h1, B0));

	return 0;
}

void RightPart(double * rhs, int N0, int N1, double h0, double h1, double A0, double B0, ProcInfo *info)
{
	int i, j, ci, cj;
	memset(rhs, 0, info->rn0*info->rn1 * sizeof(double));

	for (i = 0; i < info->rn0; ++i) {
		ci = info->st[0] - 1 + i;
		for (j = 0; j < info->rn1; ++j) {
			cj = info->st[1] - 1 + j;
			rhs[i*info->rn1 + j] = R2(x(ci, h0, A0), y(cj, h1, B0)) * sin(x(ci, h0, A0) * y(cj, h1, B0));
		}
	}
}