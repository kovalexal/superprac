#include "solution.h"

void CalculateResidualVector(double *lResVect, double *lSolVect, double *lRHS_Vect, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	int i, j;

	for (i = 1; i < info->rn0 - 1; ++i)
		for (j = 1; j < info->rn1 - 1; ++j)
			lResVect[i*(info->rn1) + j] = LeftPart(lSolVect, i, j, info->rn0, info->rn1, h0, h1) - lRHS_Vect[i*(info->rn1) + j];
}

double CalculateProduct(double *lVect1, double *lVect2, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	int i, j;
	double tmp = 0.0, tmp_l = 0.0;

	for (i = 1; i < info->rn0 - 1; ++i)
		for (j = 1; j < info->rn1 - 1; ++j)
			tmp_l += lVect1[(info->rn1)*i + j] * lVect2[(info->rn1)*i + j] * h0*h1;
	MPI_Allreduce(&tmp_l, &tmp, 1, MPI_DOUBLE, MPI_SUM, *Grid_Comm);

	return tmp;
}

double CalculateProductLaplasian(double *lVect1, double *lVect2, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	int i, j;
	double tmp = 0.0, tmp_l = 0.0;

	for (i = 1; i < info->rn0 - 1; ++i)
		for (j = 1; j < info->rn1 - 1; ++j)
			tmp_l += LeftPart(lVect1, i, j, info->rn0, info->rn1, h0, h1) * lVect2[i*(info->rn1) + j] * h0*h1;
	MPI_Allreduce(&tmp_l, &tmp, 1, MPI_DOUBLE, MPI_SUM, *Grid_Comm);

	return tmp;
}

void CalculateBasisVect(double *lBasisVect, double *lResVect, double alpha, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	int i, j;

	for (i = 1; i < info->rn0 - 1; ++i)
		for (j = 1; j < info->rn1 - 1; ++j)
			lBasisVect[i*(info->rn1) + j] = lResVect[i*(info->rn1) + j] - alpha*lBasisVect[i*(info->rn1) + j];
}

double CalculateNextSolution(double *lSolVect, double *lVect, double tau, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	int i, j;
	double err = 0.0, err_l = 0.0, NewValue = 0.0;

	for (i = 1; i < info->rn0 - 1; ++i) {
		for (j = 1; j < info->rn1 - 1; ++j) {
			NewValue = lSolVect[i*(info->rn1) + j] - tau*lVect[i*(info->rn1) + j];
			err_l = Max(err_l, fabs(NewValue - lSolVect[i*(info->rn1) + j]));
			lSolVect[i*(info->rn1) + j] = NewValue;
		}
	}
	MPI_Allreduce(&err_l, &err, 1, MPI_DOUBLE, MPI_MAX, *Grid_Comm);

	return err;
}

double CalculateResidual(double *lSolVect, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	int i, j, ci, cj;
	double err = 0.0, err_l = 0.0;

	for (i = 1; i < info->rn0 - 1; ++i) {
		ci = info->st[0] - 1 + i;
		for (j = 1; j < info->rn1 - 1; ++j) {
			cj = info->st[1] - 1 + j;
			err_l = Max(err_l, fabs(Solution(x(ci, h0, A0), y(cj, h1, B0)) - lSolVect[i*(info->rn1) + j]));
		}
	}
	MPI_Allreduce(&err_l, &err, 1, MPI_DOUBLE, MPI_MAX, *Grid_Comm);

	return err;
}

void Exchange(double *M, MPI_Comm *Grid_Comm, ProcInfo *info) {
	MPI_Status status;
	MPI_Datatype ROW, COLUMN;

	// Create column and row type
	//MPI_Type_contiguous(info->n1, MPI_DOUBLE, &ROW);
	MPI_Type_contiguous(info->rn1, MPI_DOUBLE, &ROW);
	MPI_Type_commit(&ROW);
	//MPI_Type_vector(info->n0, 1, N1, MPI_DOUBLE, &COLUMN);
	MPI_Type_vector(info->rn0, 1, info->rn1, MPI_DOUBLE, &COLUMN);
	MPI_Type_commit(&COLUMN);

	// Pointers to send and receive rows and columns
	double *sendUpRow = &(M[info->rn1]);
	double *recvUpRow = &(M[0]);
	double *sendDownRow = &(M[info->rn1 * (info->rn0 - 2)]);
	double *recvDownRow = &(M[info->rn1 * (info->rn0 - 1)]);
	double *sendLeftColumn = &(M[1]);
	double *recvLeftColumn = &(M[0]);
	double *sendRightColumn = &(M[info->rn1 - 2]);
	double *recvRightColumn = &(M[info->rn1 - 1]);

	/************************************************************************/
	/* UP-DOWN and DOWN-UP                                                  */
	/************************************************************************/
	if ((info->up < 0) && (info->down >= 0)) {
		MPI_Sendrecv(sendDownRow, 1, ROW, info->down, 0, recvDownRow, 1, ROW, info->down, 0, *Grid_Comm, &status);
	}
	else if ((info->up >= 0) && (info->down >= 0)) {
		MPI_Sendrecv(sendUpRow, 1, ROW, info->up, 0, recvUpRow, 1, ROW, info->up, 0, *Grid_Comm, &status);
		MPI_Sendrecv(sendDownRow, 1, ROW, info->down, 0, recvDownRow, 1, ROW, info->down, 0, *Grid_Comm, &status);
	}
	else if ((info->up >= 0) && (info->down < 0)) {
		MPI_Sendrecv(sendUpRow, 1, ROW, info->up, 0, recvUpRow, 1, ROW, info->up, 0, *Grid_Comm, &status);
	}

	/************************************************************************/
	/* LEFT-RIGHT and RIGHT-LEFT                                            */
	/************************************************************************/
	if ((info->left < 0) && (info->right >= 0)) {
		MPI_Sendrecv(sendRightColumn, 1, COLUMN, info->right, 0, recvRightColumn, 1, COLUMN, info->right, 0, *Grid_Comm, &status);
	}
	else if ((info->left >= 0) && (info->right >= 0)) {
		MPI_Sendrecv(sendLeftColumn, 1, COLUMN, info->left, 0, recvLeftColumn, 1, COLUMN, info->left, 0, *Grid_Comm, &status);
		MPI_Sendrecv(sendRightColumn, 1, COLUMN, info->right, 0, recvRightColumn, 1, COLUMN, info->right, 0, *Grid_Comm, &status);

	}
	else if ((info->left >= 0) && (info->right < 0)) {
		MPI_Sendrecv(sendLeftColumn, 1, COLUMN, info->left, 0, recvLeftColumn, 1, COLUMN, info->left, 0, *Grid_Comm, &status);
	}

	return;
}

void Gather(double *M, double *D, int N0, int N1, MPI_Comm *Grid_comm, ProcInfo *info) {
	int i, j, k;
	int max_n0 = 0, max_n1 = 0;

	// Get world size
	int worldSize = 0;
	MPI_Comm_size(*Grid_comm, &worldSize);

	// Get maximum size of submatrix
	MPI_Allreduce(&(info->n0), &max_n0, 1, MPI_INT, MPI_MAX, *Grid_comm);
	MPI_Allreduce(&(info->n1), &max_n1, 1, MPI_INT, MPI_MAX, *Grid_comm);

	// Create local submatrix copy (first 4 elements are st[0], st[1], en[0], en[1])
	int submatrixWInfoSize = (4 + max_n0*max_n1);
	double *submatrixWInfo = (double *)malloc(submatrixWInfoSize * sizeof(double));
	memset(submatrixWInfo, 0, submatrixWInfoSize * sizeof(double));
	submatrixWInfo[0] = (double)info->st[0]; submatrixWInfo[1] = (double)info->st[1];
	submatrixWInfo[2] = (double)info->en[0], submatrixWInfo[3] = (double)info->en[1];
	double *submatrix = submatrixWInfo + 4;
	for (i = 0; i < info->n0; ++i) {
		double *submatrixRow = submatrix + i*info->n1;
		double *submatrixRowSrc = &M[(i + 1) * info->rn1 + 1];
		memcpy(submatrixRow, submatrixRowSrc, info->n1 * sizeof(double));
	}

	// Gather data from all processes
	double *gatheredData = NULL;
	if (info->rank == 0)
		gatheredData = (double *)malloc(submatrixWInfoSize*worldSize * sizeof(double));
	MPI_Gather(submatrixWInfo, submatrixWInfoSize, MPI_DOUBLE, gatheredData, submatrixWInfoSize, MPI_DOUBLE, 0, *Grid_comm);

	// Process zero copies to local matrix
	if (info->rank == 0) {
		for (k = 0; k < worldSize; ++k) {
			double *currSubmatrixWInfo = gatheredData + k*submatrixWInfoSize;
			double *currSubmatrix = currSubmatrixWInfo + 4;

			int st[2], en[2], n0, n1;
			st[0] = (int)currSubmatrixWInfo[0]; st[1] = (int)currSubmatrixWInfo[1];
			en[0] = (int)currSubmatrixWInfo[2]; en[1] = (int)currSubmatrixWInfo[3];
			n0 = en[0] - st[0]; n1 = en[1] - st[1];

			for (i = 0; i < n0; ++i) {
				double *currSubmatrixRow = currSubmatrix + i*n1;
				double *currSubmatrixRowDst = &D[(i + st[0]) * N1 + st[1]];
				memcpy(currSubmatrixRowDst, currSubmatrixRow, n1 * sizeof(double));
			}
		}
	}

	// Free memory
	free(submatrixWInfo);
	if (info->rank == 0)
		free(gatheredData);
}

void Solve(int N0, int N1, double A0, double A1, double B0, double B1, int SDINum, int CGMNum, MPI_Comm *Grid_Comm, ProcInfo *info)
{
	// Mesh steps on (ox) and (oy) axes
	double h0 = (A1 - A0) / (N0 - 1), h1 = (B1 - B0) / (N1 - 1);
	
	// The global solution array
	double *SolVect = NULL;
	// The local solution array
	double *lSolVect = NULL;
	// The local residual array
	double *lResVect = NULL;
	// The local vector of A-orthogonal system in CGM
	double *lBasisVect = NULL;
	// The local right hand side of Puasson equation
	double *lRHS_Vect = NULL;
	// Auxiliary values
	double sp, alpha, tau, err, residual;
	// The current iteration number
	int counterSDI = 0, counterCGM = 0;
	// Current number of processes
	int worldSize = 0;
	MPI_Comm_size(*Grid_Comm, &worldSize);

	int i, j;
	char str[127];
	FILE *fp, *fpErr;

	double startTime = 0.0, totalTime = 0.0;

	// Print information
	if (info->rank == 0) {
		// Open log file
		sprintf(str, "PuassonMPI_ECGM_%dx%d_%d.log", N0, N1, worldSize);
		fp = fopen(str, "w");
		fprintf(fp, "The Domain: [%f,%f]x[%f,%f], number of points: N[A0,A1] = %d, N[B0,B1] = %d;\n"
			"The steep descent iterations number: %d\n"
			"The conjugate gradient iterations number: %d\n",
			A0, A1, B0, B1, N0, N1, SDINum, CGMNum);

		// Open error file
		sprintf(str, "PuassonMPI_ECGM_%dx%d_%d.res", N0, N1, worldSize);
		fpErr = fopen(str, "w");
	}

	// Initialize local solution array with boundary values
	lSolVect = (double *)malloc((info->rn0)*(info->rn1) * sizeof(double));
	memset(lSolVect, 0, (info->rn0)*(info->rn1) * sizeof(double));
	// Fill UP
	if (info->st[0] - 1 == 0)
		for (j = 0; j < info->rn1; ++j)
			lSolVect[j] = BoundaryValue(A0, y(info->st[1] - 1 + j, h1, B0));
	// Fill DOWN
	if (info->en[0] + 1 == N1)
		for (j = 0; j < info->rn1; ++j)
			lSolVect[(info->rn0 - 1)*info->rn1 + j] = BoundaryValue(A1, y(info->st[1] - 1 + j, h1, B0));
	// Fill LEFT
	if (info->st[1] - 1 == 0)
		for (i = 0; i < info->rn0; ++i)
			lSolVect[info->rn1 * i] = BoundaryValue(x(info->st[0] - 1 + i, h0, A0), B0);
	// Fill RIGHT
	if (info->en[1] + 1 == N0)
		for (i = 0; i < info->rn0; ++i)
			lSolVect[info->rn1 * i + (info->rn1 - 1)] = BoundaryValue(x(info->st[0] - 1 + i, h0, A0), B1);

	// Initialize global solution array with boundary values
	if (info->rank == 0) {
		SolVect = (double *)malloc(N0*N1 * sizeof(double));
		memset(SolVect, 0, N0*N1 * sizeof(double));
		for (j = 0; j < N1; ++j) {
			SolVect[j] = BoundaryValue(A0, y(j, h1, B0));
			SolVect[(N0 - 1)*N1 + j] = BoundaryValue(A1, y(j, h1, B0));
		}
		for (i = 0; i < N0; ++i) {
			SolVect[i*N1] = BoundaryValue(x(i, h0, A0), B0);
			SolVect[i*N1 + (N0 - 1)] = BoundaryValue(x(i, h0, A0), B1);
		}
	}

	// Initialize local r_{ij}
	lResVect = (double *)malloc((info->rn0)*(info->rn1) * sizeof(double));
	memset(lResVect, 0, (info->rn0)*(info->rn1) * sizeof(double));

	// Initialize local p_{ij}
	lRHS_Vect = (double *)malloc((info->rn0)*(info->rn1) * sizeof(double));
	RightPart(lRHS_Vect, N0, N1, h0, h1, A0, B0, info);

	// Steep decent iterations
#ifdef SolutionPrint
	residual = CalculateResidual(lSolVect, h0, h1, Grid_Comm, info);
	if (info->rank == 0) {
		fprintf(fp, "\nNo iterations have been performed. The residual error is %.12f\n", residual);
		printf("\nSteep descent iterations begin ...\n");
	}
#endif
	startTime = MPI_Wtime();
	for (counterSDI = 1; counterSDI <= SDINum; ++counterSDI) {
		// The residual vector r(k) = Ax(k)-f is calculating ...
		CalculateResidualVector(lResVect, lSolVect, lRHS_Vect, h0, h1, Grid_Comm, info);

		// The value of product (r(k),r(k)) is calculating ...
		tau = CalculateProduct(lResVect, lResVect, h0, h1, Grid_Comm, info);
		Exchange(lResVect, Grid_Comm, info);

		// The value of product sp = (Ar(k),r(k)) is calculating ...
		sp = CalculateProductLaplasian(lResVect, lResVect, h0, h1, Grid_Comm, info);
		tau = tau / sp;

		// The x(k+1) is calculating ...
		err = CalculateNextSolution(lSolVect, lResVect, tau, h0, h1, Grid_Comm, info);
		Exchange(lSolVect, Grid_Comm, info);

		totalTime += MPI_Wtime() - startTime;
#ifdef SolutionPrint
		if (counterSDI % PrintStep == 0) {
			residual = CalculateResidual(lSolVect, h0, h1, Grid_Comm, info);
			if (info->rank == 0) {
				printf("The Steep Descent iteration %d has been performed.\n", counterSDI);
				fprintf(fp, "\nThe Steep Descent iteration k = %d has been performed.\n"
					"Step \\tau(k) = %f.\nThe difference value is estimated by %.12f.\n", \
					counterSDI, tau, err);
				fprintf(fp, "The Steep Descent iteration %d have been performed. "
					"The residual error is %.12f\n", counterSDI, residual);
			}
		}
#endif
		startTime = MPI_Wtime();
	}
	counterSDI--;

	// g(0) = r(k-1)
	lBasisVect = lResVect;
	lResVect = (double *)malloc((info->rn0)*(info->rn1) * sizeof(double));
	memset(lResVect, 0, (info->rn0)*(info->rn1) * sizeof(double));

	// CGM iterations
#ifdef SolutionPrint
	if (info->rank == 0)
		printf("\nCGM iterations begin ...\n");
#endif
	startTime = MPI_Wtime();
	for (counterCGM = 1; counterCGM <= CGMNum; ++counterCGM) {
		// The residual vector r(k) is calculating ...
		CalculateResidualVector(lResVect, lSolVect, lRHS_Vect, h0, h1, Grid_Comm, info);
		Exchange(lResVect, Grid_Comm, info);

		// The value of product (Ar(k),g(k-1)) is calculating ...
		alpha = CalculateProductLaplasian(lResVect, lBasisVect, h0, h1, Grid_Comm, info);
		alpha = alpha / sp;

		// The new basis vector g(k) is being calculated ..
		CalculateBasisVect(lBasisVect, lResVect, alpha, h0, h1, Grid_Comm, info);

		// The value of product (r(k),g(k)) is being calculated ...
		tau = CalculateProduct(lResVect, lBasisVect, h0, h1, Grid_Comm, info);
		Exchange(lBasisVect, Grid_Comm, info);

		// The value of product sp = (Ag(k),g(k)) is being calculated ...
		sp = CalculateProductLaplasian(lBasisVect, lBasisVect, h0, h1, Grid_Comm, info);
		tau = tau / sp;

		// The x(k+1) is being calculated ...
		err = CalculateNextSolution(lSolVect, lBasisVect, tau, h0, h1, Grid_Comm, info);
		Exchange(lSolVect, Grid_Comm, info);

		totalTime += MPI_Wtime() - startTime;
#ifdef SolutionPrint
		residual = CalculateResidual(lSolVect, h0, h1, Grid_Comm, info);

		if (info->rank == 0)
			fprintf(fpErr, "%d,%.12f\n", counterCGM, residual);

		if (counterCGM % PrintStep == 0) {
			if (info->rank == 0) {
				printf("The %d iteration of CGM method has been carried out.\n", counterCGM);
				fprintf(fp, "\nThe iteration %d of conjugate gradient method has been finished.\n"
					"The value of \\alpha(k) = %f, \\tau(k) = %f. The difference value is %f.\n", \
					counterCGM, alpha, tau, err);
				fprintf(fp, "The %d iteration of CGM have been performed. The residual error is %.12f\n", \
					counterCGM, residual);
			}
		}
#endif
		startTime = MPI_Wtime();

		// Stop iterations if error is small
		if (err < EPS)
			break;
	}
	counterCGM--;

	startTime = MPI_Wtime();
	// Gather results from all processes to rank 0
	Gather(lSolVect, SolVect, N0, N1, Grid_Comm, info);
	totalTime += MPI_Wtime() - startTime;

	if (info->rank == 0) {
		// printing some results ...
		fprintf(fp, "\nThe %d iterations are carried out. The error of iterations is estimated by %.12f.\n",
			counterSDI + counterCGM, err);
		fprintf(fp, "Total execution time is %f seconds.\n", totalTime);
		fclose(fp);

		sprintf(str, "PuassonMPI_ECGM_%dx%d_%d.dat", N0, N1, worldSize);
		fp = fopen(str, "w");
		fprintf(fp, "# This is the conjugate gradient method for descrete Puasson equation.\n"
			"# A0 = %f, A1 = %f, B0 = %f, B1 = %f, N[A0,A1] = %d, N[B0,B1] = %d, SDINum = %d, CGMNum = %d.\n"
			"# One can draw it by gnuplot by the command: splot 'MyPath\\FileName.dat' with lines\n", \
			A0, A1, B0, B1, N0, N1, SDINum, CGMNum);
		for (j = 0; j < N1; j++)
		{
			for (i = 0; i < N0; i++)
				fprintf(fp, "\n%f %f %f", x(i, h0, A0), y(j, h1, B0), SolVect[N0*j + i]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	free(SolVect);
	free(lSolVect);
	free(lResVect);
	free(lBasisVect);
	free(lRHS_Vect);
}