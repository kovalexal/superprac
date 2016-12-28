#include "decomposition.h"

int IsPower(int Number)
{
	unsigned int M;
	int p;

	if (Number <= 0)
		return(-1);

	M = Number; p = 0;
	while (M % 2 == 0)
	{
		++p;
		M = M >> 1;
	}
	if ((M >> 1) != 0)
		return(-1);
	else
		return(p);

}

int SplitFunction(int N0, int N1, int p)
{
	float n0, n1;
	int p0, i;

	n0 = (float)N0; n1 = (float)N1;
	p0 = 0;

	for (i = 0; i < p; i++)
		if (n0 > n1)
		{
			n0 = n0 / (float)2.0;
			++p0;
		}
		else
			n1 = n1 / (float)2.0;

	return(p0);
}

void DomainDecomp(int N0, int N1, MPI_Comm *Grid_Comm, ProcInfo *info)
{
	// The number of processes and rank in communicator
	int ProcNum;
	// ProcNum = 2^(power), power splits into sum p0 + p1.
	int power, p0, p1;
	// dims[0] = 2^p0, dims[1] = 2^p1 (--> ProcNum = dims[0]*dims[1]).
	int dims[2];
	// N0 = info->n0*dims[0] + k0, N1 = info->n1*dims[1] + k1.
	int k0, k1;

	// The number of a process topology dimensions
	const int ndims = 2;
	// Is used for creating processes topology
	int periods[2] = { 0,0 };

	// MPI Library is being activated ...
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &(info->rank));

	// Check that grid size contains positive numbers
	if ((N0 <= 0) || (N1 <= 0))
	{
		if (info->rank == 0)
			printf("The first and the second arguments (mesh numbers) should be positive.\n");

		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	// Exclude boundaries
	N0 -= 2; N1 -= 2;

	// Check that number if processes is a power of 2
	if ((power = IsPower(ProcNum)) < 0)
	{
		if (info->rank == 0)
			printf("The number of procs must be a power of 2.\n");
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	// Find such numbers that procs=2^(p0+p1)
	p0 = SplitFunction(N0, N1, power);
	p1 = power - p0;

	// Find 2^p0 and 2^p1
	dims[0] = (unsigned int)1 << p0; dims[1] = (unsigned int)1 << p1;
	// Find n0=N0/(2^p0), n1 = N1/(2^p1)
	info->n0 = N0 >> p0; info->n1 = N1 >> p1;
	k0 = N0 - dims[0] * info->n0; k1 = N1 - dims[1] * info->n1;

#ifdef DecompositionPrint
	if (info->rank == 0)
	{
		printf("The number of processes ProcNum = 2^%d. It is split into %d x %d processes.\n"
			"The number of nodes N0 = %d, N1 = %d. Blocks B(i,j) have size:\n", power, dims[0], dims[1], N0, N1);

		if ((k0 > 0) && (k1 > 0))
			printf("-->\t %d x %d iff i = 0 .. %d, j = 0 .. %d;\n", info->n0 + 1, info->n1 + 1, k0 - 1, k1 - 1);
		if (k1 > 0)
			printf("-->\t %d x %d iff i = %d .. %d, j = 0 .. %d;\n", info->n0, info->n1 + 1, k0, dims[0] - 1, k1 - 1);
		if (k0 > 0)
			printf("-->\t %d x %d iff i = 0 .. %d, j = %d .. %d;\n", info->n0 + 1, info->n1, k0 - 1, k1, dims[1] - 1);

		printf("-->\t %d x %d iff i = %d .. %d, j = %d .. %d.\n", info->n0, info->n1, k0, dims[0] - 1, k1, dims[1] - 1);
	}
#endif

	// The cartesian topology of processes is being created ...
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, TRUE, Grid_Comm);
	MPI_Comm_rank(*Grid_Comm, &(info->rank));
	MPI_Cart_coords(*Grid_Comm, info->rank, ndims, info->coords);

	// Get process start end end points, local domain size
	info->st[0] = 1 + Min(k0, info->coords[0]) * (info->n0 + 1) + Max(0, (info->coords[0] - k0)) * info->n0;
	info->st[1] = 1 + Min(k1, info->coords[1]) * (info->n1 + 1) + Max(0, (info->coords[1] - k1)) * info->n1;
	info->en[0] = info->st[0] + info->n0 + (info->coords[0] < k0 ? 1 : 0);
	info->en[1] = info->st[1] + info->n1 + (info->coords[1] < k1 ? 1 : 0);
	info->n0 = info->en[0] - info->st[0]; info->n1 = info->en[1] - info->st[1];

	// Get process real kept domain
	info->rn0 = info->n0 + 2; info->rn1 = info->n1 + 2;

	// Get process neighbors
	MPI_Cart_shift(*Grid_Comm, 0, 1, &(info->up), &(info->down));
	MPI_Cart_shift(*Grid_Comm, 1, 1, &(info->left), &(info->right));

#ifdef DecompositionPrint
	printf("My rank in Grid_Comm is %d. My topological coords is (%d,%d). Domain size is %d x %d nodes.\n"
		"My neighbors: left = %d, right = %d, down = %d, up = %d. My start coords are (%d, %d), end coords are (%d, %d)\n",
		info->rank, info->coords[0], info->coords[1], info->n0, info->n1, info->left, info->right, info->down, info->up, info->st[0], info->st[1], info->en[0], info->en[1]);
#endif
}