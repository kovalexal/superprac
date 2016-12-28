#include "kernels.cuh"

// Internal data used
CUDAInfo cudaInfo;

int NOD(int a, int b)
{
    while(a > 0 && b > 0)
        if(a > b)
            a %= b;
        else
            b %= a;
    return a + b;
}

void CUDAInit(ProcInfo *info) {
	cudaInfo.device = info->rank % 2;
	cudaGetDeviceProperties(&cudaInfo.prop, cudaInfo.device);

	cudaSetDevice(cudaInfo.device);

	cudaInfo.elemsPerThread = (info->n0*info->n1 + cudaInfo.prop.multiProcessorCount*cudaInfo.prop.maxThreadsPerMultiProcessor - 1) / (cudaInfo.prop.multiProcessorCount*cudaInfo.prop.maxThreadsPerMultiProcessor);
	cudaInfo.threads = dim3(NOD(cudaInfo.prop.maxThreadsPerBlock, cudaInfo.prop.maxThreadsPerMultiProcessor));
	cudaInfo.blocks = dim3((info->n0*info->n1 + cudaInfo.elemsPerThread * cudaInfo.threads.x - 1) / (cudaInfo.elemsPerThread * cudaInfo.threads.x));

	cudaInfo.bytesFull = info->n0 * info->n1 * sizeof(double);
	cudaInfo.bytesReduce = cudaInfo.blocks.x * sizeof(double);

	cudaInfo.reduceVect = (double *)malloc(cudaInfo.bytesReduce);
	SAFE_CUDA(cudaMalloc((void **)&cudaInfo.dtmpVect, cudaInfo.bytesFull));
	SAFE_CUDA(cudaMalloc((void **)&cudaInfo.dreduceVect, cudaInfo.bytesReduce));
}

void CUDAFinalize() {
	free(cudaInfo.reduceVect);
	SAFE_CUDA(cudaFree(cudaInfo.dtmpVect));
	SAFE_CUDA(cudaFree(cudaInfo.dreduceVect));
}

__device__
double dSolution(double x, double y)
{
	return 1 + sin(x*y);
}

__global__
void kReduceSum(double *Vect, double *reduceVect, ProcInfo info, double elemsPerThread) {
	extern __shared__ double data[];

	int tid = threadIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int ls = (x / WARPSIZE * WARPSIZE) * (elemsPerThread - 1) + x;

	data[tid] = 0.0;
	for (int i = ls, j = 0; (j < elemsPerThread) && (i < info.n0*info.n1); i += WARPSIZE, ++j)
		data[tid] += Vect[i];

	__syncthreads();
	for (int s = blockDim.x/2; s > 0; s >>= 1) {
		if (tid < s)
			data[tid] += data[tid + s];
		__syncthreads();
	}

	if (tid == 0)
		reduceVect[blockIdx.x] = data[0];
}

__global__
void kReduceMax(double *Vect, double *reduceVect, ProcInfo info, double elemsPerThread) {
	extern __shared__ double data[];

	int tid = threadIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int ls = (x / WARPSIZE * WARPSIZE) * (elemsPerThread - 1) + x;

	data[tid] = 0.0;
	for (int i = ls, j = 0; (j < elemsPerThread) && (i < info.n0*info.n1); i += WARPSIZE, ++j)
		data[tid] = max(data[tid], Vect[i]);

	__syncthreads();
	for (int s = blockDim.x/2; s > 0; s >>= 1) {
		if (tid < s)
			data[tid] = max(data[tid], data[tid + s]);
		__syncthreads();
	}

	if (tid == 0)
		reduceVect[blockIdx.x] = data[0];
}

__global__
void kCalculateResidualVector(double *dlResVect, double *dlSolVect, double *dlRHS_Vect, int N0, int N1, double h0, double h1, ProcInfo info, int elemsPerThread) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int ls = (x / WARPSIZE * WARPSIZE) * (elemsPerThread - 1) + x;

	for (int i = ls, j = 0; (j < elemsPerThread) && (i < info.n0*info.n1); i += WARPSIZE, ++j) {
		int xl = 1 + i / info.n1, yl = 1 + i % info.n1;
		int idx = xl * info.rn1 + yl;

		dlResVect[idx] = LeftPart(dlSolVect, xl, yl, info.rn0, info.rn1, h0, h1) - dlRHS_Vect[idx];
	}
}

void CalculateResidualVector(double *dlResVect, double *dlSolVect, double *dlRHS_Vect, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	kCalculateResidualVector<<<cudaInfo.blocks, cudaInfo.threads>>>(dlResVect, dlSolVect, dlRHS_Vect, N0, N1, h0, h1, *info, cudaInfo.elemsPerThread);
	SAFE_CUDA(cudaDeviceSynchronize());
}

__global__
void kCalculateProduct(double *dlVect1, double *dlVect2, double *dtmpVect, int N0, int N1, double h0, double h1, ProcInfo info, int elemsPerThread) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int ls = (x / WARPSIZE * WARPSIZE) * (elemsPerThread - 1) + x;

	for (int i = ls, j = 0; (j < elemsPerThread) && (i < info.n0*info.n1); i += WARPSIZE, ++j) {
		int xl = 1 + i / info.n1, yl = 1 + i % info.n1;
		int idx = xl * info.rn1 + yl;

		dtmpVect[i] = dlVect1[idx] * dlVect2[idx] * h0*h1;
	}
}

double CalculateProduct(double *dlVect1, double *dlVect2, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	kCalculateProduct<<<cudaInfo.blocks, cudaInfo.threads>>>(dlVect1, dlVect2, cudaInfo.dtmpVect, N0, N1, h0, h1, *info, cudaInfo.elemsPerThread);
	SAFE_CUDA(cudaDeviceSynchronize());
	kReduceSum<<<cudaInfo.blocks, cudaInfo.threads, cudaInfo.threads.x * sizeof(double)>>>(cudaInfo.dtmpVect, cudaInfo.dreduceVect, *info, cudaInfo.elemsPerThread);
	SAFE_CUDA(cudaDeviceSynchronize());

	SAFE_CUDA(cudaMemcpy(cudaInfo.reduceVect, cudaInfo.dreduceVect, cudaInfo.bytesReduce, cudaMemcpyDeviceToHost));

	double tmp = 0.0, tmp_l = 0.0;
	for (int i = 0; i < cudaInfo.blocks.x; ++i)
		tmp_l += cudaInfo.reduceVect[i];

	MPI_Allreduce(&tmp_l, &tmp, 1, MPI_DOUBLE, MPI_SUM, *Grid_Comm);

	return tmp;
}

__global__
void kCalculateProductLaplasian(double *dlVect1, double *dlVect2, double *dtmpVect, int N0, int N1, double h0, double h1, ProcInfo info, int elemsPerThread) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int ls = (x / WARPSIZE * WARPSIZE) * (elemsPerThread - 1) + x;

	for (int i = ls, j = 0; (j < elemsPerThread) && (i < info.n0*info.n1); i += WARPSIZE, ++j) {
		int xl = 1 + i / info.n1, yl = 1 + i % info.n1;
		int idx = xl * info.rn1 + yl;
		
		dtmpVect[i] = LeftPart(dlVect1, xl, yl, info.rn0, info.rn1, h0, h1) * dlVect2[idx] * h0*h1;
	}
}

double CalculateProductLaplasian(double *dlVect1, double *dlVect2, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	kCalculateProductLaplasian<<<cudaInfo.blocks, cudaInfo.threads>>>(dlVect1, dlVect2, cudaInfo.dtmpVect, N0, N1, h0, h1, *info, cudaInfo.elemsPerThread);
	SAFE_CUDA(cudaDeviceSynchronize());
	kReduceSum<<<cudaInfo.blocks, cudaInfo.threads, cudaInfo.threads.x * sizeof(double)>>>(cudaInfo.dtmpVect, cudaInfo.dreduceVect, *info, cudaInfo.elemsPerThread);
	SAFE_CUDA(cudaDeviceSynchronize());

	SAFE_CUDA(cudaMemcpy(cudaInfo.reduceVect, cudaInfo.dreduceVect, cudaInfo.bytesReduce, cudaMemcpyDeviceToHost));

	double tmp = 0.0, tmp_l = 0.0;
	for (int i = 0; i < cudaInfo.blocks.x; ++i)
		tmp_l += cudaInfo.reduceVect[i];

	MPI_Allreduce(&tmp_l, &tmp, 1, MPI_DOUBLE, MPI_SUM, *Grid_Comm);

	return tmp;
}

__global__
void kCalculateBasisVect(double *dlBasisVect, double *dlResVect, double alpha, int N0, int N1, double h0, double h1, ProcInfo info, int elemsPerThread) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int ls = (x / WARPSIZE * WARPSIZE) * (elemsPerThread - 1) + x;

	for (int i = ls, j = 0; (j < elemsPerThread) && (i < info.n0*info.n1); i += WARPSIZE, ++j) {
		int xl = 1 + i / info.n1, yl = 1 + i % info.n1;
		int idx = xl * info.rn1 + yl;
		
		dlBasisVect[idx] = dlResVect[idx] - alpha * dlBasisVect[idx];
	}
}

void CalculateBasisVect(double *dlBasisVect, double *dlResVect, double alpha, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	kCalculateBasisVect<<<cudaInfo.blocks, cudaInfo.threads>>>(dlBasisVect, dlResVect, alpha, N0, N1, h0, h1, *info, cudaInfo.elemsPerThread);
	SAFE_CUDA(cudaDeviceSynchronize());
}

__global__
void kCalculateNextSolution(double *dlSolVect, double *dlVect, double *dtmpVect, double tau, int N0, int N1, double h0, double h1, ProcInfo info, int elemsPerThread) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int ls = (x / WARPSIZE * WARPSIZE) * (elemsPerThread - 1) + x;

	for (int i = ls, j = 0; (j < elemsPerThread) && (i < info.n0*info.n1); i += WARPSIZE, ++j) {
		int xl = 1 + i / info.n1, yl = 1 + i % info.n1;
		int idx = xl * info.rn1 + yl;
		
		double NewValue = dlSolVect[idx] - tau * dlVect[idx];
		dtmpVect[i] = fabs(NewValue - dlSolVect[idx]);
		dlSolVect[idx] = NewValue;
	}
}

double CalculateNextSolution(double *dlSolVect, double *dlVect, double tau, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	kCalculateNextSolution<<<cudaInfo.blocks, cudaInfo.threads>>>(dlSolVect, dlVect, cudaInfo.dtmpVect, tau, N0, N1, h0, h1, *info, cudaInfo.elemsPerThread);
	SAFE_CUDA(cudaDeviceSynchronize());
	kReduceMax<<<cudaInfo.blocks, cudaInfo.threads, cudaInfo.threads.x * sizeof(double)>>>(cudaInfo.dtmpVect, cudaInfo.dreduceVect, *info, cudaInfo.elemsPerThread);
	SAFE_CUDA(cudaDeviceSynchronize());

	SAFE_CUDA(cudaMemcpy(cudaInfo.reduceVect, cudaInfo.dreduceVect, cudaInfo.bytesReduce, cudaMemcpyDeviceToHost));

	double err = 0.0, err_l = 0.0;
	for (int i = 0; i < cudaInfo.blocks.x; ++i)
		err_l = max(err_l, cudaInfo.reduceVect[i]);

	MPI_Allreduce(&err_l, &err, 1, MPI_DOUBLE, MPI_MAX, *Grid_Comm);

	return err;
}

__global__
void kCalculateResiduals(double *dlSolVect, double *dtmpVect, int N0, int N1, double h0, double h1, ProcInfo info, double elemsPerThread) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int ls = (x / WARPSIZE * WARPSIZE) * (elemsPerThread - 1) + x;

	for (int i = ls, j = 0; (j < elemsPerThread) && (i < info.n0*info.n1); i += WARPSIZE, ++j) {
		int xl = 1 + i / info.n1, yl = 1 + i % info.n1;
		int xg = info.st[0] - 1 + xl, yg = info.st[1] - 1 + yl;
		int idx = xl * info.rn1 + yl;
		
		dtmpVect[i] = fabs(dSolution(x(xg, h0, A0), y(yg, h1, B0)) - dlSolVect[idx]);
	}
}

double CalculateResidual(double *dlSolVect, int N0, int N1, double h0, double h1, MPI_Comm *Grid_Comm, ProcInfo *info) {
	kCalculateResiduals<<<cudaInfo.blocks, cudaInfo.threads>>>(dlSolVect, cudaInfo.dtmpVect, N0, N1, h0, h1, *info, cudaInfo.elemsPerThread);
	SAFE_CUDA(cudaDeviceSynchronize());
	kReduceMax<<<cudaInfo.blocks, cudaInfo.threads, cudaInfo.threads.x * sizeof(double)>>>(cudaInfo.dtmpVect, cudaInfo.dreduceVect, *info, cudaInfo.elemsPerThread);
	SAFE_CUDA(cudaDeviceSynchronize());

	SAFE_CUDA(cudaMemcpy(cudaInfo.reduceVect, cudaInfo.dreduceVect, cudaInfo.bytesReduce, cudaMemcpyDeviceToHost));

	double err = 0.0, err_l = 0.0;
	for (int i = 0; i < cudaInfo.blocks.x; ++i)
		err_l = max(err_l, cudaInfo.reduceVect[i]);

	MPI_Allreduce(&err_l, &err, 1, MPI_DOUBLE, MPI_MAX, *Grid_Comm);

	return err;
}