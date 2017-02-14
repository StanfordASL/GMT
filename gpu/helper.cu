/*
Author: brian ichter
Helper functions like printing, averaging, multiplying, etc.
*/

#include "helper.cuh"

void printArray(float* array, int dim1, int dim2, std::ostream& stream)
{
	for (int i = 0; i < dim1; ++i) {
		for (int j = 0; j < dim2; ++j) {
			stream << array[i*dim2 + j] << ",";
		}
		stream << std::endl;
	}
}

void printArray(int* array, int dim1, int dim2, std::ostream& stream)
{
	for (int i = 0; i < dim1; ++i) {
		for (int j = 0; j < dim2; ++j) {
			stream << array[i*dim2 + j] << ",";
		}
		stream << std::endl;
	}
}

void printArray(bool* array, int dim1, int dim2, std::ostream& stream)
{
	for (int i = 0; i < dim1; ++i) {
		for (int j = 0; j < dim2; ++j) {
			stream << array[i*dim2 + j] << ",";
		}
		stream << std::endl;
	}
}

void printArray(char* array, int dim1, int dim2, std::ostream& stream)
{
	for (int i = 0; i < dim1; ++i) {
		for (int j = 0; j < dim2; ++j) {
			stream << array[i*dim2 + j] << ",";
		}
		stream << std::endl;
	}
}

float avgArray(float* array, int length)
{
	int n = 0;
	float sum = 0;
	for (int i = 0; i < length; ++i) {
		if (array[i] > 0.0001) {
			++n;
			sum += array[i];
		}
	}
	std::cout << "sum is " << sum << " and " << n <<  " of " << length << " remain" << std::endl;
 	return sum/n;
}

void copyArray(float* arrayTo, float* arrayFrom, int length)
{
	for (int i = 0; i < length; ++i) {
		arrayTo[i] = arrayFrom[i];
	}
}

float calculateUnitBallVolume(int dim) 
{
	if (dim == 0) 
		return 1;
	else if (dim == 1)
		return 2;
	return 2*M_PI/dim*calculateUnitBallVolume(dim-2);
}

float calculateConnectionBallRadius(int dim, int samplesCount)
{
	float coverage = 0.0;
	float eta = 0.5;
	float dim_recip = 1.0/dim;
	float gammaval = (1.0+eta)*2.0*std::pow(dim_recip,dim_recip)*std::pow(((1-coverage)/calculateUnitBallVolume(dim)),dim_recip);
	float scalingFn = log((float) samplesCount)/samplesCount;
	return gammaval * std::pow(scalingFn, dim_recip);
}

void multiplyArrays(float* A, float* B, float* C, int rowsA, int colsA, int rowsB, int colsB)
{
	if (colsA != rowsB)
		std::cout << " ERROR: Matrix dimensions do not match: [" << rowsA << "x" << colsA << "] * [" << rowsB << "x" << colsB << "]" << std::cout;
	int inners = colsA;
	for (int i = 0; i < rowsA; ++i)
	    for (int j = 0; j < colsB; ++j) {
		    C[i*colsB+j]=0;
		    for (int k = 0; k < inners; ++k)
		    	C[i*colsB+j] += A[i*inners+k]*B[k*colsB+j];
		}
}

void scalarMultiplyArray(float *A, float b, float *C, int rows, int cols) {
	for (int i = 0; i < rows; ++i)
	    for (int j = 0; j < cols; ++j)
		    C[i*cols+j] = b*A[i*cols+j];
}

void subtractArrays(float* A, float* B, float* C, int rows, int cols)
{
	for (int i = 0; i < rows; ++i)
	    for (int j = 0; j < cols; ++j)
		    C[i*cols+j] = A[i*cols+j] - B[i*cols+j];
}

void horizConcat(float* A, float *B, float *C, int nA, int nB, int m)
{
	int n = nA + nB;
	for (int i = 0; i < m; ++i) {
	    for (int j = 0; j < nA; ++j)
	    	C[i*n+j] = A[i*nA+j];
	    for (int j = nA; j < n; ++j)
	    	C[i*n+j] = B[i*nB+(j-nA)];
	}
}

void vertConcat(float* A, float *B, float *C, int n, int mA, int mB) 
{
	int m = mA + mB;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < mA; ++j)
			C[j*n+i] = A[j*n+i];
		for (int j = mA; j < m; ++j)
			C[j*n+i] = B[(j-mA)*n+i];
	}
}

void transpose(float* A, int rows, int cols) 
{
	float trans[rows*cols]; 
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			trans[i + j*rows] = A[j + i*cols];	
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			A[j + i*cols] = trans[j + i*cols];
}

void transpose(float* Atrans, float* A, int rows, int cols) 
{
	float trans[rows*cols]; 
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			trans[i + j*rows] = A[j + i*cols];	
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			Atrans[j + i*cols] = trans[j + i*cols];
}

int printSolution(int samplesCount, float *d_samples, int *d_edges, float *d_costs) 
{
	std::ofstream solnFile;
	solnFile.open("soln.txt");

	int edges[samplesCount];
	float costs[samplesCount];
	float samples[samplesCount*DIM];
	cudaMemcpy(edges, d_edges, sizeof(int)*samplesCount, cudaMemcpyDeviceToHost);
	cudaMemcpy(costs, d_costs, sizeof(float)*samplesCount, cudaMemcpyDeviceToHost);
	cudaMemcpy(samples, d_samples, sizeof(float)*DIM*samplesCount, cudaMemcpyDeviceToHost);
	solnFile << "edges = [";
	printArray(edges, 1, samplesCount, solnFile);
	solnFile << "];" << std::endl << "samples = [";
	printArray(samples, samplesCount, DIM, solnFile);
	solnFile << "];" << std::endl << "costs = [";
	printArray(costs, 1, samplesCount, solnFile);
	solnFile << "];" << std::endl;

	solnFile.close();
	return 0;
}