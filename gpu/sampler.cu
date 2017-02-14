/*
Author: brian ichter
Generate state space samples
*/

#include "sampler.cuh"

// TODO: add protection on initial and goal indexes
void createSamplesIID(int seed, float *samples, float *initial, float *goal, float *lo, float *hi) 
{ 
	std::srand(seed);
	for (int d = 0; d < DIM; ++d) {
		for (int i = 0; i < NUM; ++i) {
			samples[i*DIM + d] = ((float)std::rand())/RAND_MAX*(hi[d] - lo[d]) + lo[d];
		}
	}

	// replace goal and initial nodes
	for (int d = 0; d < DIM; ++d) {
		samples[d] = initial[d];
		samples[(NUM-1)*DIM + d] = goal[d];
	}
}

void createSamplesHalton(int skip, float *samples, float *initial, float *goal, float *lo, float *hi) 
{ 	
	int numPrimes = 25;
	if (skip + DIM > numPrimes) {
		std::cout << "in sampler.cu: skip in creating halton seq too high" << std::endl;
		return;
	}
	int bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 
		43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};

	for (int d = 0; d < DIM; ++d) {
		for (int n = 0; n < NUM; ++n) {
			samples[n*DIM + d] = localHaltonSingleNumber(n, bases[d + skip])*(hi[d] - lo[d]) + lo[d];
		}
	}

	// replace goal and initial nodes
	for (int d = 0; d < DIM; ++d) {
		samples[d] = initial[d];
		samples[(NUM-1)*DIM + d] = goal[d];
	}
}

float localHaltonSingleNumber(int n, int b) 
{
	float hn = 0;
	int n0 = n;
	float f = 1/((float) b);

	while (n0 > 0) {
		float n1 = n0/b;
		int r = n0 - n1*b;
		hn += f*r;
		f = f/b;
		n0 = n1;
	}
	return hn;
}



__global__ 
void sampleFree(float* obstacles, int obstaclesCount, float* samples, bool* isFreeSamples, float *debugOutput) 
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	if (node >= NUM) 
		return;

	float nodeLoc[3];
	for (int d = 0; d < 3; ++d) {
		nodeLoc[d] = samples[node*DIM+d];
	}

	for (int obs_idx = 0; obs_idx < obstaclesCount; ++obs_idx) {
		bool notFree = true;
		for (int d = 0; d < 3; ++d) {
			notFree = notFree && 
			nodeLoc[d] > obstacles[obs_idx*2*DIM + d] && 
			nodeLoc[d] < obstacles[obs_idx*2*DIM + DIM + d];
			if (!notFree)
				break;
		}
		if (notFree) {
			isFreeSamples[node] = false;
			return;
		}
	}
	isFreeSamples[node] = true;
}

__global__ 
void fillSamples(float* samples, float* samplesAll, int* sampleFreeIdx, bool* isFreeSamples, float *debugOutput) 
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	if (node >= NUM) 
		return;
	if (!isFreeSamples[node])
		return;

	for (int d = 0; d < DIM; ++d) {
		samples[sampleFreeIdx[node]*DIM+d] = samplesAll[node*DIM+d];
	}
}

__global__ 
void createSortHeuristic(float* samples, int initial_idx, float* heuristic, int samplesCount) 
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	if (node >= samplesCount) 
		return;

	float heuristicValue = 0;
	for (int d = 0; d < DIM; ++d) {
		float dist = samples[node*DIM + d] - samples[initial_idx*DIM + d];  
		heuristicValue += dist*dist;	
	}
	for (int d = 0; d < DIM; ++d) {
		heuristic[node*DIM+d] = heuristicValue;
	}
}

// returns false if not free, true if it is free (i.e., valid)
bool sampleFreePt(float* obstacles, int obstaclesCount, float* sample) 
{
	float nodeLoc[3];
	for (int d = 0; d < 3; ++d) {
		nodeLoc[d] = sample[d];
	}

	for (int obs_idx = 0; obs_idx < obstaclesCount; ++obs_idx) {
		bool notFree = true;
		for (int d = 0; d < 3; ++d) {
			notFree = notFree && 
			nodeLoc[d] > obstacles[obs_idx*2*DIM + d] && 
			nodeLoc[d] < obstacles[obs_idx*2*DIM + DIM + d];
			if (!notFree)
				break;
		}
		if (notFree) {
			return false;
		}
	}
	return true;
}
