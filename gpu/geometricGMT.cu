
// filename: geometricGMT.cu
// author: brian ichter
/*
This is an old deprecated file that runs GMT for geometric planning. It is included here to show an example of planning with an arbitrary lambda.
*/

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "helper.cu"

__constant__ float obstaclesGlobal[2*3*DIM];

const int nnSize = NUM/(20/DIM); // size of the nearest neighbor precomp matrix
bool activeKernel = true;

__device__ bool broadphaseValidQ(float *bb_min, float *bb_max, float *obs, float* debugOutput);
__device__ bool motionValidQ(float *v, float *w, float *obs, float* debugOutput);
__device__ bool faceContainsProjection(float *v, float *v_to_w, float lambda, int j, float *obs, float* debugOutput);
__device__ bool isMotionValid(int v_idx, int w_idx, int obstaclesCount, 
	float* obstacles, float* samples, float* debugOutput);
__global__ void fillWavefrontActive(int samplesCount, int *activeWavefrontIdx, int *wavefrontIdx, bool* wavefrontMask);

__global__ 
void setupArrays(bool* wavefront, bool* wavefrontNew, bool* wavefrontWas, bool* unvisited, float *costGoal, float* costs, int samplesCount, int* edges)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	if (node >= samplesCount) 
		return;

	unvisited[node] = true;
	wavefrontNew[node] = false;
	wavefrontWas[node] = false;
	if (node == 0) {		
		wavefront[node] = true;
		*costGoal = 0;
	} else {
		wavefront[node] = false;
	}
	costs[node] = 0;
	edges[node] = -1;
}

__global__ 
void sampleFree(float* obstacles, int obstaclesCount, float* samples, bool* isFreeSamples, float *debugOutput) 
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	if (node >= NUM) 
		return;

	float nodeLoc[DIM];
	for (int d = 0; d < DIM; ++d) {
		nodeLoc[d] = samples[node*DIM+d];
	}

	for (int obs_idx = 0; obs_idx < obstaclesCount; ++obs_idx) {
		bool notFree = true;
		for (int d = 0; d < DIM; ++d) {
			notFree = notFree && 
			nodeLoc[d] > obstaclesGlobal[obs_idx*2*DIM + d] && 
			nodeLoc[d] < obstaclesGlobal[obs_idx*2*DIM + DIM + d];
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
void setupPrecompNN(float *distances, int *nnEdges, int samplesCount)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount) 
		return;

	for (int nn = 0; nn < nnSize; ++nn) {
		distances[node*nnSize + nn] = 0;
		nnEdges[node*nnSize + nn] = -1;
	}
}

__global__ 
void precompNN(float r2, float *samples, float *distances, int *nnEdges, int samplesCount)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount) 
		return;

	// load up the node's location
	float nodeLoc[DIM];
	for (int d = 0; d < DIM; ++d) {
		nodeLoc[d] = samples[node*DIM+d];
	}

	// find if part of next wavefront and cost to come
	int nnIdx = 0;
	for (int sample = 0; sample < samplesCount; ++sample) {
		if (nnIdx >= nnSize)
			return; // prevent having too many neighbors for my array size (really should just code the array size more robustly)
		if (sample == node)
			continue;
		float distance2 = 0;
		for (int d = 0; d < DIM; ++d) {
			float difference = nodeLoc[d] - samples[sample*DIM+d];
			distance2 += difference * difference;
		}
		if (distance2 < r2) {
			nnEdges[node*nnSize + nnIdx] = sample;
			distances[node*nnSize + nnIdx] = sqrt(distance2);
			++nnIdx;
		}
	}
}

__global__
void expandWavefront(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal,
	float *distances, int *nnEdges,
	float* debugOutput)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	// const int obsSize = 3*DIM*2;
	// __shared__ float obstaclesShared[obsSize];
	// if (threadIdx.x < obsSize) {
	// 	obstaclesShared[threadIdx.x] = obstaclesGlobal[threadIdx.x];
	// }

	if (node >= samplesCount)
		return;
	if (!unvisited[node])
		return;

	// mark if in current wavefront
	if (wavefront[node]) {
		wavefrontWas[node] = true;
		return;
	}

	for (int i = 0; i < nnSize; ++i) {
		int nnIdx = nnEdges[node*nnSize + i];
		if (nnIdx == -1)
			break;
		if (wavefront[nnIdx]) {
			float costTmp = costs[nnIdx] + distances[node*nnSize + i];
			if (costs[node] == 0 || costTmp < costs[node]) {
				costs[node] = costTmp;
				edges[node] = nnIdx;
			}
			wavefrontNew[node] = true;
		}
	}

	if (wavefrontNew[node] && !isMotionValid(node, edges[node], obstaclesCount, obstacles,
		samples, debugOutput)) {
		costs[node] = 0;
		edges[node] = -1;
		wavefrontNew[node] = false;
	}
}

__global__
void updateWavefront(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal,
	float *distances, int *nnEdges, 
	float* debugOutput)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount)
		return;
	if (!unvisited[node])
		return;

	if (wavefrontWas[node]) {
		wavefront[node] = false; // remove from the current wavefront
		unvisited[node] = false; // remove from unvisited
	} else if (wavefrontNew[node]) {
		wavefront[node] = true; // add to wavefront
		if (node == goal_idx)
			*costGoal = costs[node];
	}
}

/* 
begin active kernel definitions
*/

__global__
void fillWavefrontActive(int samplesCount, int *wavefrontActiveIdx, int *wavefrontScanIdx, bool* wavefront)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount)
		return;
	if (!wavefront[node])
		return;

	wavefrontActiveIdx[wavefrontScanIdx[node]] = node;
}

__global__
void findWavefrontActive(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal,
	float *distances, int *nnEdges, int wavefrontSize, int* wavefrontActiveIdx,
	float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;

	int node = wavefrontActiveIdx[tid];

	if (wavefront[node])
	{
		wavefrontWas[node] = true;
		for (int i = 0; i < nnSize; ++i) {
			int nnIdx = nnEdges[node*nnSize + i];
			if (nnIdx == -1)
				return;
			if (unvisited[nnIdx] && !wavefront[nnIdx]) {
				wavefrontNew[nnIdx] = true;
			}
		}
	}
}

__global__
void findOptimalConnection(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal,
	float *distances, int *nnEdges, int wavefrontSize, int* wavefrontActiveIdx,
	float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;

	// const int obsSize = 3*DIM*2;
	// __shared__ float obstaclesShared[obsSize];
	// if (threadIdx.x < obsSize) {
	// 	obstaclesShared[threadIdx.x] = obstaclesGlobal[threadIdx.x];
	// }

	int node = wavefrontActiveIdx[tid];

	for (int i = 0; i < nnSize; ++i) {
		int nnIdx = nnEdges[node*nnSize + i];
		if (nnIdx == -1)
			break;
		if (wavefront[nnIdx]) {
			float costTmp = costs[nnIdx] + distances[node*nnSize + i];
			if (costs[node] == 0 || costTmp < costs[node]) {
				costs[node] = costTmp;
				edges[node] = nnIdx;
			}
		}
	}
}

__global__
void verifyExpansion(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal,
	float *distances, int *nnEdges, int wavefrontSize, int* wavefrontActiveIdx,
	float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;

	int node = wavefrontActiveIdx[tid];

	if (!isMotionValid(node, edges[node], obstaclesCount, obstacles,
		samples, debugOutput)) {
		costs[node] = 0;
		edges[node] = -1;
		wavefrontNew[node] = false;
	}
}


__global__
void updateWavefrontActive(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal,
	float *distances, int *nnEdges, 
	float* debugOutput)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount)
		return;
	if (!unvisited[node])
		return;

	if (wavefrontWas[node]) {
		wavefront[node] = false; // remove from the current wavefront
		unvisited[node] = false; // remove from unvisited
	} else if (wavefrontNew[node]) {
		wavefront[node] = true; // add to wavefront
		wavefrontNew[node] = false;
		if (node == goal_idx)
			*costGoal = costs[node];
	}
}

/********************
Device functions
********************/

__device__
bool isMotionValid(int v_idx, int w_idx, int obstaclesCount, 
	float* obstacles, float* samples, float *debugOutput)
{
	// TODO: eventually put each point (v, w) into shared memory
	// TODO: read http://http.developer.nvidia.com/GPUGems3/gpugems3_ch32.html
	// identify which obstacle this processor is checking against

	// the y index of the thread will inform which obstacle
	// int i = threadIdx.y; 

	// calculate bounds of the bounding box
	float v[DIM], w[DIM];
	float bb_min[DIM], bb_max[DIM];
	for (int d = 0; d < DIM; ++d) {
		v[d] = samples[v_idx*DIM + d];
		w[d] = samples[w_idx*DIM + d];

		if (v[d] > w[d]) {
			
			bb_min[d] = w[d];
			bb_max[d] = v[d];
		} else {
			bb_min[d] = v[d];
			bb_max[d] = w[d];
		}
	}

	// if split by obstacles, load in here then basically run the interior of the following for loop

	// go through each obstacle and do broad then narrow phase collision checking
	for (int obs_idx = 0; obs_idx < obstaclesCount; ++obs_idx) {
		float obs[DIM*2];
		for (int d = 0; d < DIM; ++d) {
			obs[d] = obstacles[obs_idx*2*DIM + d];
			obs[DIM+d] = obstacles[obs_idx*2*DIM + DIM + d];
		}
		if (!broadphaseValidQ(bb_min, bb_max, obs, debugOutput)) {
			if (!motionValidQ(v, w, obs, debugOutput)) {
				return false;
			}
		}
	}
	return true;
}

__device__
bool broadphaseValidQ(float *bbMin, float *bbMax, float *obs, float *debugOutput) 
{
	for (int d = 0; d < DIM/2; ++d) {
		if (bbMax[d] <= obs[d] || obs[DIM+d] <= bbMin[d]) 
			return true;
	}
	return false;
}

__device__
bool motionValidQ(float *v, float *w, float *obs, float *debugOutput) 
{
	float v_to_w[DIM/2];

	for (int d = 0; d < DIM/2; ++d) {
		float lambda;
		v_to_w[d] = w[d] - v[d];
		if (v[d] < obs[d]) {
			lambda = (obs[d] - v[d])/v_to_w[d];
		} else {
			lambda = (obs[DIM + d] - v[d])/v_to_w[d];
		}
		if (faceContainsProjection(v, w, lambda, d, obs, debugOutput))
			return false;
	}
	return true;
}

__device__
bool faceContainsProjection(float *v, float *w, float lambda, int j, float *obs, 
	float* debugOutput)
{
	for (int d = 0; d < DIM/2; ++d) {
		float projection = v[d] + (w[d] - v[d])*lambda;
		if (d != j && !(obs[d] <= projection && projection <= obs[DIM+d]))
			return false;
	}
	return true;
}

/********************
GPU Setup and Runner
********************/
__host__ 
void run_wfGMT(float *obstacles, int obstaclesCount, float *costs, float *times, float *initial, float *goal)
{
	/****************
	Timing details 
	timed actions: sampleFree, wavefront expansion (optimal connection, CC), termination check
	untimed actions: sample generation, nearest neighbor (emulate precomputed NN)
	****************/

	double cost_tot	= 0;
	int itrs_tot = 0;
	double t_tot_overall = 0;
	double t_tot_sampleFree = 0;
	double t_tot_precomp = 0;
	double t_tot_setup = 0;
	double t_tot_loop = 0;
	double t_tot_expand = 0;
	double t_tot_update = 0;
	double t_tot_term = 0;
	double ms = 1000;

	for (int run = 0; run < RUNS; ++run) {
		std::cout << run << "	" << std::flush;
		if (run%10 == 9)
			std::cout << std::endl;
		// create samples
		float samplesAll[DIM*NUM];
		createSamples(DIM, run, NUM, samplesAll, initial, goal);
		
		// copy samplesAll to device
		float *d_samplesAll, *d_obstacles;
		bool *d_isFreeSamples;
		cudaMalloc(&d_samplesAll, sizeof(float)*DIM*NUM);
		cudaMalloc(&d_obstacles, sizeof(float)*2*obstaclesCount*DIM);
		cudaMalloc(&d_isFreeSamples, sizeof(bool)*NUM);
		cudaMemcpy(d_samplesAll, samplesAll, sizeof(float)*DIM*NUM, cudaMemcpyHostToDevice);
		cudaMemcpy(d_obstacles, obstacles, sizeof(float)*2*obstaclesCount*DIM, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(obstaclesGlobal, obstacles, sizeof(float)*2*obstaclesCount*DIM);

		float *d_debugOutput2;
		cudaMalloc(&d_debugOutput2, sizeof(float)*NUM);

		// sampleFree
		// create thrust vector of samplesAll
		// sampleFree creates a mask of valid points
		// remove copy if with thrust
		const int blockSizeSF = 192;
		const int gridSizeSF = std::min((NUM + blockSizeSF - 1) / blockSizeSF, 65535);
		double t_sampleFreeMask_start = std::clock();
		sampleFree<<<gridSizeSF, blockSizeSF>>>(d_obstacles, obstaclesCount, d_samplesAll, d_isFreeSamples, d_debugOutput2);
		cudaDeviceSynchronize();
		double t_sampleFreeMask = std::clock() - t_sampleFreeMask_start;

		double t_sampleFree_start = std::clock();
		bool isFreeSamples[NUM];
		cudaMemcpy(isFreeSamples, d_isFreeSamples, sizeof(bool)*NUM, cudaMemcpyDeviceToHost);
		int idx = 0;
		for (int i = 0; i < NUM; ++i) {
			if (isFreeSamples[i])
				++idx;
		}
		int samplesCount = idx;
		int goal_idx = samplesCount-1;
		float samples[samplesCount*DIM];

		idx = 0;
		for (int i = 0; i < NUM; ++i) {
			if (isFreeSamples[i]) {
				for (int d = 0; d < DIM; ++d) 
					samples[idx*DIM + d] = samplesAll[i*DIM + d];
				++idx;
			}
		}
		
		cudaFree(d_samplesAll);
		double t_sampleFree = (std::clock() - t_sampleFree_start) / (double) CLOCKS_PER_SEC;

		bool printSampleFree = false;
		if (printSampleFree) {
			printArray(isFreeSamples,1,NUM);
			printArray(samples,NUM,DIM);
		}

		double t_sort_start = std::clock();
		// sort samples
		float heuristic[samplesCount*DIM];
		for (int i = 0; i < samplesCount; ++i) {
			float heur = 0;
			for (int d = 0; d < DIM; ++d) {
				float diff = samples[i*DIM + d] - initial[d];
				heur += diff*diff;
			}
			for (int d = 0; d < DIM; ++d) {
				heuristic[i*DIM + d] = heur;
			}
		}
		
		thrust::stable_sort_by_key(heuristic, heuristic + DIM*samplesCount, samples);
		float error = 100;
		for (int i = 0; i < samplesCount; ++i) {
			float tmp_error = 0;
			for (int d = 0; d < DIM; ++d) {
				float diff = samples[i*DIM + d] - goal[d];
				tmp_error += diff*diff;
			}
			if (tmp_error < error) {
				error = tmp_error;
				goal_idx = i;
			}
		}
		double t_sort = (std::clock() - t_sort_start) / (double) CLOCKS_PER_SEC;

		float *d_samples;
		cudaMalloc(&d_samples, sizeof(float)*DIM*samplesCount);
		cudaMemcpy(d_samples, samples, sizeof(float)*DIM*samplesCount, cudaMemcpyHostToDevice);

		float r = calculateConnectionBallRadius(DIM, samplesCount);
		// std::cout << " r is " << r << std::endl;
		float r2 = r*r;

		// create precomputation structure
		double t_precomp_start = std::clock();
		float* d_distances;
		int* d_nnEdges;
		cudaMalloc(&d_distances, sizeof(float)*samplesCount*nnSize);
		cudaMalloc(&d_nnEdges, sizeof(int)*samplesCount*nnSize);

		// create block sizes
		const int blockSize = 128;
		const int gridSize = std::min((samplesCount + blockSize - 1) / blockSize, 65535);
		setupPrecompNN<<<gridSize, blockSize>>>(d_distances, d_nnEdges, samplesCount);
		cudaDeviceSynchronize();
		
		precompNN<<<gridSize, blockSize>>>(r2, d_samples, d_distances, d_nnEdges, samplesCount);
		cudaDeviceSynchronize();
		
		double t_precomp = (std::clock() - t_precomp_start) / (double) CLOCKS_PER_SEC;

		double t_setup_start = std::clock();

		// create additional cuda arrays: wavefront, unvisited,
		// bool *d_wavefront, *d_wavefrontNew;
		bool *d_wavefrontWas, *d_unvisited;
		int *d_edges;
		float *d_costs, *d_debugOutput;
		float *d_costGoal;
		thrust::device_vector<bool> d_wavefront(samplesCount);
		thrust::device_vector<bool> d_wavefrontNew(samplesCount);

		// cudaMalloc(&d_wavefront, sizeof(bool)*samplesCount);
		// cudaMalloc(&d_wavefrontNew, sizeof(bool)*samplesCount);
		cudaMalloc(&d_wavefrontWas, sizeof(bool)*samplesCount);
		cudaMalloc(&d_unvisited, sizeof(bool)*samplesCount);
		cudaMalloc(&d_edges, sizeof(int)*samplesCount);
		cudaMalloc(&d_costs, sizeof(float)*samplesCount);
		cudaMalloc(&d_debugOutput, sizeof(float)*samplesCount);
		cudaMalloc(&d_costGoal, sizeof(float));

		if (d_unvisited == NULL) {
			std::cout << "Allocation Failure" << std::endl;
			exit(1);
		}

		// call __global__ setup- this will fill unvisited (all) and wavefront (false except the first node) correctly
		setupArrays<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontNew.data()), d_wavefrontWas, d_unvisited, d_costGoal, d_costs, samplesCount, d_edges);
		cudaDeviceSynchronize();
		double t_setup = (std::clock() - t_setup_start) / (double) CLOCKS_PER_SEC;

		// call kernels (nn, expand, terminate)
		float costGoal = 0;
		int maxItrs = 100;
		int itrs = 0;

		// TAG
		thrust::device_vector<int> d_wavefrontScanIdx(samplesCount);
		thrust::device_vector<int> d_wavefrontActiveIdx(samplesCount);

		// timing information
		double t_loop = 0;
		double t_expand = 0;
		double t_update = 0;
		double t_term = 0;
		double t_loop_start = std::clock();
		double t_activeKernel = 0;

		int maxWFSize = 0;
		bool printWavefrontSize = true;

		int activeSize = 0;

		while (itrs < maxItrs && costGoal == 0) {
			++itrs;

			if (activeKernel) {
				double t_activeKernel_start = std::clock();

				// prefix sum to fill wavefrontScanIdx to find points in the current band of the wavefront
				thrust::exclusive_scan(d_wavefront.begin(), d_wavefront.end(), d_wavefrontScanIdx.begin());
				// cudaDeviceSynchronize();

				// fill active wavefront
				fillWavefrontActive<<<gridSize, blockSize>>>(samplesCount,
					thrust::raw_pointer_cast(d_wavefrontActiveIdx.data()+1-1),
					thrust::raw_pointer_cast(d_wavefrontScanIdx.data()),
					thrust::raw_pointer_cast(d_wavefront.data()));
				// cudaDeviceSynchronize();

				activeSize = d_wavefrontScanIdx[samplesCount-1];
				if (d_wavefront[d_wavefront.size() - 1])
					++activeSize;

				std::cout << "new DIM = " << DIM << " Itr " << itrs << " size " << activeSize << std::endl;

				const int blockSizeActive = 128;
				const int gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 65535);
				findWavefrontActive<<<gridSizeActive, blockSizeActive>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontNew.data()), 
					d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges, activeSize, thrust::raw_pointer_cast(d_wavefrontActiveIdx.data()),
					d_debugOutput);
				// cudaDeviceSynchronize();

				// prefix sum to fill wavefrontScanIdx to connect to the next wavefront
				thrust::exclusive_scan(d_wavefrontNew.begin(), d_wavefrontNew.end(), d_wavefrontScanIdx.begin());
				// cudaDeviceSynchronize();

				fillWavefrontActive<<<gridSize, blockSize>>>(samplesCount,
					thrust::raw_pointer_cast(d_wavefrontActiveIdx.data()),
					thrust::raw_pointer_cast(d_wavefrontScanIdx.data()),
					thrust::raw_pointer_cast(d_wavefrontNew.data()));
				// cudaDeviceSynchronize();

				// increase by one if the last point in array is true because exclusive scan doesnt increment otherwise
				activeSize = d_wavefrontScanIdx[samplesCount-1]; 
				if (d_wavefrontNew[d_wavefrontNew.size() - 1])
					++activeSize;

				std::cout << "exp DIM = " << DIM << " Itr " << itrs << " size " << activeSize << std::endl;

				const int blockSizeActiveExp = 128;
				const int gridSizeActiveExp = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 65535);
				findOptimalConnection<<<gridSizeActiveExp, blockSizeActiveExp>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontNew.data()), 
					d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges, activeSize, thrust::raw_pointer_cast(d_wavefrontActiveIdx.data()),
					d_debugOutput);

				verifyExpansion<<<gridSizeActiveExp, blockSizeActiveExp>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontNew.data()), 
					d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges, activeSize, thrust::raw_pointer_cast(d_wavefrontActiveIdx.data()),
					d_debugOutput);
				// cudaDeviceSynchronize();

				t_activeKernel += std::clock() - t_activeKernel_start;

				double t_update_start = std::clock();
				updateWavefrontActive<<<gridSize, blockSize>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontNew.data()), 
					d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges,
					d_debugOutput);
				// cudaDeviceSynchronize(); // may not be necessary
				t_update += (std::clock() - t_update_start) / (double) CLOCKS_PER_SEC;
			} else {
				// thrust::host_vector<bool> h_wavefront(samplesCount);
				// thrust::copy(d_wavefront.begin(), d_wavefront.end(), h_wavefront.begin());
				// printArray(thrust::raw_pointer_cast(h_wavefront.data()),1,samplesCount);

				double t_expand_start = std::clock();
				expandWavefront<<<gridSize, blockSize>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontNew.data()), 
					d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges,
					d_debugOutput);
				cudaDeviceSynchronize(); // may not be necessary
				t_expand += (std::clock() - t_expand_start) / (double) CLOCKS_PER_SEC;

				double t_update_start = std::clock();
				updateWavefront<<<gridSize, blockSize>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontNew.data()), 
					d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges,
					d_debugOutput);
				cudaDeviceSynchronize(); // may not be necessary
				t_update += (std::clock() - t_update_start) / (double) CLOCKS_PER_SEC;
			}

			if (printWavefrontSize && run == 0) {
				// float debugOutput[samplesCount];
				// cudaMemcpy(debugOutput, d_debugOutput, samplesCount*sizeof(float), cudaMemcpyDeviceToHost);
				// std::cout << "debug output post exp: ";
				// printArray(debugOutput, 1, 20);

				bool debugWavefront[samplesCount];
				cudaMemcpy(debugWavefront, thrust::raw_pointer_cast(d_wavefront.data()), samplesCount*sizeof(bool), cudaMemcpyDeviceToHost);
				int wavefrontSize = 0;
				for(int i = 0; i < samplesCount; ++i) {
					if (debugWavefront[i])
						++wavefrontSize;
				}
				if (wavefrontSize > maxWFSize) {
					maxWFSize = wavefrontSize;
				}
				// std::cout << run << " itr " << itrs << " wavefront size: " << wavefrontSize << std::endl;
			}

			// std::cout << "Itr " << itrs << " call time is " << (std::clock() - t_kernel_start) / (double) CLOCKS_PER_SEC << std::endl;
			double t_term_start = std::clock();
			cudaMemcpy(&costGoal, d_costGoal, sizeof(float), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			t_term += (std::clock() - t_term_start) / (double) CLOCKS_PER_SEC;
		}
		t_loop = (std::clock() - t_loop_start) / (double) CLOCKS_PER_SEC;
		double t_overall = t_sampleFree + t_setup + t_loop;
		costs[run] = costGoal;
		times[run] = t_overall * ms;

		if (printWavefrontSize && run == 0) {
			std::cout << run << " max wavefront size: " << maxWFSize << std::endl;
		}
				
		// output ready to copy pasted into matlabViz/vizSoln.m
		bool printSoln = false;
		if (printSoln && run == 12) {
			int edges[samplesCount];
			float costs[samplesCount];
			cudaMemcpy(edges, d_edges, sizeof(int)*samplesCount, cudaMemcpyDeviceToHost);
			cudaMemcpy(costs, d_costs, sizeof(float)*samplesCount, cudaMemcpyDeviceToHost);
			std::cout << "edges = [";
			printArray(edges,1,samplesCount);
			std::cout << "];" << std::endl << "samples = [";
			printArray(samples,samplesCount,DIM);
			std::cout << "];" << std::endl << "costs = [";
			printArray(costs,1,samplesCount);
			std::cout << "];" << std::endl;
		}

		// free precomputed
		cudaFree(d_distances);
		cudaFree(d_nnEdges);

		// free cuda memory
		cudaFree(d_obstacles);
		cudaFree(d_samples);
		cudaFree(d_isFreeSamples);
		// cudaFree(d_wavefront);
		// cudaFree(d_wavefrontNew);
		cudaFree(d_wavefrontWas);
		cudaFree(d_unvisited);
		cudaFree(d_edges);
		cudaFree(d_costs);
		cudaFree(d_debugOutput);
		cudaFree(d_costGoal);
		cudaFree(d_debugOutput2);

		itrs_tot 			+= itrs;
		cost_tot 			+= costGoal;
		t_tot_overall 		+= t_overall;
		t_tot_sampleFree 	+= t_sampleFree;
		t_tot_precomp 		+= t_precomp;
		t_tot_setup 		+= t_setup;
		t_tot_loop 			+= t_loop;
		t_tot_expand 		+= t_expand;
		t_tot_update 		+= t_update;
		t_tot_term 			+= t_term;
	}

	std::cout << std::endl << "********* Final Results Averaged ********" << std::endl <<
		"Iterations: " << itrs_tot/((float) RUNS) << " and cost: " << cost_tot/((float) RUNS) << " and total time: " << t_tot_overall*ms/((float) RUNS) << std::endl;
	std::cout << "Timing breakdown--- " << std::endl << 
		"	sampleFree 	" << t_tot_sampleFree*ms/((float) RUNS) << std::endl << 
		"	precomp 	" << t_tot_precomp*ms/((float) RUNS) << std::endl << 
		"	setup 		" << t_tot_setup*ms/((float) RUNS) << std::endl << 
		"	loop 		" << t_tot_loop*ms/((float) RUNS) << std::endl <<
		"	  expand 		" << t_tot_expand*ms/((float) RUNS) << std::endl << 
		"	  update 		" << t_tot_update*ms/((float) RUNS) << std::endl << 
		"	  term 			" << t_tot_term*ms/((float) RUNS) << std::endl;
}

/********
GMT with lambda implementation
********/

__global__ 
void setupArraysBucket(bool* wavefront, bool* wavefrontBuckets, bool* wavefrontNew, bool* wavefrontWas, 
	bool* unvisited, float *costGoal, float* costs, int samplesCount, int* edges, int numBuckets, float *debugOutput)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	if (node >= samplesCount) 
		return;

	unvisited[node] = true;
	wavefrontNew[node] = false;
	wavefrontWas[node] = false;
	if (node == 0) {		
		wavefront[node] = true;
		wavefrontBuckets[node] = true;
		*costGoal = 0;
	} else {
		wavefront[node] = false;
		wavefrontBuckets[node] = false;
	}

	for (int i = 1; i < numBuckets; ++i) {
		wavefrontBuckets[samplesCount*i + node] = false;
	}

	costs[node] = 0;
	edges[node] = -1;
	debugOutput[node] = 0;
}

__global__
void expandWavefrontBucket(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontBuckets, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal,
	float *distances, int *nnEdges, int bucket,
	float* debugOutput)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	// const int obsSize = 3*DIM*2;
	// __shared__ float obstaclesShared[obsSize];
	// if (threadIdx.x < obsSize) {
	// 	obstaclesShared[threadIdx.x] = obstaclesGlobal[threadIdx.x];
	// }

	if (node >= samplesCount)
		return;
	if (!unvisited[node])
		return;

	// mark if in current wavefront bucket
	if (wavefrontBuckets[bucket*samplesCount + node]) { // in the currently expanding wavefront
		wavefrontWas[node] = true;
		return;
	} else if (wavefront[node]) {
		return; // in the wavefront but not this time
	}

	for (int i = 0; i < nnSize; ++i) {
		int nnIdx = nnEdges[node*nnSize + i];
		if (nnIdx == -1)
			break;
		if (wavefrontBuckets[bucket*samplesCount + nnIdx])
			wavefrontNew[node] = true;
		if (wavefront[nnIdx]) {
			float costTmp = costs[nnIdx] + distances[node*nnSize + i];
			if (costs[node] == 0 || costTmp < costs[node]) {
				costs[node] = costTmp;
				edges[node] = nnIdx;
			}
		}
	}

	if (!wavefrontNew[node] || !isMotionValid(node, edges[node], obstaclesCount, obstacles,
		samples, debugOutput)) {
		costs[node] = 0;
		edges[node] = -1;
		wavefrontNew[node] = false;
	}
}

__global__
void updateWavefrontBucket(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontBuckets, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal,
	float *distances, int *nnEdges, int bucket, float wavefrontLimit, float bucketWidth, int numBuckets,
	float* debugOutput)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount)
		return;
	if (!unvisited[node])
		return;

	if (wavefrontWas[node]) { // was previously in the wavefront
		wavefrontBuckets[samplesCount*bucket + node] = false; 
		wavefront[node] = false; 
		unvisited[node] = false; 
	} else if (wavefrontNew[node]) { // is now in the wavefront
		int newBucket = (costs[node] - wavefrontLimit)/bucketWidth + 1;
		if (newBucket < 1)
			newBucket = 1;
		wavefrontBuckets[((bucket+newBucket) % numBuckets)*samplesCount + node] = true; 
		wavefront[node] = true; 
		wavefrontNew[node] = false;
		if (node == goal_idx)
			*costGoal = costs[node];
	}
}

__global__
void findWavefrontActiveBucket(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontBuckets, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal,
	float *distances, int *nnEdges, int wavefrontSize, int* wavefrontActiveIdx, int bucket,
	float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;

	int node = wavefrontActiveIdx[tid];

	if (wavefrontBuckets[bucket*samplesCount + node]) {
		wavefrontWas[node] = true;
		for (int i = 0; i < nnSize; ++i) {
			int nnIdx = nnEdges[node*nnSize + i];
			if (nnIdx == -1)
				return;
			if (unvisited[nnIdx] && !wavefront[nnIdx]) {
				wavefrontNew[nnIdx] = true;
			}
		}
	}
}

__global__
void findOptimalConnectionBucket(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal,
	float *distances, int *nnEdges, int wavefrontSize, int* wavefrontActiveIdx,
	float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;

	// const int obsSize = 3*DIM*2;
	// __shared__ float obstaclesShared[obsSize];
	// if (threadIdx.x < obsSize) {
	// 	obstaclesShared[threadIdx.x] = obstaclesGlobal[threadIdx.x];
	// }

	int node = wavefrontActiveIdx[tid];

	for (int i = 0; i < nnSize; ++i) {
		int nnIdx = nnEdges[node*nnSize + i];
		if (nnIdx == -1)
			break;
		if (wavefront[nnIdx]) {
			float costTmp = costs[nnIdx] + distances[node*nnSize + i];
			if (costs[node] == 0 || costTmp < costs[node]) {
				costs[node] = costTmp;
				edges[node] = nnIdx;
			}
		}
	}
}

__global__
void verifyExpansionBucket(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontBuckets, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal,
	float *distances, int *nnEdges, int wavefrontSize, int* wavefrontActiveIdx,
	float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;

	int node = wavefrontActiveIdx[tid];

	if (!isMotionValid(node, edges[node], obstaclesCount, obstacles,
		samples, debugOutput)) {
		costs[node] = 0;
		edges[node] = -1;
		wavefrontNew[node] = false;
	}
}


__global__
void updateWavefrontActiveBucket(int samplesCount, int dim, int obstaclesCount, int goal_idx, float r2,
	bool *unvisited, bool *wavefront, bool *wavefrontBuckets, bool *wavefrontNew, bool *wavefrontWas, int* edges,
	float *samples, float* obstacles, float* costs, float* costGoal, 
	float *distances, int *nnEdges, float wavefrontLimit, float bucketWidth, int bucket, int numBuckets, 
	float* debugOutput)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount)
		return;
	if (!unvisited[node])
		return;

	if (wavefrontWas[node]) {
		wavefrontBuckets[samplesCount*bucket + node] = false; 
		wavefront[node] = false; 
		unvisited[node] = false; 
	} else if (wavefrontNew[node]) { // is now in the wavefront
		int newBucket = (costs[node] - wavefrontLimit)/bucketWidth + 1;
		if (newBucket < 1)
			newBucket = 1;
		wavefrontBuckets[((bucket+newBucket) % numBuckets)*samplesCount + node] = true; 
		wavefront[node] = true; 
		wavefrontNew[node] = false;
		if (node == goal_idx)
			*costGoal = costs[node];
	}
}

__host__ 
void run_GMT(float *obstacles, int obstaclesCount, float *costs, float *times, float *initial, float *goal, float lambda)
{
	/****************
	Timing details 
	timed actions: sampleFree, wavefront expansion (optimal connection, CC), termination check
	untimed actions: sample generation, nearest neighbor (emulate precomputed NN)
	****************/

	double cost_tot	= 0;
	int itrs_tot = 0;
	double t_tot_overall = 0;
	double t_tot_sampleFree = 0;
	double t_tot_precomp = 0;
	double t_tot_setup = 0;
	double t_tot_loop = 0;
	double t_tot_expand = 0;
	double t_tot_update = 0;
	double t_tot_term = 0;
	double ms = 1000;

	for (int run = 0; run < RUNS; ++run) {
		std::cout << run << "	" << std::flush;
		if (run%10 == 9)
			std::cout << std::endl;
		// create samples
		float samplesAll[DIM*NUM];
		createSamples(DIM, run, NUM, samplesAll, initial, goal);
		
		// copy samplesAll to device
		float *d_samplesAll, *d_obstacles;
		bool *d_isFreeSamples;
		cudaMalloc(&d_samplesAll, sizeof(float)*DIM*NUM);
		cudaMalloc(&d_obstacles, sizeof(float)*2*obstaclesCount*DIM);
		cudaMalloc(&d_isFreeSamples, sizeof(bool)*NUM);
		cudaMemcpy(d_samplesAll, samplesAll, sizeof(float)*DIM*NUM, cudaMemcpyHostToDevice);
		cudaMemcpy(d_obstacles, obstacles, sizeof(float)*2*obstaclesCount*DIM, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(obstaclesGlobal, obstacles, sizeof(float)*2*obstaclesCount*DIM);

		float *d_debugOutput2;
		cudaMalloc(&d_debugOutput2, sizeof(float)*NUM);

		// sampleFree
		// create thrust vector of samplesAll
		// sampleFree creates a mask of valid points
		// remove copy if with thrust
		const int blockSizeSF = 192;
		const int gridSizeSF = std::min((NUM + blockSizeSF - 1) / blockSizeSF, 65535);
		double t_sampleFreeMask_start = std::clock();
		sampleFree<<<gridSizeSF, blockSizeSF>>>(d_obstacles, obstaclesCount, d_samplesAll, d_isFreeSamples, d_debugOutput2);
		cudaDeviceSynchronize();
		double t_sampleFreeMask = std::clock() - t_sampleFreeMask_start;

		double t_sampleFree_start = std::clock();
		bool isFreeSamples[NUM];
		cudaMemcpy(isFreeSamples, d_isFreeSamples, sizeof(bool)*NUM, cudaMemcpyDeviceToHost);
		int idx = 0;
		for (int i = 0; i < NUM; ++i) {
			if (isFreeSamples[i])
				++idx;
		}
		int samplesCount = idx;
		int goal_idx = samplesCount-1;
		float samples[samplesCount*DIM];

		idx = 0;
		for (int i = 0; i < NUM; ++i) {
			if (isFreeSamples[i]) {
				for (int d = 0; d < DIM; ++d) 
					samples[idx*DIM + d] = samplesAll[i*DIM + d];
				++idx;
			}
		}
		
		cudaFree(d_samplesAll);
		double t_sampleFree = (std::clock() - t_sampleFree_start) / (double) CLOCKS_PER_SEC;

		bool printSampleFree = false;
		if (printSampleFree) {
			printArray(isFreeSamples,1,NUM);
			printArray(samples,NUM,DIM);
		}

		double t_sort_start = std::clock();
		// sort samples
		float heuristic[samplesCount*DIM];
		for (int i = 0; i < samplesCount; ++i) {
			float heur = 0;
			for (int d = 0; d < DIM; ++d) {
				float diff = samples[i*DIM + d] - initial[d];
				heur += diff*diff;
			}
			for (int d = 0; d < DIM; ++d) {
				heuristic[i*DIM + d] = heur;
			}
		}
		
		thrust::stable_sort_by_key(heuristic, heuristic + DIM*samplesCount, samples);
		float error = 100;
		for (int i = 0; i < samplesCount; ++i) {
			float tmp_error = 0;
			for (int d = 0; d < DIM; ++d) {
				float diff = samples[i*DIM + d] - goal[d];
				tmp_error += diff*diff;
			}
			if (tmp_error < error) {
				error = tmp_error;
				goal_idx = i;
			}
		}
		double t_sort = (std::clock() - t_sort_start) / (double) CLOCKS_PER_SEC;

		float *d_samples;
		cudaMalloc(&d_samples, sizeof(float)*DIM*samplesCount);
		cudaMemcpy(d_samples, samples, sizeof(float)*DIM*samplesCount, cudaMemcpyHostToDevice);

		float r = calculateConnectionBallRadius(DIM, samplesCount);
		// std::cout << " r is " << r << std::endl;
		float r2 = r*r;
		float bucketWidth = lambda*r;
		int numBuckets = 1/lambda + 1;

		// create precomputation structure
		double t_precomp_start = std::clock();
		float* d_distances;
		int* d_nnEdges;
		cudaMalloc(&d_distances, sizeof(float)*samplesCount*nnSize);
		cudaMalloc(&d_nnEdges, sizeof(int)*samplesCount*nnSize);

		// create block sizes
		const int blockSize = 256;
		const int gridSize = std::min((samplesCount + blockSize - 1) / blockSize, 65535);
		setupPrecompNN<<<gridSize, blockSize>>>(d_distances, d_nnEdges, samplesCount);
		cudaDeviceSynchronize();
		
		precompNN<<<gridSize, blockSize>>>(r2, d_samples, d_distances, d_nnEdges, samplesCount);
		cudaDeviceSynchronize();
		
		double t_precomp = (std::clock() - t_precomp_start) / (double) CLOCKS_PER_SEC;

		double t_setup_start = std::clock();

		// create additional cuda arrays: wavefront, unvisited,
		// bool *d_wavefront, *d_wavefrontNew;
		bool *d_wavefrontWas, *d_unvisited;
		int *d_edges;
		float *d_costs, *d_debugOutput;
		float *d_costGoal;
		thrust::device_vector<bool> d_wavefront(samplesCount);
		thrust::device_vector<bool> d_wavefrontBuckets(samplesCount*numBuckets);
		thrust::device_vector<bool> d_wavefrontNew(samplesCount);

		// cudaMalloc(&d_wavefront, sizeof(bool)*samplesCount);
		// cudaMalloc(&d_wavefrontNew, sizeof(bool)*samplesCount);
		cudaMalloc(&d_wavefrontWas, sizeof(bool)*samplesCount);
		cudaMalloc(&d_unvisited, sizeof(bool)*samplesCount);
		cudaMalloc(&d_edges, sizeof(int)*samplesCount);
		cudaMalloc(&d_costs, sizeof(float)*samplesCount);
		cudaMalloc(&d_debugOutput, sizeof(float)*samplesCount);
		cudaMalloc(&d_costGoal, sizeof(float));

		if (d_unvisited == NULL) {
			std::cout << "Allocation Failure" << std::endl;
			exit(1);
		}

		// call __global__ setup- this will fill unvisited (all) and wavefront (false except the first node) correctly
		setupArraysBucket<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontBuckets.data()),
			thrust::raw_pointer_cast(d_wavefrontNew.data()), d_wavefrontWas, d_unvisited, d_costGoal, d_costs, samplesCount, d_edges, numBuckets, d_debugOutput);
		cudaDeviceSynchronize();
		double t_setup = (std::clock() - t_setup_start) / (double) CLOCKS_PER_SEC;

		// call kernels (nn, expand, terminate)
		float costGoal = 0;
		int maxItrs = 100*numBuckets;
		int itrs = 0;

		// TAG
		thrust::device_vector<int> d_wavefrontScanIdx(samplesCount);
		thrust::device_vector<int> d_wavefrontActiveIdx(samplesCount);

		// timing information
		double t_loop = 0;
		double t_expand = 0;
		double t_update = 0;
		double t_term = 0;
		double t_loop_start = std::clock();
		double t_activeKernel = 0;

		int maxWFSize = 0;
		bool printWavefrontSize = true;

		int activeSize = 0;
		float wavefrontLimit = 0;

		while (itrs < maxItrs && costGoal == 0) {
			int bucket = itrs % numBuckets;
			// std::cout << "in bucket " << bucket << " of " << numBuckets << " with limit " << wavefrontLimit << std::endl;

			if (activeKernel) {
				double t_activeKernel_start = std::clock();

				// prefix sum to fill wavefrontScanIdx to find points in the current band of the wavefront
				thrust::exclusive_scan(d_wavefrontBuckets.begin() + bucket*samplesCount, 
					d_wavefrontBuckets.begin() + (bucket+1)*samplesCount, 
					d_wavefrontScanIdx.begin());
				// cudaDeviceSynchronize();

				// fill active wavefront
				fillWavefrontActive<<<gridSize, blockSize>>>(samplesCount,
					thrust::raw_pointer_cast(d_wavefrontActiveIdx.data()),
					thrust::raw_pointer_cast(d_wavefrontScanIdx.data()),
					thrust::raw_pointer_cast(d_wavefrontBuckets.data() + bucket*samplesCount));
				// cudaDeviceSynchronize();

				activeSize = d_wavefrontScanIdx[samplesCount-1];
				if (d_wavefront[d_wavefront.size() - 1])
					++activeSize;

				const int blockSizeActive = 256;
				const int gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 65535);
				findWavefrontActiveBucket<<<gridSizeActive, blockSizeActive>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontBuckets.data()), thrust::raw_pointer_cast(d_wavefrontNew.data()), 
					d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges, activeSize, thrust::raw_pointer_cast(d_wavefrontActiveIdx.data()), bucket,
					d_debugOutput);
				// cudaDeviceSynchronize();

				// prefix sum to fill wavefrontScanIdx to connect to the next wavefront
				thrust::exclusive_scan(d_wavefrontNew.begin(), d_wavefrontNew.end(), d_wavefrontScanIdx.begin());
				// cudaDeviceSynchronize();

				fillWavefrontActive<<<gridSize, blockSize>>>(samplesCount,
					thrust::raw_pointer_cast(d_wavefrontActiveIdx.data()),
					thrust::raw_pointer_cast(d_wavefrontScanIdx.data()),
					thrust::raw_pointer_cast(d_wavefrontNew.data()));
				// cudaDeviceSynchronize();

				// increase by one if the last point in array is true because exclusive scan doesnt increment otherwise
				activeSize = d_wavefrontScanIdx[samplesCount-1]; 
				if (d_wavefrontNew[d_wavefrontNew.size() - 1])
					++activeSize;

				if (printWavefrontSize && run == 0) {
					if (activeSize > maxWFSize) {
						maxWFSize = activeSize;
					}
				}

				const int blockSizeActiveExp = 256;
				const int gridSizeActiveExp = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 65535);
				findOptimalConnectionBucket<<<gridSizeActiveExp, blockSizeActiveExp>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontNew.data()), 
					d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges, activeSize, thrust::raw_pointer_cast(d_wavefrontActiveIdx.data()),
					d_debugOutput);

				verifyExpansionBucket<<<gridSizeActiveExp, blockSizeActiveExp>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontBuckets.data()), 
					thrust::raw_pointer_cast(d_wavefrontNew.data()), 
					d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges, activeSize, thrust::raw_pointer_cast(d_wavefrontActiveIdx.data()),
					d_debugOutput);
				// cudaDeviceSynchronize();

				t_activeKernel += std::clock() - t_activeKernel_start;

				double t_update_start = std::clock();
				updateWavefrontActiveBucket<<<gridSize, blockSize>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontBuckets.data()), 
					thrust::raw_pointer_cast(d_wavefrontNew.data()), d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges, wavefrontLimit, bucketWidth, bucket, numBuckets,
					d_debugOutput);
				// cudaDeviceSynchronize(); // may not be necessary
				t_update += (std::clock() - t_update_start) / (double) CLOCKS_PER_SEC;			
			} else {
				double t_expand_start = std::clock();
				expandWavefrontBucket<<<gridSize, blockSize>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontBuckets.data()), 
					thrust::raw_pointer_cast(d_wavefrontNew.data()), 
					d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges, bucket,
					d_debugOutput);
				cudaDeviceSynchronize(); // may not be necessary
				t_expand += (std::clock() - t_expand_start) / (double) CLOCKS_PER_SEC;

				// float debugOutput[samplesCount];
				// cudaMemcpy(debugOutput, d_debugOutput, samplesCount*sizeof(float), cudaMemcpyDeviceToHost);
				// std::cout << "debug output post exp: ";
				// printArray(debugOutput, 1, 100);

				double t_update_start = std::clock();
				updateWavefrontBucket<<<gridSize, blockSize>>>(samplesCount, DIM, obstaclesCount, goal_idx, r2,
					d_unvisited, thrust::raw_pointer_cast(d_wavefront.data()), thrust::raw_pointer_cast(d_wavefrontBuckets.data()),
					thrust::raw_pointer_cast(d_wavefrontNew.data()), 
					d_wavefrontWas, d_edges,
					d_samples, d_obstacles, d_costs, d_costGoal,
					d_distances, d_nnEdges, bucket, wavefrontLimit, bucketWidth, numBuckets,
					d_debugOutput);
				cudaDeviceSynchronize(); // may not be necessary
				t_update += (std::clock() - t_update_start) / (double) CLOCKS_PER_SEC;
			}

			bool debug = false;
			if (debug) {
				float debugOutput[samplesCount];
				cudaMemcpy(debugOutput, d_debugOutput, samplesCount*sizeof(float), cudaMemcpyDeviceToHost);
				std::cout << "debug output post exp: ";
				printArray(debugOutput, 1, 100);
			}

			// std::cout << "Itr " << itrs << " call time is " << (std::clock() - t_kernel_start) / (double) CLOCKS_PER_SEC << std::endl;
			double t_term_start = std::clock();
			cudaMemcpy(&costGoal, d_costGoal, sizeof(float), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			t_term += (std::clock() - t_term_start) / (double) CLOCKS_PER_SEC;

			wavefrontLimit += bucketWidth;
			++itrs;
		}
		t_loop = (std::clock() - t_loop_start) / (double) CLOCKS_PER_SEC;
		double t_overall = t_sampleFree + t_setup + t_loop;
		costs[run] = costGoal;
		times[run] = t_overall * ms;

		if (printWavefrontSize && run == 0) {
			std::cout << run << " max wavefront size: " << maxWFSize << std::endl;
		}
				
		// output ready to copy pasted into matlabViz/vizSoln.m
		bool printSoln = false;
		if (printSoln && run == 12) {
			int edges[samplesCount];
			float costs[samplesCount];
			cudaMemcpy(edges, d_edges, sizeof(int)*samplesCount, cudaMemcpyDeviceToHost);
			cudaMemcpy(costs, d_costs, sizeof(float)*samplesCount, cudaMemcpyDeviceToHost);
			std::cout << "edges = [";
			printArray(edges,1,samplesCount);
			std::cout << "];" << std::endl << "samples = [";
			printArray(samples,samplesCount,DIM);
			std::cout << "];" << std::endl << "costs = [";
			printArray(costs,1,samplesCount);
			std::cout << "];" << std::endl;
		}

		// free precomputed
		cudaFree(d_distances);
		cudaFree(d_nnEdges);

		// free cuda memory
		cudaFree(d_obstacles);
		cudaFree(d_samples);
		cudaFree(d_isFreeSamples);
		// cudaFree(d_wavefront);
		// cudaFree(d_wavefrontNew);
		cudaFree(d_wavefrontWas);
		cudaFree(d_unvisited);
		cudaFree(d_edges);
		cudaFree(d_costs);
		cudaFree(d_debugOutput);
		cudaFree(d_costGoal);
		cudaFree(d_debugOutput2);

		itrs_tot 			+= itrs;
		cost_tot 			+= costGoal;
		t_tot_overall 		+= t_overall;
		t_tot_sampleFree 	+= t_sampleFree;
		t_tot_precomp 		+= t_precomp;
		t_tot_setup 		+= t_setup;
		t_tot_loop 			+= t_loop;
		t_tot_expand 		+= t_expand;
		t_tot_update 		+= t_update;
		t_tot_term 			+= t_term;
	}

	std::cout << std::endl << "********* Final Results Averaged ********" << std::endl <<
		"Iterations: " << itrs_tot/((float) RUNS) << " and cost: " << cost_tot/((float) RUNS) << " and total time: " << t_tot_overall*ms/((float) RUNS) << std::endl;
	std::cout << "Timing breakdown--- " << std::endl << 
		"	sampleFree 	" << t_tot_sampleFree*ms/((float) RUNS) << std::endl << 
		"	precomp 	" << t_tot_precomp*ms/((float) RUNS) << std::endl << 
		"	setup 		" << t_tot_setup*ms/((float) RUNS) << std::endl << 
		"	loop 		" << t_tot_loop*ms/((float) RUNS) << std::endl <<
		"	  expand 		" << t_tot_expand*ms/((float) RUNS) << std::endl << 
		"	  update 		" << t_tot_update*ms/((float) RUNS) << std::endl << 
		"	  term 			" << t_tot_term*ms/((float) RUNS) << std::endl;
}