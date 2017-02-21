/*
Author: brian ichter
Runs the group marching tree algorithm.
*/
#include "GMT.cuh"

const int numDiscInit = 8;

void GMT(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	float *d_distancesCome, int *d_nnGoEdges, int *d_nnComeEdges, int nnSize, float *d_discMotions, int *d_nnIdxs,
	float *d_samples, int samplesCount, bool *d_isFreeSamples, float r, int numDisc,
	float *d_costs, int *d_edges, int initial_idx, int goal_idx)
{
	// only implementating for lambda = 0.5 for now
	// numBuckets = 2;
	// double t_gmtStart = std::clock();
	// int maxItrs = 100; 
	
	// bool *d_wavefrontWas, *d_unvisited, *d_isCollision;
	// float *d_costGoal;
	// thrust::device_vector<float> d_debugOutput(samplesCount*numDisc);
	// thrust::device_vector<bool> d_wavefrontNew(samplesCount);
	// thrust::device_vector<int> d_wavefrontScanIdx(samplesCount);
	// thrust::device_vector<int> d_wavefrontActiveIdx(samplesCount);

	// thrust::device_vector<bool> d_wavefront(numBuckets*samplesCount);

	// float *d_debugOutput_ptr = thrust::raw_pointer_cast(d_debugOutput.data());
	// bool *d_wavefront_ptr = thrust::raw_pointer_cast(d_wavefront.data());
	// bool *d_wavefrontNew_ptr = thrust::raw_pointer_cast(d_wavefrontNew.data());
	// int *d_wavefrontScanIdx_ptr = thrust::raw_pointer_cast(d_wavefrontScanIdx.data());
	// int *d_wavefrontActiveIdx_ptr = thrust::raw_pointer_cast(d_wavefrontActiveIdx.data());

	// cudaMalloc(&d_wavefrontWas, sizeof(bool)*samplesCount);
	// cudaMalloc(&d_unvisited, sizeof(bool)*samplesCount);
	// cudaMalloc(&d_costGoal, sizeof(float));
	// cudaMalloc(&d_isCollision, sizeof(bool)*numDisc*samplesCount);

	// if (d_unvisited == NULL) {
	// 	std::cout << "Allocation Failure" << std::endl;
	// 	exit(1);
	// }

	// // setup array values
	// const int blockSize = 128;
	// const int gridSize = std::min((samplesCount + blockSize - 1) / blockSize, 2147483647);
	// setupArrays_buck<<<gridSize, blockSize>>>(d_wavefront_ptr, d_wavefrontNew_ptr, 
	// d_wavefrontWas, d_unvisited, d_isFreeSamples, d_costGoal, d_costs, d_edges, samplesCount);
}

// __global__ void setupArrays_buck(bool* wavefront, bool* wavefrontNew, bool* wavefrontWas, bool* unvisited, 
// 	bool *isFreeSamples, float *costGoal, float* costs, int* edges, int samplesCount)
// {
// 	int node = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (node >= samplesCount) 
// 		return;

// 	unvisited[node] = isFreeSamples[node];
// 	wavefrontNew[node] = false;
// 	wavefrontWas[node] = !isFreeSamples[node];
// 	if (node == 0) {		
// 		wavefront[node] = true;
// 		*costGoal = 0;
// 	} else {
// 		wavefront[node] = false;
// 	}
// 	wavefront[node + samplesCount] = false; // second bucket starts empty
// 	costs[node] = 0;
// 	edges[node] = -1;

// 	if (!isFreeSamples[node]) {
// 		costs[node] = -11;
// 		edges[node] = -2;
// 	}
// }


// ***************** takes new initial state not currently in the tree
void GMTinit(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	float *d_distancesCome, int *d_nnGoEdges, int *d_nnComeEdges, int nnSize, float *d_discMotions, int *d_nnIdxs,
	float *d_samples, int samplesCount, bool *d_isFreeSamples, float r, int numDisc,
	float *d_costs, int *d_edges, int initial_idx, int goal_idx)
{
	cudaError_t code;
	double t_gmtStart = std::clock();
	int maxItrs = 100; 
	
	bool *d_wavefrontWas, *d_unvisited, *d_isCollision;
	float *d_costGoal;
	thrust::device_vector<float> d_debugOutput(samplesCount*numDisc);
	thrust::device_vector<bool> d_wavefront(samplesCount);
	thrust::device_vector<bool> d_wavefrontNew(samplesCount);
	thrust::device_vector<int> d_wavefrontScanIdx(samplesCount);
	thrust::device_vector<int> d_wavefrontActiveIdx(samplesCount);

	float *d_debugOutput_ptr = thrust::raw_pointer_cast(d_debugOutput.data());
	bool *d_wavefront_ptr = thrust::raw_pointer_cast(d_wavefront.data());
	bool *d_wavefrontNew_ptr = thrust::raw_pointer_cast(d_wavefrontNew.data());
	int *d_wavefrontScanIdx_ptr = thrust::raw_pointer_cast(d_wavefrontScanIdx.data());
	int *d_wavefrontActiveIdx_ptr = thrust::raw_pointer_cast(d_wavefrontActiveIdx.data());

	cudaMalloc(&d_wavefrontWas, sizeof(bool)*samplesCount);
	cudaMalloc(&d_unvisited, sizeof(bool)*samplesCount);
	cudaMalloc(&d_costGoal, sizeof(float));
	cudaMalloc(&d_isCollision, sizeof(bool)*numDisc*samplesCount);

	if (d_unvisited == NULL) {
		std::cout << "Allocation Failure" << std::endl;
		exit(1);
	}

	int *d_edges_backup;
	cudaMalloc(&d_edges_backup, sizeof(int)*samplesCount);
	cudaMemcpy(d_edges_backup, d_edges, sizeof(int)*samplesCount, cudaMemcpyDeviceToDevice);

	// setup array values
	const int blockSize = 128;
	const int gridSize = std::min((samplesCount + blockSize - 1) / blockSize, 2147483647);
	setupArrays<<<gridSize, blockSize>>>(
		d_wavefront_ptr, d_wavefrontNew_ptr, 
		d_wavefrontWas, d_unvisited, d_isFreeSamples, d_costGoal, d_costs, d_edges, samplesCount);

	// attach initial condition (referred to as node number -2)
	// std::cout << "init" << initial[0] << ", " << initial[1] << ", " << initial[2] << ", " << initial[3] << std::endl;
	float *d_initial;
	cudaMalloc(&d_initial, sizeof(float)*DIM);
	cudaMemcpy(d_initial, initial, sizeof(float)*DIM, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	// std::cout << "attaching wavefront to "<< initial[0] << ", " << initial[1] << ", " << initial[2] << ", " << initial[3] << std::endl;
	attachWavefront<<<gridSize,blockSize>>>(
		samplesCount, d_samples, d_initial, 
		r, goal_idx, d_wavefront_ptr, d_edges,
		obstaclesCount, d_obstacles, 
		d_costs, d_costGoal, d_debugOutput_ptr);
	cudaDeviceSynchronize();
	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on attachWavefront: " << cudaGetErrorString(code) << std::endl; }
	// std::cout << "wavefront attached "<< initial[0] << ", " << initial[1] << ", " << initial[2] << ", " << initial[3] << std::endl;

	float costGoal = 0;
	int itrs = 0;
	int activeSize = 0;
	while (itrs < maxItrs && costGoal == 0)
	{ 
		++itrs;

		// std::cout << "scanning" << std::endl;

		// prefix sum to fill wavefrontScanIdx to find points in the current band of the wavefront
		thrust::exclusive_scan(d_wavefront.begin(), d_wavefront.end(), d_wavefrontScanIdx.begin());
		fillWavefront<<<gridSize, blockSize>>>(samplesCount,
			d_wavefrontActiveIdx_ptr, d_wavefrontScanIdx_ptr, d_wavefront_ptr);
		// std::cout << "scanned" << std::endl;

		activeSize = d_wavefrontScanIdx[samplesCount-1];
		// increase by one if the last point in array is true because exclusive scan doesnt increment otherwise
		(d_wavefront[d_wavefront.size() - 1]) ? ++activeSize : 0;

		// std::cout << "new DIM = " << DIM << " Itr " << itrs << " size " << activeSize << std::endl;

		if (activeSize == 0) {
			// failed to connect to the tree (often this is from the initial condition, just return the same tree that came in)
			cudaMemcpy(d_edges, d_edges_backup, sizeof(int)*samplesCount, cudaMemcpyDeviceToDevice);
			return;
		}

		const int blockSizeActive = 128;
		const int gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 2147483647);
		
		// std::cout << "find wavefront (" << blockSizeActive << ", " gridSizeActive << ")" << std::endl;
		findWavefront<<<gridSizeActive, blockSizeActive>>>(
			d_unvisited, d_wavefront_ptr, d_wavefrontNew_ptr, d_wavefrontWas, 
			d_nnGoEdges, nnSize, activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);


		// prefix sum to fill wavefrontScanIdx to connect to the next wavefront
		thrust::exclusive_scan(d_wavefrontNew.begin(), d_wavefrontNew.end(), d_wavefrontScanIdx.begin());
		
		// to print thrust vector) thrust::copy(d_wavefrontScanIdx.begin(), d_wavefrontScanIdx.end(), std::ostream_iterator<int>(std::cout, " "));

		fillWavefront<<<gridSize, blockSize>>>(
			samplesCount, d_wavefrontActiveIdx_ptr,
			d_wavefrontScanIdx_ptr, d_wavefrontNew_ptr);


		activeSize = d_wavefrontScanIdx[samplesCount-1]; 
		// increase by one if the last point in array is true because exclusive scan doesnt increment otherwise
		(d_wavefrontNew[d_wavefrontNew.size() - 1]) ?  ++activeSize : 0;
		if (activeSize == 0) // the next wavefront is empty (only valid for GMTwavefront)
			break;

		// std::cout << "exp DIM = " << DIM << " Itr " << itrs << " size " << activeSize << std::endl;

		const int blockSizeActiveExp = 128;
		const int gridSizeActiveExp = std::min((activeSize + blockSizeActiveExp - 1) / blockSizeActiveExp, 2147483647);
		findOptimalConnection<<<gridSizeActiveExp, blockSizeActiveExp>>>(
			d_wavefront_ptr, d_edges, d_costs, d_distancesCome, 
			d_nnComeEdges, nnSize, activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);

		const int blockSizeActiveVerify = 128;
		const int gridSizeActiveVerify = std::min((activeSize*numDisc + blockSizeActiveVerify - 1) / blockSizeActiveVerify, 2147483647);
		verifyExpansion<<<gridSizeActiveVerify, blockSizeActiveVerify>>>(
			obstaclesCount, d_wavefrontNew_ptr, d_edges,
			d_samples, d_obstacles, d_costs, 
			d_nnIdxs, d_discMotions, numDisc, d_isCollision,
			activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);


		removeInvalidExpansions<<<gridSizeActiveExp, blockSizeActiveExp>>>(
			obstaclesCount, d_wavefrontNew_ptr, d_edges,
			d_samples, d_obstacles, d_costs, 
			d_nnIdxs, d_discMotions, numDisc, d_isCollision,
			activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);

		// std::cout << "debug output: " << std::endl;
		// thrust::copy(d_debugOutput.begin(), d_debugOutput.begin()+100, std::ostream_iterator<float>(std::cout, " "));
		updateWavefront<<<gridSize, blockSize>>>(
			samplesCount, goal_idx,
			d_unvisited, d_wavefront_ptr, d_wavefrontNew_ptr, d_wavefrontWas,
			d_costs, d_costGoal, d_debugOutput_ptr);

		// copy over the goal cost (if non zero, then a solution was found)
		cudaMemcpy(&costGoal, d_costGoal, sizeof(float), cudaMemcpyDeviceToHost);
	}
	double t_gmt = (std::clock() - t_gmtStart) / (double) CLOCKS_PER_SEC;
	// std::cout << "time inside GMT is " << t_gmt << std::endl;
		
	// free arrays
	cudaFree(d_wavefrontWas);
	cudaFree(d_unvisited);
	cudaFree(d_costGoal);
	cudaFree(d_isCollision);
	cudaFree(d_initial);
}

// ***************** takes new initial and goal state not currently in the tree
int GMTinitGoal(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	float *d_distancesCome, int *d_nnGoEdges, int *d_nnComeEdges, int nnSize, float *d_discMotions, int *d_nnIdxs,
	float *d_samples, int samplesCount, bool *d_isFreeSamples, float r, int numDisc,
	float *d_costs, int *d_edges, int initial_idx)
{
	int goal_idx = -10; // does not implement this style goal checking

	cudaError_t code;
	double t_gmtStart = std::clock();
	int maxItrs = 100; 
	
	bool *d_wavefrontWas, *d_unvisited, *d_isCollision;
	float *d_costGoal;
	thrust::device_vector<float> d_debugOutput(samplesCount*numDisc);
	thrust::device_vector<bool> d_wavefront(samplesCount);
	thrust::device_vector<bool> d_wavefrontNew(samplesCount);
	thrust::device_vector<int> d_wavefrontScanIdx(samplesCount);
	thrust::device_vector<int> d_wavefrontActiveIdx(samplesCount);

	float *d_debugOutput_ptr = thrust::raw_pointer_cast(d_debugOutput.data());
	bool *d_wavefront_ptr = thrust::raw_pointer_cast(d_wavefront.data());
	bool *d_wavefrontNew_ptr = thrust::raw_pointer_cast(d_wavefrontNew.data());
	int *d_wavefrontScanIdx_ptr = thrust::raw_pointer_cast(d_wavefrontScanIdx.data());
	int *d_wavefrontActiveIdx_ptr = thrust::raw_pointer_cast(d_wavefrontActiveIdx.data());

	cudaMalloc(&d_wavefrontWas, sizeof(bool)*samplesCount);
	cudaMalloc(&d_unvisited, sizeof(bool)*samplesCount);
	cudaMalloc(&d_costGoal, sizeof(float));
	cudaMalloc(&d_isCollision, sizeof(bool)*numDisc*samplesCount);

	if (d_unvisited == NULL) {
		std::cout << "Allocation Failure" << std::endl;
		exit(1);
	}

	int *d_edges_backup;
	cudaMalloc(&d_edges_backup, sizeof(int)*samplesCount);
	cudaMemcpy(d_edges_backup, d_edges, sizeof(int)*samplesCount, cudaMemcpyDeviceToDevice);

	// setup array values
	const int blockSize = 128;
	const int gridSize = std::min((samplesCount + blockSize - 1) / blockSize, 2147483647);
	setupArrays<<<gridSize, blockSize>>>(
		d_wavefront_ptr, d_wavefrontNew_ptr, d_wavefrontWas, d_unvisited, d_isFreeSamples,
		d_costGoal, d_costs, d_edges, samplesCount);

	// attach initial condition (referred to as node number -2)
	// std::cout << "init" << initial[0] << ", " << initial[1] << ", " << initial[2] << ", " << initial[3] << std::endl;
	float *d_initial;
	cudaMalloc(&d_initial, sizeof(float)*DIM);
	cudaMemcpy(d_initial, initial, sizeof(float)*DIM, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	// std::cout << "attaching wavefront to "<< initial[0] << ", " << initial[1] << ", " << initial[2] << ", " << initial[3] << std::endl;
	attachWavefront<<<gridSize,blockSize>>>(
		samplesCount, d_samples, d_initial, r, goal_idx, d_wavefront_ptr, d_edges,
		obstaclesCount, d_obstacles, d_costs, d_costGoal, d_debugOutput_ptr);
	cudaDeviceSynchronize();
	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on attachWavefront: " << cudaGetErrorString(code) << std::endl; }
	// std::cout << "wavefront attached "<< initial[0] << ", " << initial[1] << ", " << initial[2] << ", " << initial[3] << std::endl;

	// setup goal condition
	bool *d_inGoal;
	cudaMalloc(&d_inGoal, sizeof(bool)*samplesCount);
	float *d_goal;
	cudaMalloc(&d_goal, sizeof(float)*DIM);
	cudaMemcpy(d_goal, goal, sizeof(float)*DIM, cudaMemcpyHostToDevice);

	float goalD2 = 0.02; // define a ball around the goal region
	// TODO: verify initial condition isn't already in the goal region
	// TODO: check if initial attaching finds a solution

	buildInGoal<<<gridSize,blockSize>>>(samplesCount, d_samples, d_goal, goalD2,
		d_inGoal, d_debugOutput_ptr);
	cudaDeviceSynchronize();
	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on buildInGoal: " << cudaGetErrorString(code) << std::endl; }
	
	bool inGoal[samplesCount];
	CUDA_ERROR_CHECK(cudaMemcpy(inGoal, d_inGoal, sizeof(bool)*samplesCount, cudaMemcpyDeviceToHost));
	bool wavefront[samplesCount];

	float costGoal = 0;
	int itrs = 0;
	int activeSize = 0;
	while (itrs < maxItrs && costGoal == 0)
	{ 
		++itrs;

		// prefix sum to fill wavefrontScanIdx to find points in the current band of the wavefront
		thrust::exclusive_scan(d_wavefront.begin(), d_wavefront.end(), d_wavefrontScanIdx.begin());
		fillWavefront<<<gridSize, blockSize>>>(samplesCount,
			d_wavefrontActiveIdx_ptr, d_wavefrontScanIdx_ptr, d_wavefront_ptr);
		// std::cout << "scanned" << std::endl;

		activeSize = d_wavefrontScanIdx[samplesCount-1];
		// increase by one if the last point in array is true because exclusive scan doesnt increment otherwise
		(d_wavefront[d_wavefront.size() - 1]) ? ++activeSize : 0;

		std::cout << "new DIM = " << DIM << " Itr " << itrs << " size " << activeSize << std::endl;

		if (activeSize == 0) {
			// failed to connect to the tree (often this is from the initial condition, just return the same tree that came in)
			cudaMemcpy(d_edges, d_edges_backup, sizeof(int)*samplesCount, cudaMemcpyDeviceToDevice);
			return -1;
		}

		const int blockSizeActive = 128;
		const int gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 2147483647);
		
		// std::cout << "find wavefront (" << blockSizeActive << ", " gridSizeActive << ")" << std::endl;
		findWavefront<<<gridSizeActive, blockSizeActive>>>(
			d_unvisited, d_wavefront_ptr, d_wavefrontNew_ptr, d_wavefrontWas, 
			d_nnGoEdges, nnSize, activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);


		// prefix sum to fill wavefrontScanIdx to connect to the next wavefront
		thrust::exclusive_scan(d_wavefrontNew.begin(), d_wavefrontNew.end(), d_wavefrontScanIdx.begin());
		
		// to print thrust vector) thrust::copy(d_wavefrontScanIdx.begin(), d_wavefrontScanIdx.end(), std::ostream_iterator<int>(std::cout, " "));

		fillWavefront<<<gridSize, blockSize>>>(
			samplesCount, d_wavefrontActiveIdx_ptr,
			d_wavefrontScanIdx_ptr, d_wavefrontNew_ptr);


		activeSize = d_wavefrontScanIdx[samplesCount-1]; 
		// increase by one if the last point in array is true because exclusive scan doesnt increment otherwise
		(d_wavefrontNew[d_wavefrontNew.size() - 1]) ?  ++activeSize : 0;
		if (activeSize == 0) // the next wavefront is empty (only valid for GMTwavefront)
			break;

		// std::cout << "exp DIM = " << DIM << " Itr " << itrs << " size " << activeSize << std::endl;

		const int blockSizeActiveExp = 128;
		const int gridSizeActiveExp = std::min((activeSize + blockSizeActiveExp - 1) / blockSizeActiveExp, 2147483647);
		findOptimalConnection<<<gridSizeActiveExp, blockSizeActiveExp>>>(
			d_wavefront_ptr, d_edges, d_costs, d_distancesCome, 
			d_nnComeEdges, nnSize, activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);

		const int blockSizeActiveVerify = 128;
		const int gridSizeActiveVerify = std::min((activeSize*numDisc + blockSizeActiveVerify - 1) / blockSizeActiveVerify, 2147483647);
		verifyExpansion<<<gridSizeActiveVerify, blockSizeActiveVerify>>>(
			obstaclesCount, d_wavefrontNew_ptr, d_edges,
			d_samples, d_obstacles, d_costs, 
			d_nnIdxs, d_discMotions, numDisc, d_isCollision,
			activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);


		removeInvalidExpansions<<<gridSizeActiveExp, blockSizeActiveExp>>>(
			obstaclesCount, d_wavefrontNew_ptr, d_edges,
			d_samples, d_obstacles, d_costs, 
			d_nnIdxs, d_discMotions, numDisc, d_isCollision,
			activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);

		// std::cout << "debug output: " << std::endl;
		// thrust::copy(d_debugOutput.begin(), d_debugOutput.begin()+100, std::ostream_iterator<float>(std::cout, " "));
		updateWavefront<<<gridSize, blockSize>>>(
			samplesCount, goal_idx,
			d_unvisited, d_wavefront_ptr, d_wavefrontNew_ptr, d_wavefrontWas,
			d_costs, d_costGoal, d_debugOutput_ptr);

		// check if any nodes are in the goal region
		cudaMemcpy(wavefront, d_wavefront_ptr, sizeof(bool)*samplesCount, cudaMemcpyDeviceToHost);
		float costs[samplesCount];
		cudaMemcpy(costs, d_costs, sizeof(float)*samplesCount, cudaMemcpyDeviceToHost);
		for (int node = 0; node < samplesCount; ++node) {
			if (wavefront[node] && inGoal[node]) {
				if (costGoal == 0) {
					costGoal = costs[node];
					goal_idx = node;
				} else if (costGoal > costs[node]) {
					costGoal = costs[node];
					goal_idx = node;
				} 
				std::cout << "solved! with cost " << costGoal << " at node " << node << std::endl;
			}
		}
	}
	double t_gmt = (std::clock() - t_gmtStart) / (double) CLOCKS_PER_SEC;
	// std::cout << "time inside GMT is " << t_gmt << std::endl;
		
	// free arrays
	cudaFree(d_wavefrontWas);
	cudaFree(d_unvisited);
	cudaFree(d_costGoal);
	cudaFree(d_isCollision);
	cudaFree(d_initial);
	return goal_idx;
}

// ***************** normal GMT with lambda = 1
void GMTwavefront(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	float *d_distancesCome, int *d_nnGoEdges, int *d_nnComeEdges, int nnSize, float *d_discMotions, int *d_nnIdxs,
	float *d_samples, int samplesCount, bool *d_isFreeSamples, float r, int numDisc,
	float *d_costs, int *d_edges, int initial_idx, int goal_idx)
{
	double t_gmtStart = std::clock();
	int maxItrs = 100; 
	
	bool *d_wavefrontWas, *d_unvisited, *d_isCollision;
	float *d_costGoal;
	thrust::device_vector<float> d_debugOutput(samplesCount*numDisc);
	thrust::device_vector<bool> d_wavefront(samplesCount);
	thrust::device_vector<bool> d_wavefrontNew(samplesCount);
	thrust::device_vector<int> d_wavefrontScanIdx(samplesCount);
	thrust::device_vector<int> d_wavefrontActiveIdx(samplesCount);

	float *d_debugOutput_ptr = thrust::raw_pointer_cast(d_debugOutput.data());
	bool *d_wavefront_ptr = thrust::raw_pointer_cast(d_wavefront.data());
	bool *d_wavefrontNew_ptr = thrust::raw_pointer_cast(d_wavefrontNew.data());
	int *d_wavefrontScanIdx_ptr = thrust::raw_pointer_cast(d_wavefrontScanIdx.data());
	int *d_wavefrontActiveIdx_ptr = thrust::raw_pointer_cast(d_wavefrontActiveIdx.data());

	cudaMalloc(&d_wavefrontWas, sizeof(bool)*samplesCount);
	cudaMalloc(&d_unvisited, sizeof(bool)*samplesCount);
	cudaMalloc(&d_costGoal, sizeof(float));
	cudaMalloc(&d_isCollision, sizeof(bool)*numDisc*samplesCount);

	if (d_unvisited == NULL) {
		std::cout << "Allocation Failure" << std::endl;
		exit(1);
	}

	// setup array values
	const int blockSize = 128;
	const int gridSize = std::min((samplesCount + blockSize - 1) / blockSize, 2147483647);
	setupArrays<<<gridSize, blockSize>>>(d_wavefront_ptr, d_wavefrontNew_ptr, 
		d_wavefrontWas, d_unvisited, d_isFreeSamples, d_costGoal, d_costs, d_edges, samplesCount);
	
	float costGoal = 0;
	int itrs = 0;
	int activeSize = 0;
	while (itrs < maxItrs && costGoal == 0)
	{ 
		++itrs;

		// prefix sum to fill wavefrontScanIdx to find points in the current band of the wavefront
		thrust::exclusive_scan(d_wavefront.begin(), d_wavefront.end(), d_wavefrontScanIdx.begin());
		fillWavefront<<<gridSize, blockSize>>>(samplesCount,
			d_wavefrontActiveIdx_ptr, d_wavefrontScanIdx_ptr, d_wavefront_ptr);

		activeSize = d_wavefrontScanIdx[samplesCount-1];
		// increase by one if the last point in array is true because exclusive scan doesnt increment otherwise
		(d_wavefront[d_wavefront.size() - 1]) ? ++activeSize : 0;

		std::cout << "new DIM = " << DIM << " Itr " << itrs << " size " << activeSize << std::endl;

		const int blockSizeActive = 128;
		const int gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 2147483647);
		
		findWavefront<<<gridSizeActive, blockSizeActive>>>(
			d_unvisited, d_wavefront_ptr, d_wavefrontNew_ptr, d_wavefrontWas, 
			d_nnGoEdges, nnSize, activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);

		// prefix sum to fill wavefrontScanIdx to connect to the next wavefront
		thrust::exclusive_scan(d_wavefrontNew.begin(), d_wavefrontNew.end(), d_wavefrontScanIdx.begin());
		
		// to print thrust vector) thrust::copy(d_wavefrontScanIdx.begin(), d_wavefrontScanIdx.end(), std::ostream_iterator<int>(std::cout, " "));

		fillWavefront<<<gridSize, blockSize>>>(
			samplesCount, d_wavefrontActiveIdx_ptr,
			d_wavefrontScanIdx_ptr, d_wavefrontNew_ptr);

		activeSize = d_wavefrontScanIdx[samplesCount-1]; 
		// increase by one if the last point in array is true because exclusive scan doesnt increment otherwise
		(d_wavefrontNew[d_wavefrontNew.size() - 1]) ?  ++activeSize : 0;
		if (activeSize == 0) // the next wavefront is empty (only valid for GMTwavefront)
			break;

		std::cout << "exp DIM = " << DIM << " Itr " << itrs << " size " << activeSize << std::endl;

		const int blockSizeActiveExp = 128;
		const int gridSizeActiveExp = std::min((activeSize + blockSizeActiveExp - 1) / blockSizeActiveExp, 2147483647);
		findOptimalConnection<<<gridSizeActiveExp, blockSizeActiveExp>>>(
			d_wavefront_ptr, d_edges, d_costs, d_distancesCome, 
			d_nnComeEdges, nnSize, activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);
		const int blockSizeActiveVerify = 128;
		const int gridSizeActiveVerify = std::min((activeSize*numDisc + blockSizeActiveVerify - 1) / blockSizeActiveVerify, 2147483647);
		verifyExpansion<<<gridSizeActiveVerify, blockSizeActiveVerify>>>(
			obstaclesCount, d_wavefrontNew_ptr, d_edges,
			d_samples, d_obstacles, d_costs, 
			d_nnIdxs, d_discMotions, numDisc, d_isCollision,
			activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);
		removeInvalidExpansions<<<gridSizeActiveExp, blockSizeActiveExp>>>(
			obstaclesCount, d_wavefrontNew_ptr, d_edges,
			d_samples, d_obstacles, d_costs, 
			d_nnIdxs, d_discMotions, numDisc, d_isCollision,
			activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);
		// std::cout << "debug output: " << std::endl;
		// thrust::copy(d_debugOutput.begin(), d_debugOutput.begin()+100, std::ostream_iterator<float>(std::cout, " "));
		updateWavefront<<<gridSize, blockSize>>>(
			samplesCount, goal_idx,
			d_unvisited, d_wavefront_ptr, d_wavefrontNew_ptr, d_wavefrontWas,
			d_costs, d_costGoal, d_debugOutput_ptr);

		// copy over the goal cost (if non zero, then a solution was found)
		cudaMemcpy(&costGoal, d_costGoal, sizeof(float), cudaMemcpyDeviceToHost);
	}
	double t_gmt = (std::clock() - t_gmtStart) / (double) CLOCKS_PER_SEC;
	std::cout << "time inside GMT is " << t_gmt << std::endl;
		
	// free arrays
	cudaFree(d_wavefrontWas);
	cudaFree(d_unvisited);
	cudaFree(d_costGoal);
	cudaFree(d_isCollision);
}

__global__ void setupArrays(bool* wavefront, bool* wavefrontNew, bool* wavefrontWas, bool* unvisited, 
	bool *isFreeSamples, float *costGoal, float* costs, int* edges, int samplesCount)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	if (node >= samplesCount) 
		return;

	unvisited[node] = isFreeSamples[node];
	wavefrontNew[node] = false;
	wavefrontWas[node] = !isFreeSamples[node];
	if (node == 0) {		
		wavefront[node] = true;
		*costGoal = 0;
	} else {
		wavefront[node] = false;
	}
	costs[node] = 0;
	edges[node] = -1;

	if (!isFreeSamples[node]) {
		costs[node] = -11;
		edges[node] = -2;
	}
}

__global__
void findWavefront(bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas,
	int *nnGoEdges, int nnSize, int wavefrontSize, int* wavefrontActiveIdx, float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;

	int node = wavefrontActiveIdx[tid];

	if (wavefront[node])
	{
		wavefrontWas[node] = true;
		for (int i = 0; i < nnSize; ++i) {
			int nnIdx = nnGoEdges[node*nnSize + i];
			if (nnIdx == -1)
				return;
			if (unvisited[nnIdx] && !wavefront[nnIdx]) {
				wavefrontNew[nnIdx] = true;
			}
		}
	}
}

__global__
void fillWavefront(int samplesCount, int *wavefrontActiveIdx, int *wavefrontScanIdx, bool* wavefront)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	if (node >= samplesCount)
		return;
	if (!wavefront[node])
		return;

	wavefrontActiveIdx[wavefrontScanIdx[node]] = node;
}

__global__
void findOptimalConnection(bool *wavefront, int* edges, float* costs, float *distancesCome, 
	int *nnComeEdges, int nnSize, int wavefrontSize, int* wavefrontActiveIdx, float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;

	int node = wavefrontActiveIdx[tid];

	for (int i = 0; i < nnSize; ++i) {
		int nnIdx = nnComeEdges[node*nnSize + i];
		if (nnIdx == -1)
			break;
		if (wavefront[nnIdx]) {
			float costTmp = costs[nnIdx] + distancesCome[node*nnSize + i];
			if (costs[node] == 0 || costTmp < costs[node]) {
				costs[node] = costTmp;
				edges[node] = nnIdx;
			}
		}
	}
} 

__global__
void verifyExpansion(int obstaclesCount, bool *wavefrontNew, int* edges,
	float *samples, float* obstacles, float* costs,
	int *nnIdxs, float *discMotions, int numDisc, bool *isCollision,
	int wavefrontSize, int* wavefrontActiveIdx, float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize*numDisc)
		return;
	int nodeIdx = tid/numDisc;
	int discIdx = tid%numDisc;
	int node = wavefrontActiveIdx[nodeIdx];
	
	waypointCollisionCheck(node, edges[node], obstaclesCount, obstacles, 
			nnIdxs, discMotions, discIdx, numDisc, isCollision, tid, debugOutput);
}

__global__
void removeInvalidExpansions(int obstaclesCount, bool *wavefrontNew, int* edges,
	float *samples, float* obstacles, float* costs,
	int *nnIdxs, float *discMotions, int numDisc, bool *isCollision,
	int wavefrontSize, int* wavefrontActiveIdx, float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;
	int node = wavefrontActiveIdx[tid];

	bool notValid = isCollision[tid*numDisc];
	for (int i = 1; i < numDisc; ++i) 
		notValid = notValid || isCollision[tid*numDisc + i];
	if (notValid) {
		costs[node] = 0;
		edges[node] = -1;
		wavefrontNew[node] = false;
	}
}

__global__
void updateWavefront(int samplesCount, int goal_idx,
	bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas,
	float* costs, float* costGoal, float* debugOutput)
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

__global__
void attachWavefront(int samplesCount, float *samples, float *initial, 
	float r, int goal_idx, bool *wavefront, int *edges,
	int obstaclesCount, float *obstacles, 
	float* costs, float* costGoal, float* debugOutput)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount)
		return;
	if (node == 0) // undo setupArray init
		wavefront[node] = false;

	float sample[DIM];
	for (int d = 0; d < DIM; ++d) 
		sample[d] = samples[node*DIM+d];

	// check cost is below r
	float topt = toptBisection(initial, sample, 20);
	float copt = cost(topt, initial, sample);
	if (copt > r*2)
		return;

	// if it is, check path is collision free
	float splitPath[DIM*(numDiscInit+1)];
	findDiscretizedPath(splitPath, initial, sample, numDiscInit);

	float v[DIM], w[DIM];
	float bbMin[DIM], bbMax[DIM];
	bool motionValid = true;
	for (int i = 0; i < numDiscInit; ++i) {
		if (!motionValid)
			break;
		for (int d = 0; d < DIM; ++d) {
			v[d] = splitPath[i*DIM + d];
			w[d] = splitPath[i*DIM + d + DIM];

			if (v[d] > w[d]) {
				bbMin[d] = w[d];
				bbMax[d] = v[d];
			} else {
				bbMin[d] = v[d];
				bbMax[d] = w[d];
			}
		}
		motionValid = motionValid && isMotionValid(v, w, bbMin, bbMax, obstaclesCount, obstacles, debugOutput);
	}

	if (!motionValid)
		return;

	// add the cost to costs, check if we have a goal connection, and add node to wavefront, update edges
	costs[node] = copt;
	wavefront[node] = true;
	edges[node] = -1; // -2 marks our starting point

	if (node == goal_idx)
		costGoal[0] = copt;

}

__global__ 
void buildInGoal(int samplesCount,float *samples, float *goal, float goalD2,
	bool *inGoal, float *debugOutput)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount)
		return;

	float dist2 = 0;
	for (int d = 0; d < DIM/2; ++d)
		dist2 += (samples[node*DIM+d] - goal[d])*(samples[node*DIM+d] - goal[d]);

	if (dist2 < goalD2)
		inGoal[node] = true;
	else 
		inGoal[node] = false;
}