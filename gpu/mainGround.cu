/*
mainGround.cu
Author: Brian Ichter

This runs the GMT* algorithm (and FMT, PRM) for a double integrator system (representing a quadrotor model). This main file 
is used primarily for timing results and evaluations of solution quality.

Run instructions:
	TODO 

*/

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <algorithm> 
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <limits>
#include <fstream>
#include <sstream>

#include "motionPlanningProblem.cuh"
#include "motionPlanningSolution.cuh"
#include "obstacles.cuh"
#include "helper.cuh"
#include "sampler.cuh"
#include "2pBVP.cuh"
#include "GMT.cuh"
#include "PRM.cuh"
#include "FMT.cuh"

// compiler inputs
#ifndef DIM
#error Please define DIM.
#endif
#ifndef NUM
#error Please define NUM.
#endif

// ***************** offline settings (paste setup here)
float lo[DIM] = {0, 0, 0, -1, -1, -1};
float hi[DIM] = {1, 1, 1, 1, 1, 1};
int edgeDiscNum = 12;
float dt = 0.05; // time step for dynamic propagation
int numDisc = 4; // number of discretizations of kinodynamic paths
float ms = 1000;
bool verbose = true;

int main(int argc, const char* argv[]) {
	std::cout << "*********** Beginning Ground Run (DIM = " << DIM << ", NUM = " << NUM << ") **********" << std::endl;

	// check setup is 3D DI
	if (DIM != 6) {
		std::cout << "DIM must be 6, currently only implemented for the 3D double integrator" << std::endl;
		return -1;
	}

	// check a file has been specific
	if (argc != 2) {
		std::cout << "Must specify an online setting filename, i.e. $ ./ground file.txt" << std::endl;
		return -1;
	}

	// set cuda device
	int count = 0;
	cudaGetDeviceCount(&count);
	cudaError_t code;
	int deviceNum = count-1;
	cudaSetDevice(deviceNum);
	std::cout << "Number of CUDA devices = " << count << ", selecting " << deviceNum << std::endl;
	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on selecting device: " << cudaGetErrorString(code) << std::endl; }
	
	// create MotionPlanningProblem, which is currently not used except for print statements
	MotionPlanningProblem mpp;
	mpp.filename = argv[1];
	mpp.dimC = DIM;
	mpp.dimW = DIM/2;
	mpp.numSamples = NUM;
	mpp.edgeDiscNum = edgeDiscNum;
	mpp.dt = dt;
	mpp.hi.resize(mpp.dimC);
	mpp.lo.resize(mpp.dimC);

	for (int i = 0; i < DIM; ++i) {
		mpp.hi[i] = hi[i];
		mpp.lo[i] = lo[i];
	}

	// init and goal states (must then connect to the final tree)
	std::vector<float> init(DIM, 0.1);
	std::vector<float> goal(DIM, 0.9);
	// goal[0] = 0.8; goal[1] = 0.8214;
	// init[1] = 0.05; 
	for (int d = DIM/2; d < DIM; ++d) {
	 	init[d] = 0;
	 	goal[d] = 0;
	}
	int goalIdx = NUM-1;
	int initIdx = 0;

	// obstacles
	int numObstacles = getObstaclesCount();
	std::vector<float> obstacles(numObstacles*2*DIM);
	generateObstacles(obstacles.data(), numObstacles*2*DIM);

	std::ifstream file (mpp.filename);
	std::string readValue;
	if (file.good()) {
		// read in init
		for (int d = 0; d < DIM; ++d) {
			getline(file, readValue, ',');	
	        std::stringstream convertorInit(readValue);
	        convertorInit >> init[d];
		}

		for (int d = 0; d < DIM; ++d) {
			getline(file, readValue, ',');	
	        std::stringstream convertorGoal(readValue);
	        convertorGoal >> goal[d];
		}

		for (int d = 0; d < DIM; ++d) {
			getline(file, readValue, ',');	
	        std::stringstream convertorLo(readValue);
	        convertorLo >> lo[d];
		}

		for (int d = 0; d < DIM; ++d) {
			getline(file, readValue, ',');	
	        std::stringstream convertorHi(readValue);
	        convertorHi >> hi[d];
		}

		getline(file, readValue, ',');
		std::stringstream convertorNumObs(readValue);	
		convertorNumObs >> numObstacles;

		obstacles.resize(numObstacles*DIM*2);
		for (int obs = 0; obs < numObstacles; ++obs) {
			for (int d = 0; d < DIM*2; ++d) {
				getline(file, readValue, ',');	
				std::stringstream convertorObs(readValue);
				convertorObs >> obstacles[obs*DIM*2 + d];
			}
		}

		if (verbose) {
			std::cout << "***** Inputs from " << mpp.filename << " are: *****" << std::endl;
		}
		if (verbose) {
			std::cout << "init is: "; printArray(&init[0],1,DIM,std::cout);
		}
		if (verbose) {
			std::cout << "goal is: "; printArray(&goal[0],1,DIM,std::cout);
		}
		if (verbose) {
			std::cout << "lo is: "; printArray(&lo[0],1,DIM,std::cout);
		}
		if (verbose) {
			std::cout << "hi is: "; printArray(&hi[0],1,DIM,std::cout);
		}
		if (verbose) {
			std::cout << "obstacle count = " << numObstacles << std::endl;
		}
		if(verbose) {
			printArray(&obstacles[0],numObstacles,2*DIM,std::cout);
		}

	}

	std::cout << "--- Motion planning problem, " << mpp.filename << " ---" << std::endl;
	std::cout << "Sample count = " << mpp.numSamples << ", C-space dim = " << mpp.dimC << ", Workspace dim = " << mpp.dimW << std::endl;
	std::cout << "hi = ["; for (int i = 0; i < mpp.dimC; ++i) { std::cout << hi[i] << " "; } std::cout << "], ";
	std::cout << "lo = ["; for (int i = 0; i < mpp.dimC; ++i) { std::cout << lo[i] << " "; } std::cout << "]" << std::endl;
	std::cout << "edge discretizations " << mpp.edgeDiscNum << ", dt = " << mpp.dt << std::endl;
	std::cout << "Obstacle set, count = " << numObstacles << ":" << std::endl;
	printArray(obstacles.data(), numObstacles, 2*DIM, std::cout);

	// read in 
	// hi, lo
	// goal, init
	// obstacles, obstacles count

	/*********************** create array to return debugging information ***********************/
	float *d_debugOutput;
	cudaMalloc(&d_debugOutput, sizeof(float)*NUM);

	// ***************** precomputation

	// Sampling- to allow more parallelism we sample NUM points on X here (rather than Xfree)
	// and perform sample free online once obstacles are known.
	std::vector<float> samplesAll (DIM*NUM);
	createSamplesHalton(0, samplesAll.data(), &(init[0]), &(goal[0]), lo, hi);
	thrust::device_vector<float> d_samples_thrust(DIM*NUM);
	float *d_samples = thrust::raw_pointer_cast(d_samples_thrust.data());
	CUDA_ERROR_CHECK(cudaMemcpy(d_samples, samplesAll.data(), sizeof(float)*DIM*NUM, cudaMemcpyHostToDevice));

	// calculate nn
	double t_calc10thStart = std::clock();
	std::vector<float> topts ((NUM-1)*(NUM));
	std::vector<float> copts ((NUM-1)*(NUM));
	float rnPerc = 0.05;
	int rnIdx = (int) ((NUM-1)*(NUM)*rnPerc); // some percentile of the number of edges
	int numEdges = rnIdx;
	std::cout << "rnIdx is " << rnIdx << std::endl;
	int idx;
	float tmax = 5;

	// GPU version of finding rn and costs/times
	thrust::device_vector<float> d_copts_thrust((NUM-1)*(NUM));
	float* d_copts = thrust::raw_pointer_cast(d_copts_thrust.data());
	thrust::device_vector<float> d_topts_thrust((NUM-1)*(NUM));
	float* d_topts = thrust::raw_pointer_cast(d_topts_thrust.data());
	const int blockSizeFillCoptsTopts = 192;
	const int gridSizeFillCoptsTopts = std::min((NUM*NUM + blockSizeFillCoptsTopts - 1) / blockSizeFillCoptsTopts, 2147483647);
	if (gridSizeFillCoptsTopts == 2147483647)
		std::cout << "...... ERROR: increase grid size for fillCoptsTopts" << std::endl;
	fillCoptsTopts<<<gridSizeFillCoptsTopts, blockSizeFillCoptsTopts>>>(
		d_samples, d_copts, d_topts, tmax);
	cudaDeviceSynchronize();

	CUDA_ERROR_CHECK(cudaMemcpy(topts.data(), d_topts, sizeof(float)*((NUM-1)*(NUM)), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(copts.data(), d_copts, sizeof(float)*((NUM-1)*(NUM)), cudaMemcpyDeviceToHost));
	thrust::sort(d_copts_thrust.begin(), d_copts_thrust.end());
	float rn = d_copts_thrust[rnIdx];

	double t_2pbvpTestStart = std::clock();
	float x0[DIM], x1[DIM];

	double t_discMotionsStart = std::clock();
	std::vector<int> nnIdxs(NUM*NUM,-3);
	int nnGoSizes[NUM];
	int nnComeSizes[NUM];
	for (int i = 0; i < NUM; ++i) {
		nnGoSizes[i] = 0;
		nnComeSizes[i] = 0;
	}
	std::vector<float> discMotions (numEdges*(numDisc+1)*DIM,0); // array of motions, but its a vector who idk for some reason it won't work as an array
	int nnIdx = 0; // index position in NN discretization array
	idx = 0; // index position in copts vector above

	std::vector<float> coptsEdge (numEdges); // edge index accessed copts
	std::vector<float> toptsEdge (numEdges); // edge index accessed topts
	std::vector<float> c2g (NUM);
	for (int i = 0; i < NUM; ++i) {
		for (int d = 0; d < DIM; ++d)
			x0[d] = samplesAll[d + DIM*i];
		for (int j = 0; j < NUM; ++j) {
			if (j == i)
				continue;
			if (copts[idx] < rn) {
				coptsEdge[nnIdx] = copts[idx];
				toptsEdge[nnIdx] = topts[idx];
				for (int d = 0; d < DIM; ++d)
					x1[d] = samplesAll[d + DIM*j];
				nnIdxs[j*NUM+i] = nnIdx; // look up for discrete motions from i -> j
				findDiscretizedPath(&(discMotions[nnIdx*DIM*(numDisc+1)]), x0, x1, numDisc); // TODO: give topt
				nnGoSizes[i]++;
				nnComeSizes[j]++;
				nnIdx++;
			}
			idx++;
		}
		float topt_c2g = toptBisection(&(samplesAll[i*DIM]), &(samplesAll[goalIdx*DIM]), tmax);
		c2g[i] = cost(topt_c2g, &(samplesAll[i*DIM]), &(samplesAll[goalIdx*DIM]));
	}
	double t_discMotions = (std::clock() - t_discMotionsStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Discretizing motions took: " << t_discMotions*ms << " ms for " << nnIdx << " solves" << std::endl;

	float *d_toptsEdge;
	CUDA_ERROR_CHECK(cudaMalloc(&d_toptsEdge, sizeof(float)*numEdges));
	CUDA_ERROR_CHECK(cudaMemcpy(d_toptsEdge, toptsEdge.data(), sizeof(float)*numEdges, cudaMemcpyHostToDevice));
	float *d_coptsEdge;
	CUDA_ERROR_CHECK(cudaMalloc(&d_coptsEdge, sizeof(float)*numEdges));
	CUDA_ERROR_CHECK(cudaMemcpy(d_coptsEdge, coptsEdge.data(), sizeof(float)*numEdges, cudaMemcpyHostToDevice));

	int maxNNSize = 0;
	for (int i = 0; i < NUM; ++i) {
		if (maxNNSize < nnGoSizes[i])
			maxNNSize = nnGoSizes[i];
		if (maxNNSize < nnComeSizes[i])
			maxNNSize = nnComeSizes[i];
	}
	std::cout << "max number of nn is " << maxNNSize << std::endl;

	std::vector<float> distancesCome (NUM*maxNNSize, 0);
	std::vector<int> nnGoEdges (NUM*maxNNSize, -1); // edge gives indices (i,j) to check nnIdx to then find the discretized path
	std::vector<int> nnComeEdges (NUM*maxNNSize, -1); // edge gives indices (j,i) to check nnIdx to then find the discretized path
	std::vector<float> adjCosts (NUM*NUM,10000);
	std::vector<float> adjTimes (NUM*NUM,10000);
	idx = 0;
	for (int i = 0; i < NUM; ++i) {
		nnGoSizes[i] = 0; // clear nnSizes again
		nnComeSizes[i] = 0; // clear nnSizes again
	}
	for (int i = 0; i < NUM; ++i) {
		for (int j = 0; j < NUM; ++j) {
			if (j == i)
				continue;
			if (copts[idx] < rn) {
				nnGoEdges[i*maxNNSize + nnGoSizes[i]] = j; // edge from i to j (i -> j)
				nnComeEdges[j*maxNNSize + nnComeSizes[j]] = i;
				distancesCome[j*maxNNSize + nnComeSizes[j]] = copts[idx];
				nnGoSizes[i]++;
				nnComeSizes[j]++;
				adjCosts[i*NUM + j] = copts[idx]; // cost to go from i to j
				adjTimes[i*NUM + j] = topts[idx]; // time to go from i to j
			}
			idx++;
		}
	}	

	// put NN onto device
	float *d_discMotions;
	CUDA_ERROR_CHECK(cudaMalloc(&d_discMotions, sizeof(float)*numEdges*(numDisc+1)*DIM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_discMotions, &discMotions[0], sizeof(float)*numEdges*(numDisc+1)*DIM, cudaMemcpyHostToDevice));
	// std::cout << "**** disc motions = " << std::endl;
	// printArray(&discMotions[0], 30, DIM, std::cout);

	int *d_nnIdxs;
	CUDA_ERROR_CHECK(cudaMalloc(&d_nnIdxs, sizeof(int)*NUM*NUM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_nnIdxs, &(nnIdxs[0]), sizeof(int)*NUM*NUM, cudaMemcpyHostToDevice));

	float *d_distancesCome;
	CUDA_ERROR_CHECK(cudaMalloc(&d_distancesCome, sizeof(float)*NUM*maxNNSize));
	CUDA_ERROR_CHECK(cudaMemcpy(d_distancesCome, &(distancesCome[0]), sizeof(float)*NUM*maxNNSize, cudaMemcpyHostToDevice));

	int *d_nnGoEdges;
	CUDA_ERROR_CHECK(cudaMalloc(&d_nnGoEdges, sizeof(int)*NUM*maxNNSize));
	CUDA_ERROR_CHECK(cudaMemcpy(d_nnGoEdges, &(nnGoEdges[0]), sizeof(int)*NUM*maxNNSize, cudaMemcpyHostToDevice));

	int *d_nnComeEdges;
	CUDA_ERROR_CHECK(cudaMalloc(&d_nnComeEdges, sizeof(int)*NUM*maxNNSize));
	CUDA_ERROR_CHECK(cudaMemcpy(d_nnComeEdges, &(nnComeEdges[0]), sizeof(int)*NUM*maxNNSize, cudaMemcpyHostToDevice));

	float *d_costs;
	cudaMalloc(&d_costs, sizeof(float)*NUM);
	thrust::device_vector<int> d_edges(NUM);
	int* d_edges_ptr = thrust::raw_pointer_cast(d_edges.data());


	// ******** online computation

	// load obstacles on device
	float *d_obstacles;
	CUDA_ERROR_CHECK(cudaMalloc(&d_obstacles, sizeof(float)*2*numObstacles*DIM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_obstacles, obstacles.data(), sizeof(float)*2*numObstacles*DIM, cudaMemcpyHostToDevice));

	// sample free	
	bool isFreeSamples[NUM];
	thrust::device_vector<bool> d_isFreeSamples_thrust(NUM);
	bool* d_isFreeSamples = thrust::raw_pointer_cast(d_isFreeSamples_thrust.data());

	double t_sampleFreeStart = std::clock();
	const int blockSizeSF = 192;
	const int gridSizeSF = std::min((NUM + blockSizeSF - 1) / blockSizeSF, 2147483647);
	if (gridSizeSF == 2147483647)
		std::cout << "...... ERROR: increase grid size for sampleFree" << std::endl;
	sampleFree<<<gridSizeSF, blockSizeSF>>>(
		d_obstacles, numObstacles, d_samples, d_isFreeSamples, d_debugOutput);
	cudaDeviceSynchronize();
	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on freeEdges: " << cudaGetErrorString(code) << std::endl; }

	double t_sampleFree = (std::clock() - t_sampleFreeStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Sample free took: " << t_sampleFree << " s" << std::endl;
	CUDA_ERROR_CHECK(cudaMemcpy(isFreeSamples, d_isFreeSamples, sizeof(bool)*NUM, cudaMemcpyDeviceToHost));

	// run GMT
	double t_gmtStart = std::clock();
	std::cout << "Running wavefront expansion GMT" << std::endl;
	GMTwavefront(&(init[0]), &(goal[0]), d_obstacles, numObstacles,
		d_distancesCome, d_nnGoEdges, d_nnComeEdges, maxNNSize, d_discMotions, d_nnIdxs,
		d_samples, NUM, d_isFreeSamples, rn, numDisc,
		d_costs, d_edges_ptr, initIdx, goalIdx) ;

	double t_gmt = (std::clock() - t_gmtStart) / (double) CLOCKS_PER_SEC;
	std::cout << "******** GMT took: " << t_gmt << " s" << std::endl;
	float costGoal = 0;
	cudaMemcpy(&costGoal, d_costs+goalIdx, sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "Solution cost: " << costGoal << std::endl;
	
	// ***************** output results
	std::ofstream matlabData;
	matlabData.open ("matlabInflationData.txt");
	matlabData << "obstacles.data() = ["; 
	printArray(obstacles.data(), 2*numObstacles, DIM, matlabData); 
	matlabData << "];" << std::endl;

	true && printSolution(NUM, d_samples, d_edges_ptr, d_costs);
	matlabData.close();

	// ***************** PRM
	std::vector<float> costs(NUM,10000);
	
	double t_PRMStart = std::clock();
	std::cout << "Running PRM" << std::endl;
	PRM(&(init[0]), &(goal[0]), d_obstacles, numObstacles,
		adjCosts, nnGoEdges, nnComeEdges, maxNNSize, d_discMotions, nnIdxs,
		d_samples, NUM, d_isFreeSamples, rn, numDisc, numEdges,
		costs, d_edges_ptr, initIdx, goalIdx,
		c2g);
	double t_PRM = (std::clock() - t_PRMStart) / (double) CLOCKS_PER_SEC;
	std::cout << "******** PRM took: " << t_PRM << " s" << std::endl;

	// ***************** FMT
	double t_FMTStart = std::clock();
	std::cout << "Running FMT" << std::endl;
	// call FMT
	FMT(&(init[0]), &(goal[0]), obstacles.data(), numObstacles,
		adjCosts, nnGoEdges, nnComeEdges, maxNNSize, discMotions, nnIdxs,
		NUM, d_isFreeSamples, rn, numDisc, numEdges,
		costs, initIdx, goalIdx,
		samplesAll, adjTimes);
	
	double t_FMT = (std::clock() - t_FMTStart) / (double) CLOCKS_PER_SEC;
	std::cout << "******** FMT took: " << t_FMT << " s" << std::endl;

	// ***************** free memory
	cudaFree(d_obstacles);
	cudaFree(d_toptsEdge);
	cudaFree(d_coptsEdge);
	cudaFree(d_discMotions);
	cudaFree(d_nnIdxs);
	cudaFree(d_distancesCome);
	cudaFree(d_nnGoEdges);
	cudaFree(d_nnComeEdges);
	cudaFree(d_costs);
}
