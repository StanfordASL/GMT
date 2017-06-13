/*
mainGround.cu
Author: Brian Ichter

This runs the GMT* and MCMP algorithms for a double integrator system (representing a quadrotor model). This main file 
is used primarily for timing results and evaluations of solution quality, rather than to be run directly with the quad
(we use mainQuad.cu for this).

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
#include "dubinsAirplane.cuh"

// compiler inputs
#ifndef DIM
#error Please define DIM.
#endif
#ifndef NUM
#error Please define NUM.
#endif

// horrible coding, but will get the job done for now
int DIMdubMain = 4;

// ***************** offline settings (paste setup here)
float lo[DIM] = {0, 0, 0, 0};
float hi[DIM] = {5, 5, 5, 2*M_PI};
int edgeDiscNum = 8;
float dt = 0.05; // time step for dynamic propagation
int numDisc = 4; // number of discretizations of kinodynamic paths
int numControls = 3; // number of total controls (i.e., Dubins word length)
float ms = 1000;
bool verbose = true;

int main(int argc, const char* argv[]) {
	float x0dub[4] = {0, 0, 0, M_PI/2};
	float x1dub[4] = {1, -3, 1, 0};
	std::vector<int> control(numControls);
	std::vector<float> controlD(numControls);
	std::vector<float> path(DIMdubMain*numDisc*numControls);

	float cmin = dubinsAirplaneCost(x0dub, x1dub, control.data(), controlD.data());
	std::cout << "cost = " << cmin << ", word is " << control[0] << ", " << control[1] << ", " << control[2] << std::endl;
	std::cout << "durations (" << controlD[0] << ", " << controlD[1] << ", " << controlD[2] << ")" << std::endl;

	dubinsAirplanePath(x0dub, x1dub, control.data(), controlD.data(), path.data(), numDisc);
	printArray(path.data(), numControls*numDisc, DIMdubMain, std::cout);




	std::cout << "*********** Beginning Dubins Aircraft Run (DIM = " << DIM << ", NUM = " << NUM << ") **********" << std::endl;

	// check setup is 3D DI
	if (DIM != 4) {
		std::cout << "DIM must be 4 for Dubins airplane (x y z theta)" << std::endl;
		return -1;
	}

	// check a file has been specific
	if (argc != 2) {
		std::cout << "Must specify an problem setup filename, i.e. $ ./dubins file.txt" << std::endl;
		return -1;
	}

	int count = 0;
	cudaGetDeviceCount(&count);
	cudaError_t code;
	int deviceNum = 1;
	cudaSetDevice(deviceNum);
	std::cout << "Number of CUDA devices = " << count << ", selecting " << deviceNum << std::endl;
	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on selecting device: " << cudaGetErrorString(code) << std::endl; }
	

	MotionPlanningProblem mpp;
	mpp.filename = argv[1];
	mpp.dimC = DIM;
	mpp.dimW = 3;
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
	std::vector<float> goal(DIM, 4.9);
	init[3] = M_PI/2;
	goal[3] = M_PI/2;

	int goalIdx = NUM-1;
	int initIdx = 0;

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
	} else {
		std::cout << "didn't read in file, bad file!" << std::endl;
	}

	std::cout << "--- Motion planning problem, " << mpp.filename << " ---" << std::endl;
	std::cout << "Sample count = " << mpp.numSamples << ", C-space dim = " << mpp.dimC << ", Workspace dim = " << mpp.dimW << std::endl;
	std::cout << "hi = ["; for (int i = 0; i < mpp.dimC; ++i) { std::cout << hi[i] << " "; } std::cout << "], ";
	std::cout << "lo = ["; for (int i = 0; i < mpp.dimC; ++i) { std::cout << lo[i] << " "; } std::cout << "]" << std::endl;
	std::cout << "edge discretizations " << mpp.edgeDiscNum << ", dt = " << mpp.dt << std::endl;

	/*********************** create array to return debugging information ***********************/
	float *d_debugOutput;
	cudaMalloc(&d_debugOutput, sizeof(float)*NUM);

	// ***************** setup data structures and struct


	// ***************** precomputation

	std::vector<float> samplesAll (DIM*NUM);
	createSamplesHalton(0, samplesAll.data(), &(init[0]), &(goal[0]), lo, hi);
	thrust::device_vector<float> d_samples_thrust(DIM*NUM);
	float *d_samples = thrust::raw_pointer_cast(d_samples_thrust.data());
	CUDA_ERROR_CHECK(cudaMemcpy(d_samples, samplesAll.data(), sizeof(float)*DIM*NUM, cudaMemcpyHostToDevice));

	// calculate nn
	int nullControl[3]; // throw away values for computing the cost of each connection
	float nullControlD[3]; 

	double t_calc10thStart = std::clock();
	std::vector<float> topts ((NUM-1)*(NUM));
	std::vector<float> copts ((NUM-1)*(NUM));
	float percentile = 0.05;
	int rnIdx = (NUM-1)*(NUM)*percentile; // 10th quartile and number of NN
	int numEdges = rnIdx;
	std::cout << "rnIdx is " << rnIdx << std::endl;
	std::cout << "This is a bit slow as it is currently implemented on CPU, can be sped up significantly with a simple GPU implementation" << std::endl;
	int idx = 0;
	std::vector<float> c2g (NUM);
	for (int i = 0; i < NUM; ++i) {
		for (int j = 0; j < NUM; ++j) {
			if (j == i)
				continue;
			topts[idx] = dubinsAirplaneCost(&(samplesAll[i*DIM]), &(samplesAll[j*DIM]), nullControl, nullControlD);
			copts[idx] = dubinsAirplaneCost(&(samplesAll[i*DIM]), &(samplesAll[j*DIM]), nullControl, nullControlD);
			idx++;
		}
		c2g[i] = dubinsAirplaneCost(&(samplesAll[i*DIM]), &(samplesAll[goalIdx*DIM]), nullControl, nullControlD);
	}
	std::vector<float> coptsSorted ((NUM-1)*(NUM));
	coptsSorted = copts;
	std::sort (coptsSorted.begin(), coptsSorted.end());
	float rn = coptsSorted[rnIdx];
	double t_calc10th = (std::clock() - t_calc10thStart) / (double) CLOCKS_PER_SEC;
	std::cout << percentile << "th percentile pre calc took: " << t_calc10th*ms << " ms for " << idx << " solves and cutoff is " 
		<< rn << " at " << rnIdx << std::endl;	

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
	std::vector<float> discMotions (numEdges*(numControls*numDisc)*DIM,0); // array of motions, but its a vector who idk for some reason it won't work as an array
	int nnIdx = 0; // index position in NN discretization array
	idx = 0; // index position in copts vector above

	std::vector<float> coptsEdge (numEdges); // edge index accessed copts
	std::vector<float> toptsEdge (numEdges); // edge index accessed topts
	std::vector<int> controlEdge (numEdges*numControls); // edge index accessed controls
	std::vector<float> controlDEdge (numEdges*numControls); // edge index accessed control durations
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
				float tmpC = dubinsAirplaneCost(x0, x1, &(controlEdge[nnIdx*numControls]), 
					&(controlDEdge[nnIdx*numControls]));
				dubinsAirplanePath(x0, x1, 
					&(controlEdge[nnIdx*numControls]), 
					&(controlDEdge[nnIdx*numControls]), 
					&(discMotions[nnIdx*DIM*(numControls*numDisc)]), numDisc);
				nnGoSizes[i]++;
				nnComeSizes[j]++;
				nnIdx++;
			}
			idx++;
		}
	}
	double t_discMotions = (std::clock() - t_discMotionsStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Discretizing motions took: " << t_discMotions*ms << " ms for " << nnIdx << " solves" << std::endl;

	// printArray(&(discMotions[20000*DIM]), 200, DIM, std::cout);
	// printArray(&(nnGoSizes[0]), 1, 1000, std::cout);

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
	CUDA_ERROR_CHECK(cudaMalloc(&d_discMotions, sizeof(float)*numEdges*(numControls*numDisc)*DIM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_discMotions, &discMotions[0], sizeof(float)*numEdges*(numControls*numDisc)*DIM, cudaMemcpyHostToDevice));
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


	// ***************** read in online problem parameters from filename input
	// obstacles
	std::cout << "Obstacle set, count = " << numObstacles << ":" << std::endl;
	// printArray(obstacles.data(), numObstacles, 2*DIM, std::cout);

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
		d_samples, NUM, d_isFreeSamples, rn, numDisc*numControls-1,
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
		d_samples, NUM, d_isFreeSamples, rn, numDisc*numControls-1, numEdges,
		costs, d_edges_ptr, initIdx, goalIdx,
		c2g);
	double t_PRM = (std::clock() - t_PRMStart) / (double) CLOCKS_PER_SEC;
	std::cout << "******** PRM took: " << t_PRM << " s" << std::endl;	

	// ***************** FMT
	double t_FMTStart = std::clock();
	std::cout << "Running FMT" << std::endl;
	// call FMT
	FMTdub(&(init[0]), &(goal[0]), obstacles.data(), numObstacles,
		adjCosts, nnGoEdges, nnComeEdges, maxNNSize, discMotions, nnIdxs,
		NUM, d_isFreeSamples, rn, numDisc*numControls-1, numEdges,
		costs, initIdx, goalIdx,
		samplesAll, adjTimes,
		numControls, controlEdge, controlDEdge);
	
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