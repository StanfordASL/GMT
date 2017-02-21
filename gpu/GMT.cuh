/*
GMT.cuh
author: Brian Ichter

Group Marching Tree algorithm
*/
#pragma once

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "collisionCheck.cuh"
#include "helper.cuh"
#include "2pBVP.cuh"

/***********************
CPU functions
***********************/
// GMT if lambda < 1 (not currently implemented, but a similar goemetric version is in geometricGMT.cu)
void GMT(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	float *d_distancesCome, int *d_nnGoEdges, int *d_nnComeEdges, int nnSize, float *d_samples, int samplesCount,
	float lambda, float r, float *d_costs, int *d_edges, int initial_idx, int goal_idx);

// pregenerated wavefront with initial condition attached
void GMTinit(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	float *d_distancesCome, int *d_nnGoEdges, int *d_nnComeEdges, int nnSize, float *d_discMotions, int *d_nnIdxs,
	float *d_samples, int samplesCount, bool *d_isFreeSamples, float r, int numDisc,
	float *d_costs, int *d_edges, int initial_idx, int goal_idx);

// pregenerated wavefront with initial condition attached and moving goal, return goal index
int GMTinitGoal(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	float *d_distancesCome, int *d_nnGoEdges, int *d_nnComeEdges, int nnSize, float *d_discMotions, int *d_nnIdxs,
	float *d_samples, int samplesCount, bool *d_isFreeSamples, float r, int numDisc,
	float *d_costs, int *d_edges, int initial_idx);

// GMT with lambda = 1 (i.e. expand entire wavefront at once)
void GMTwavefront(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	float *d_distancesCome, int *d_nnGoEdges, int *d_nnComeEdges, int nnSize, float *d_discMotions, int *d_nnIdxs,
	float *d_samples, int samplesCount, bool *d_isFreeSamples, float r, int numDisc,
	float *d_costs, int *d_edges, int initial_idx, int goal_idx);

void outputSolution();

/***********************
GPU kernels
***********************/
// set default values for arrays
__global__ void setupArrays(bool *wavefront, bool *wavefrontNew, bool *wavefrontWas, bool *unvisited, 
	bool *isFreeSamples, float *costGoal, float *costs, int *edges, int samplesCount);
// find new nodes to expand to
__global__ void findWavefront(bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas,
	int *nnEdgesGo, int nnSize, int wavefrontSize, int *wavefrontActiveIdx, float *debugOutput);
// update arrays to represent new wavefront
__global__ void fillWavefront(int samplesCount, int *wavefrontActiveIdx, int *wavefrontScanIdx, bool *wavefront);
// find the optimal connection to the tree
__global__ void findOptimalConnection(bool *wavefront, int *edges, float *costs, float *distancesCome, 
	int *nnEdgesCome, int nnSize, int wavefrontSize, int *wavefrontActiveIdx, float *debugOutput);
// check if optimal connection is collision free
__global__ void verifyExpansion(int obstaclesCount, bool *wavefrontNew, int *edges,
	float *samples, float *obstacles, float *costs,
	int *nnIdxs, float *discMotions, int numDisc, bool *isCollision,
	int wavefrontSize, int *wavefrontActiveIdx, float *debugOutput);
// remove edges found invalid through verifyExpansion (didn't work in verifyExpansion)
__global__ void removeInvalidExpansions(int obstaclesCount, bool *wavefrontNew, int *edges,
	float *samples, float *obstacles, float *costs,
	int *nnIdxs, float *discMotions, int numDisc, bool *isCollision,
	int wavefrontSize, int *wavefrontActiveIdx, float *debugOutput);
// update arrays to represent new wavefront
__global__ void updateWavefront(int samplesCount, int goal_idx,
	bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas,
	float *costs, float *costGoal, float *debugOutput);
// attach initial to the graph
__global__ void attachWavefront(int samplesCount, float *samples, float *initial, 
	float r, int goal_idx, bool *wavefront, int *edges,
	int obstaclesCount, float *obstacles, 
	float* costs, float* costGoal, float* debugOutput);
// build inGoal array to store samples that are inside the goal region
__global__ void buildInGoal(int samplesCount,float *samples, float *goal, float goalD2,
	bool *inGoal, float *debugOutput);