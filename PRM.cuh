/*
PRM.cuh
author: Brian Ichter

PRM algorithm
*/
#pragma once

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <iostream>     // std::cout
#include <algorithm>    // std::make_heap, std::pop_heap, std::push_heap, std::sort_heap
#include <vector>       // std::vector
#include <set>
#include <utility>


#include "collisionCheck.cuh"
#include "helper.cuh"

/***********************
CPU functions
***********************/
// GMT with lambda = 1 (i.e. expand entire wavefront at once)

void PRM(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	std::vector<float> adjCosts, std::vector<int> nnGoEdges, std::vector<int> nnComeEdges, int nnSize, float *d_discMotions, std::vector<int> nnIdxs,
	float *d_samples, int samplesCount, bool *d_isFreeSamples, float r, int numDisc, int numEdges,
	std::vector<float> costs, int *d_edges, int initial_idx, int goal_idx,
	std::vector<float> c2g);

/***********************
GPU kernels
***********************/