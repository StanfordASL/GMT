/*
FMT.cuh
author: Brian Ichter

FMT algorithm
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
#include "2pBVP.cuh"
#include "dubinsAirplane.cuh"

/***********************
CPU functions
***********************/
// GMT with lambda = 1 (i.e. expand entire wavefront at once)

void FMT(float *initial, float *goal, float *obstacles, int obstaclesCount,
	std::vector<float> adjCosts, std::vector<int> nnGoEdges, std::vector<int> nnComeEdges, 
	int nnSize, std::vector<float> discMotions, std::vector<int> nnIdxs,
	int samplesCount, bool *d_isFreeSamples, float r, int numDisc, int numEdges,
	std::vector<float> costs, int initial_idx, int goal_idx,
	std::vector<float> samplesAll, std::vector<float> times);

void FMTdub(float *initial, float *goal, float *obstacles, int obstaclesCount,
	std::vector<float> adjCosts, std::vector<int> nnGoEdges, std::vector<int> nnComeEdges, 
	int nnSize, std::vector<float> discMotions, std::vector<int> nnIdxs,
	int samplesCount, bool *d_isFreeSamples, float r, int numDisc, int numEdges,
	std::vector<float> costs, int initial_idx, int goal_idx,
	std::vector<float> samplesAll, std::vector<float> times,
	int numControls, std::vector<int> controlEdge, std::vector<float> controlDEdge);
