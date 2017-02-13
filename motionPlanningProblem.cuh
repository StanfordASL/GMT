/*
motionPlanningProblem.cuh
Author: Brian Ichter

Struct definition for a motion planning problem.
*/

#pragma once

#include "motionPlanningSolution.cuh"
#include <iostream>
#include <vector>

typedef struct MotionPlanningProblem {
	const char *filename;
	int dimW; // workspace dimension
	int dimC; // configuration space dimension
	int numSamples; // number of samples
	std::vector<float> hi; // high bound on configuration space
	std::vector<float> lo; // low bound on configuration space

	float cpTarget; // target collision probability
	int mcSamplesNum; // number of Monte Carlo samples for CP calculation
	float lambda; // expansion cost threshold expansion rate (threshold += rn*lambda at each step)
	float dt;

	int edgeDiscNum; // for collision checks, number of discretizations in edge representation
	int obstaclesNum; // 
	std::vector<float> obstacles;
	float *d_obstacles;

	std::vector<float> init; // initial state
	std::vector<float> goal; // goal state
	int initIdx; // index in samples of 
	int goalIdx; 

	// calculated
	float rnPerc; // percentile cutoff of for nn connections (e.g. 0.1 means 10th percentile)
	float rn; // nearest neighbor radius, as 100*rnPerc percentile connection cost

	// host and device big arrays
	// samples
	std::vector<float> samples; // samples[i*DIM + d] is sample i, dimension d 
	float *d_samples; // size NUM*DIM

	// nn's
	std::vector<bool> nn; // nn[i*NUM + j] is true if a connection exists from i -> j
	bool *d_nn; // size NUM*NUM
	std::vector<bool>  nnT; // (nn' for locality) nn[j*NUM + i] is true if a connection exists from i -> j
	bool *d_nnT; // size NUM*NUM
	std::vector<int> nnIdx; // nnIdx[i*NUM*j] gives the index in nnEdges of the edge connecting i -> j
	int *d_nnIdx; // size NUM*NUM
	std::vector<float> nnEdges; // discretized edges for nn optimal path connections, nnEdges[i*DIM*edgeDiscNum + j*DIM + d] is the state for edge i, waypoint j, and dimension d
	float *d_nnEdges; // size NUM*(NUM-1)*rnPerc*DIM*edgeDiscNum

	std::vector<float> costs; // costs for edge i
	float *d_costs; // size NUM*(NUM-1)*rnPerc
	std::vector<float> times; // times for edge i
	float *d_times; // size NUM*(NUM-1)*rnPerc

	// uncertainty (either offsets hardcoded in LQG or matricies for uncertainty)


	// precomputed path offsets
	

	// other preallocated
	

	MotionPlanningSolution *soln; // solution to problem
} MotionPlanningProblem;

void printMotionPlanningProblem(MotionPlanningProblem mpp, std::ostream& stream);
// sizeMotionPlanningProblem // once all ints for sizing are initialized, we size the vectors