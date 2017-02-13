/*
collisionCheck.cuh
author: Brian Ichter

CUDA collision checker is contained for AABB
*/

#pragma once
#include <vector>

// splits an motion into its waypoints, returns true if motion is valid
__device__ void waypointCollisionCheck(int v_idx, int w_idx, int obstaclesCount, float* obstacles, 
	int *nnIdxs, float *discMotions, int discIdx, int numDisc, bool *isCollision, int tid, float *debugOutput);
__device__ bool isMotionValid(float *v, float *w, float *bbMin, float *bbMax, int obstaclesCount, float* obstacles, float *debugOutput);
__device__ bool broadphaseValidQ(float *bbMin, float *bbMax, float *obs, float *debugOutput);
__device__ bool motionValidQ(float *v, float *w, float *obs, float *debugOutput);
__device__ bool faceContainsProjection(float *v, float *w, float lambda, int j, float *obs, float* debugOutput);
__device__ bool faceContainsProjectionError(float *v, float *v_to_w, float lambda, int j, float *obs, float* debugOutput);
__global__ void freeEdges(float *obstacles, int obstaclesCount, float *samples, 
	bool *isFreeSamples, int numDisc, float *discMotions, bool *isFreeEdges, int numEdges, float *debugOutput);

// CPU
void waypointCollisionCheck_h(int v_idx, int w_idx, int obstaclesCount, float* obstacles, 
	int *nnIdxs, float *discMotions, int discIdx, int numDisc, bool *isCollision, int tid, float *debugOutput);
bool isMotionValid_h(float *v, float *w, float *bbMin, float *bbMax, int obstaclesCount, float* obstacles, float *debugOutput);
bool broadphaseValidQ_h(float *bbMin, float *bbMax, float *obs, float *debugOutput);
bool motionValidQ_h(float *v, float *w, float *obs, float *debugOutput);
bool faceContainsProjection_h(float *v, float *w, float lambda, int j, float *obs, float* debugOutput);
bool faceContainsProjectionError_h(float *v, float *v_to_w, float lambda, int j, float *obs, float* debugOutput);
bool isFreeEdge_h(int edgeIdx, float *obstacles, int obstaclesCount, 
	int numDisc, std::vector<float> discMotions, float *debugOutput);