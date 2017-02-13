/*
sampler.cuh
author: Brian Ichter

This file includes state sampling methods
*/

#pragma once

#include <cstdlib>
#include <iostream>

void createSamplesIID(int seed, float *samples, float *initial, float *goal, float *lo, float *hi);
void createSamplesHalton(int skip, float *samples, float *initial, float *goal, float *lo, float *hi);
float localHaltonSingleNumber(int n, int b);

// halton
// lattice (may want to use for NN benefits)

__global__ void sampleFree(float* obstacles, int obstaclesCount, float* samples, bool* isFreeSamples, float *debugOutput);
__global__ void fillSamples(float* samples, float* samplesAll, int* sampleFreeIdx, bool* isFreeSamples, float *debugOutput);
__global__ void createSortHeuristic(float* samples, int initial_idx, float* heuristic, int samplesCount);
bool sampleFreePt(float* obstacles, int obstaclesCount, float* sample);