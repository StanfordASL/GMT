/*
dubinsAirplane.cuh
author: Brian Ichter

This file implements local steering for the Dubins aircraft, a vehicle which (in this code)
can in the x-y plane move (at unit velocity) straight or turn with a bounded turning radius (unit radius)
and can move freely in the z-direction. The state is [x y z theta] and the minimum distance path consists of 
dubins curves in the x-y and theta and constant zdot in the z direction.
*/

#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <limits.h>
#include <cfloat>

#define _USE_MATH_DEFINES

// naming conventions:
// Control is a vector of ints that is 3 long, where each element is the letter for the dubins 
// word. -1 means right, 0 straight, 1 left
// ControlD is a vector of floats that is 3 long and is the duration of each letter in the dubins word.

// find optimal cost and word (LSL, RSR, RSL, LSR, RLR, LRL)
// __device__ __host__
float dubinsAirplaneCost(float *x0, float *x1, int* control, float* controlD);

// find optimal path given word
// given x0 and x1 (states), control which specifies a word (3 elements -1, 0, or +1), and the duration of each control
void dubinsAirplanePath(float *x0, float *x1, int* control, float* controlD, float* path, int numDisc);

// ************* word cost functions
// return the cost of each manuever and the duration of each control (to later build the path)
float dubinsLSLCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD);
float dubinsRSRCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD);
float dubinsRSLCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD);
float dubinsLSRCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD);
float dubinsRLRCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD);
float dubinsLRLCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD);

// ************* word path functions
// return the path for each manuever
// void dubinsLSLPath(float d, float a, float b, float* path, int numDisc);
// void dubinsRSRPath(float d, float a, float b, float* path, int numDisc);
// void dubinsRSLPath(float d, float a, float b, float* path, int numDisc);
// void dubinsLRSPath(float d, float a, float b, float* path, int numDisc);
// void dubinsRLRPath(float d, float a, float b, float* path, int numDisc);
// void dubinsLRLPath(float d, float a, float b, float* path, int numDisc);

// ************* path builder functions
// ang mod 2*pi
float mod2pi(float ang);

// create path from control (control means -1 = right, 0 = straight, 1 = left) and d is the duration
void carControl(int control, float d, float* path, int numDisc);

// transform path by initial conditions (rotate and offset)
void transformPath(float *xinit, float* path, int numDisc);
