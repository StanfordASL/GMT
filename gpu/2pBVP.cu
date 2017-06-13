/*
Author: brian ichter
Solves a two point boundary value problem for double integrator dynamics given a mixed time/control effort
cost function.
*/
#include "2pBVP.cuh"

const float g = -9.81;

// fill splitPath with the optimal path along waypoints and return the time of the path
float findOptimalPath(float dt, float *splitPath, float *xs, int numWaypoints, int *pathLength)
{
    float tmax = 5;
    float toff = 0;
    float topt = 0;
    float x0[DIM], x1[DIM];
    int Tprev = 0;
    float t = 0;
    float ttotal = 0;
    for (int i = 0; i < numWaypoints-1; ++i) {
        // load waypoints into x0 and x1
        for (int d = 0; d < DIM; ++d) {
            x0[d] = xs[d + i*DIM];
            x1[d] = xs[d + (i+1)*DIM];
        }
        // perform bisection search to find optimal time
        topt = toptBisection(x0, x1, tmax);
        // std::cout << "optimal time is " << topt << std::endl;
        // load up the splitPath array with the path
        int T = (topt-toff)/dt + 1;
        for (int tidx = 0; tidx < T; ++tidx) {
            t = dt*tidx + toff;
            pathPoint(t, topt, x0, x1, &splitPath[Tprev*DIM + tidx*DIM]);
        }
        // update timing
        Tprev += T;
        toff = (t-topt) + dt;
        ttotal += topt;
    }
    *pathLength = Tprev;
    return ttotal;
}

// given the optimal time, fill splitPath
void discretizePath(float *splitPath, float *x0, float *x1, int numDisc, float topt)
{
    float dt = topt/numDisc;
    // load up start and end points
    for (int d = 0; d < DIM; ++d) { 
        splitPath[d] = x0[d];
        splitPath[d + numDisc*DIM] = x1[d];
    }

    // load way points
    for (int i = 1; i < numDisc; ++i) {
        pathPoint(dt*i, topt, x0, x1, &splitPath[i*DIM]);
    }
}

// find the optimal path between two points
__device__ __host__
float findDiscretizedPath(float *splitPath, float *x0, float *x1, int numDisc)
{
    float tmax = 20;
    float topt =toptBisection(x0, x1, tmax);
    
    float dt = topt/numDisc;
    for (int d = 0; d < DIM; ++d) {
        splitPath[d] = x0[d];
        splitPath[d + numDisc*DIM] = x1[d];
    }

    for (int i = 1; i < numDisc; ++i) {
        pathPoint(dt*i, topt, x0, x1, &splitPath[i*DIM]);
    }

    return topt;
}

// bisection search to find the time for the optimal path
__device__ __host__
float toptBisection(float *x0, float *x1, float tmax)
{
    float TOL = 0.0001;
	float tu = tmax;
    if (dcost(tmax,x0,x1) < 0) {
        return tmax;
    }
    float tl = 0.01;
    while (dcost(tl,x0,x1) > 0) {
        tl = tl/2;
    }
    float topt = 0;
    float dcval = 1;
    int maxItrs = 50;
    int itr = 0;
    while (abs(dcval) > TOL && itr < maxItrs && ++itr) {
        topt = (tu+tl)/2;
        dcval = dcost(topt,x0,x1);
        if (dcval > 0)
            tu = topt;
        else 
            tl = topt;
    }
	return topt;
}

// compute the cost for the optimal path
__device__ __host__
float cost(float tau, float *x0, float *x1)
{
	float cost = (1/pow(tau,3))*(12*pow(x0[0],2) + 12*pow(x0[2],2) + 12*pow(x0[1],2) + 12*pow(x1[0],2) + 12*pow(x1[2],2) - 24*x0[1]*x1[1] + 
	    12*pow(x1[1],2) + 12*x0[4]*x0[1]*tau + 12*x0[1]*x1[4]*tau - 12*x0[3]*x1[0]*tau - 12*x1[3]*x1[0]*tau - 
	    12*x0[5]*x1[2]*tau - 12*x1[5]*x1[2]*tau - 12*x0[4]*x1[1]*tau - 12*x1[4]*x1[1]*tau + 4*pow(x0[4],2)*pow(tau,2) + 
	    4*pow(x0[3],2)*pow(tau,2) + 4*pow(x0[5],2)*pow(tau,2) + 4*x0[4]*x1[4]*pow(tau,2) + 4*pow(x1[4],2)*pow(tau,2) + 4*x0[3]*x1[3]*pow(tau,2) + 
	    4*pow(x1[3],2)*pow(tau,2) + 4*x0[5]*x1[5]*pow(tau,2) + 4*pow(x1[5],2)*pow(tau,2) + 2*g*x0[5]*pow(tau,3) - 2*g*x1[5]*pow(tau,3) + pow(tau,4) + 
	    pow(g,2)*pow(tau,4) + 12*x0[0]*(-2*x1[0] + (x0[3] + x1[3])*tau) + 12*x0[2]*(-2*x1[2] + (x0[5] + x1[5])*tau));
	return cost;
}

// derivative of cost
__device__ __host__
float dcost(float tau, float *x0, float *x1)
{
    float dtau = 0.000001;
	float dcost = (cost(tau+dtau/2,x0,x1) - cost(tau-dtau/2,x0,x1))/dtau;
	return dcost;
}

// fills out the state along a given path at time tau
__device__ __host__
void pathPoint(float t, float tau, float *x0, float *x1, float *x)
{
	x[0] = t*x0[3] + x0[0] + (pow(t,3)*(2*x0[0] - 2*x1[0] + (x0[3] + x1[3])*tau))/pow(tau,3) - 
            (pow(t,2)*(3*x0[0] - 3*x1[0] + (2*x0[3] + x1[3])*tau))/pow(tau,2);
	x[1] = t*x0[4] + x0[1] + (pow(t,3)*(2*x0[1] - 2*x1[1] + (x0[4] + x1[4])*tau))/pow(tau,3) - 
            (pow(t,2)*(3*x0[1] - 3*x1[1] + (2*x0[4] + x1[4])*tau))/pow(tau,2);
	x[2] = (t*x0[5]*pow(tau,3) + x0[2]*pow(tau,3) + pow(t,2)*tau*(-3.*x0[2] + 3.*x1[2] - 2.*x0[5]*tau - 1.*x1[5]*tau) + 
             pow(t,3)*(2.*x0[2] - 2.*x1[2] + (x0[5] + x1[5])*tau))/pow(tau,3);
	x[3] = -(2*t*(3*x0[0] - 3*x1[0] + x1[3]*tau))/pow(tau,2) + (3*pow(t,2)*(2*x0[0] - 2*x1[0] + x1[3]*tau))/pow(tau,3) + 
            (x0[3]*(3*pow(t,2) - 4*t*tau + pow(tau,2)))/pow(tau,2);
	x[4] = -(2*t*(3*x0[1] - 3*x1[1] + x1[4]*tau))/pow(tau,2) + 
            (3*pow(t,2)*(2*x0[1] - 2*x1[1] + x1[4]*tau))/pow(tau,3) + (x0[4]*(3*pow(t,2) - 4*t*tau + pow(tau,2)))/pow(tau,2), 
	x[5] = (x0[5]*pow(tau,3) + t*tau*(-6.*x0[2] + 6.*x1[2] - 4.*x0[5]*tau - 2.*x1[5]*tau) + 
             pow(t,2)*(6.*x0[2] - 6.*x1[2] + 3.*x0[5]*tau + 3.*x1[5]*tau))/pow(tau,3);
}

__global__ 
void fillCoptsTopts(float *samples, float *copts, float *topts, float tmax) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM*NUM)
        return;

    int i = tid/NUM;
    int j = tid%NUM;
    if (i == j) // connection to the same node
        return;

    int idx = i*(NUM-1)+j;
    if (i < j) 
        idx--;
    
    topts[idx] = toptBisection(&(samples[i*DIM]), &(samples[j*DIM]), tmax);
    copts[idx] = cost(topts[idx], &(samples[i*DIM]), &(samples[j*DIM]));
}