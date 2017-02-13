#include "collisionCheck.cuh"

// TODO: fix to only check collision in the spatial dimensions
// 3 has been fixed to be the workspace dimension

__global__ 
void freeEdges(float *obstacles, int obstaclesCount, float *samples, 
	bool *isFreeSamples, int numDisc, float *discMotions, 
	bool *isFreeEdges, int numEdges, float *debugOutput) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numEdges)
		return;

	float v[DIM], w[DIM];
	float bbMin[DIM], bbMax[DIM];
	bool motionValid = true;
	for (int i = 0; i < numDisc; ++i) {
		if (!motionValid)
			break;
		int baseIdx = tid*(numDisc+1)*DIM + i*DIM;
		for (int d = 0; d < DIM; ++d) {
			v[d] = discMotions[baseIdx + d];
			w[d] = discMotions[baseIdx + d + DIM];

			if (v[d] > w[d]) {
				bbMin[d] = w[d];
				bbMax[d] = v[d];
			} else {
				bbMin[d] = v[d];
				bbMax[d] = w[d];
			}
		}
		motionValid = motionValid && isMotionValid(v, w, bbMin, bbMax, obstaclesCount, obstacles, debugOutput);
	}

	isFreeEdges[tid] = motionValid;
}

__device__ 
void waypointCollisionCheck(int v_idx, int w_idx, int obstaclesCount, float* obstacles, 
	int *nnIdxs, float *discMotions, int discIdx, int numDisc, bool *isCollision, int tid, float *debugOutput)
{ 
	// motion from w_idx to v_idx
	int discMotionsIdx = nnIdxs[v_idx*NUM + w_idx];

	// calculate bounds of the bounding box
	float v[DIM], w[DIM]; // TODO: do v and w need ot be vectors?
	float bbMin[DIM], bbMax[DIM];
	for (int d = 0; d < DIM; ++d) {
		v[d] = discMotions[discMotionsIdx*(numDisc+1)*DIM + discIdx*DIM + d];
		w[d] = discMotions[discMotionsIdx*(numDisc+1)*DIM + (discIdx+1)*DIM + d];

		if (v[d] > w[d]) {
			bbMin[d] = w[d];
			bbMax[d] = v[d];
		} else {
			bbMin[d] = v[d];
			bbMax[d] = w[d];
		}
	}

	isCollision[tid] = !isMotionValid(v, w, bbMin, bbMax, obstaclesCount, obstacles, debugOutput);
}

__device__
bool isMotionValid(float *v, float *w, float *bbMin, float *bbMax, int obstaclesCount, float* obstacles, float *debugOutput)
{
	// TODO: eventually put each point (v, w) into shared memory
	// TODO: read http://http.developer.nvidia.com/GPUGems3/gpugems3_ch32.html
	// identify which obstacle this processor is checking against

	// I don't think necessary, but routine to check if point is within an obstacle
	// for (int obsIdx = 0; obsIdx < obstaclesCount; ++obsIdx) {
	// 	bool notFree = true;
	// 	for (int d = 0; d < 3; ++d) {
	// 		notFree = notFree && 
	// 		v[d] > obstacles[obsIdx*2*DIM + d] && 
	// 		v[d] < obstacles[obsIdx*2*DIM + DIM + d];
	// 		if (!notFree)
	// 			break;
	// 	}
	// 	if (notFree) {
	// 		return false;
	// 	}
	// }

	// bool same = true;
	// for (int d = 0; d < DIM/2; ++d)
	// 	same = same && (v[d] == w[d]);
	// if (same)
	// 	return true;

	// go through each obstacle and do broad then narrow phase collision checking
	for (int obsIdx = 0; obsIdx < obstaclesCount; ++obsIdx) {
		float obs[DIM*2];
		for (int d = 0; d < DIM; ++d) {
			obs[d] = obstacles[obsIdx*2*DIM + d];
			obs[DIM+d] = obstacles[obsIdx*2*DIM + DIM + d];
		}
		if (!broadphaseValidQ(bbMin, bbMax, obs, debugOutput)) {
			bool motionValid = motionValidQ(v, w, obs, debugOutput);
			if (!motionValid) {
				return false;
			}
		}
	}
	return true;
}

__device__
bool broadphaseValidQ(float *bbMin, float *bbMax, float *obs, float *debugOutput) 
{
	for (int d = 0; d < 3; ++d) {
		if (bbMax[d] <= obs[d] || obs[DIM+d] <= bbMin[d]) 
			return true;
	}
	return false;
}

__device__
bool motionValidQ(float *v, float *w, float *obs, float *debugOutput) 
{
	float v_to_w[3];

	for (int d = 0; d < 3; ++d) {
		float lambda;
		v_to_w[d] = w[d] - v[d];
		if (v[d] < obs[d]) {
			lambda = (obs[d] - v[d])/v_to_w[d];
		} else {
			lambda = (obs[DIM + d] - v[d])/v_to_w[d];
		}
		if (faceContainsProjection(v, w, lambda, d, obs, debugOutput))
			return false;
	}
	return true;
}

__device__
bool faceContainsProjection(float *v, float *w, float lambda, int j, float *obs, 
	float* debugOutput)
{
	for (int d = 0; d < 3; ++d) {
		float projection = v[d] + (w[d] - v[d])*lambda;
		if (d != j && !(obs[d] <= projection && projection <= obs[DIM+d]))
			return false;
	}
	return true;
}

// odd bug when called with v_to_w where the value is not passed correctly 
// resulting in v_to_w[d] = -2e+30 (for example), and collisions being allowed through
// this code is left here to remind me of the error/so I can figure it out later
__device__
bool faceContainsProjectionError(float *v, float *v_to_w, float lambda, int j, float *obs, 
	float* debugOutput)
{
	for (int d = 0; d < 3; ++d) {
		float projection = v[d] + v_to_w[d]*lambda;
		if (d != j && !(obs[d] <= projection && projection <= obs[DIM+d]))
			return false;
	}
	return true;
}

// CPU versions

bool isFreeEdge_h(int edgeIdx, float *obstacles, int obstaclesCount, 
	int numDisc, std::vector<float> discMotions, float *debugOutput) {

	float v[DIM], w[DIM];
	float bbMin[DIM], bbMax[DIM];
	bool motionValid = true;
	for (int i = 0; i < numDisc; ++i) {
		if (!motionValid)
			break;
		int baseIdx = edgeIdx*(numDisc+1)*DIM + i*DIM;
		for (int d = 0; d < DIM; ++d) {
			v[d] = discMotions[baseIdx + d];
			w[d] = discMotions[baseIdx + d + DIM];

			if (v[d] > w[d]) {
				bbMin[d] = w[d];
				bbMax[d] = v[d];
			} else {
				bbMin[d] = v[d];
				bbMax[d] = w[d];
			}
		}
		motionValid = motionValid && isMotionValid_h(v, w, bbMin, bbMax, obstaclesCount, obstacles, debugOutput);
	}

	return motionValid;
}

 
void waypointCollisionCheck_h(int v_idx, int w_idx, int obstaclesCount, float* obstacles, 
	int *nnIdxs, float *discMotions, int discIdx, int numDisc, bool *isCollision, int tid, float *debugOutput)
{ 
	// motion from w_idx to v_idx
	int discMotionsIdx = nnIdxs[v_idx*NUM + w_idx];

	// calculate bounds of the bounding box
	float v[DIM], w[DIM]; // TODO: do v and w need ot be vectors?
	float bbMin[DIM], bbMax[DIM];
	for (int d = 0; d < DIM; ++d) {
		v[d] = discMotions[discMotionsIdx*(numDisc+1)*DIM + discIdx*DIM + d];
		w[d] = discMotions[discMotionsIdx*(numDisc+1)*DIM + (discIdx+1)*DIM + d];

		if (v[d] > w[d]) {
			bbMin[d] = w[d];
			bbMax[d] = v[d];
		} else {
			bbMin[d] = v[d];
			bbMax[d] = w[d];
		}
	}

	isCollision[tid] = !isMotionValid_h(v, w, bbMin, bbMax, obstaclesCount, obstacles, debugOutput);
}

bool isMotionValid_h(float *v, float *w, float *bbMin, float *bbMax, int obstaclesCount, float* obstacles, float *debugOutput)
{
	// TODO: eventually put each point (v, w) into shared memory
	// TODO: read http://http.developer.nvidia.com/GPUGems3/gpugems3_ch32.html
	// identify which obstacle this processor is checking against

	// I don't think necessary, but routine to check if point is within an obstacle
	// for (int obsIdx = 0; obsIdx < obstaclesCount; ++obsIdx) {
	// 	bool notFree = true;
	// 	for (int d = 0; d < 3; ++d) {
	// 		notFree = notFree && 
	// 		v[d] > obstacles[obsIdx*2*DIM + d] && 
	// 		v[d] < obstacles[obsIdx*2*DIM + DIM + d];
	// 		if (!notFree)
	// 			break;
	// 	}
	// 	if (notFree) {
	// 		return false;
	// 	}
	// }
	
	bool same = true;
	for (int d = 0; d < DIM/2; ++d)
		same = same && (v[d] == w[d]);
	if (same)
		return true;

	// go through each obstacle and do broad then narrow phase collision checking
	for (int obsIdx = 0; obsIdx < obstaclesCount; ++obsIdx) {
		float obs[DIM*2];
		for (int d = 0; d < DIM; ++d) {
			obs[d] = obstacles[obsIdx*2*DIM + d];
			obs[DIM+d] = obstacles[obsIdx*2*DIM + DIM + d];
		}
		if (!broadphaseValidQ_h(bbMin, bbMax, obs, debugOutput)) {
			bool motionValid = motionValidQ_h(v, w, obs, debugOutput);
			if (!motionValid) {
				return false;
			}
		}
	}
	return true;
}

bool broadphaseValidQ_h(float *bbMin, float *bbMax, float *obs, float *debugOutput) 
{
	for (int d = 0; d < 3; ++d) {
		if (bbMax[d] <= obs[d] || obs[DIM+d] <= bbMin[d]) 
			return true;
	}
	return false;
}

bool motionValidQ_h(float *v, float *w, float *obs, float *debugOutput) 
{
	float v_to_w[3];

	for (int d = 0; d < 3; ++d) {
		float lambda;
		v_to_w[d] = w[d] - v[d];
		if (v[d] < obs[d]) {
			lambda = (obs[d] - v[d])/v_to_w[d];
		} else {
			lambda = (obs[DIM + d] - v[d])/v_to_w[d];
		}
		if (faceContainsProjection_h(v, w, lambda, d, obs, debugOutput))
			return false;
	}
	return true;
}

bool faceContainsProjection_h(float *v, float *w, float lambda, int j, float *obs, 
	float* debugOutput)
{
	for (int d = 0; d < 3; ++d) {
		float projection = v[d] + (w[d] - v[d])*lambda;
		if (d != j && !(obs[d] <= projection && projection <= obs[DIM+d]))
			return false;
	}
	return true;
}

// odd bug when called with v_to_w where the value is not passed correctly 
// resulting in v_to_w[d] = -2e+30 (for example), and collisions being allowed through
// this code is left here to remind me of the error/so I can figure it out later
bool faceContainsProjectionError_h(float *v, float *v_to_w, float lambda, int j, float *obs, 
	float* debugOutput)
{
	for (int d = 0; d < 3; ++d) {
		float projection = v[d] + v_to_w[d]*lambda;
		if (d != j && !(obs[d] <= projection && projection <= obs[DIM+d]))
			return false;
	}
	return true;
}