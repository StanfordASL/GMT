#include "motionPlanningSolution.cuh"

typedef struct MotionPlanningSolution {
	std::vector<int> path; // list of path indexes
	float cost;
	float cp;
	float time;

} MotionPlanningSolution;