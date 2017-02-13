#include "dubinsAirplane.cuh"

// (control means -1 = right, 0 = straight, 1 = left) 
// state is [x y z theta]
float dubinsAirplaneCost(float *x0, float *x1, int* control, float* controlD)
{
	float v[2] = {x1[0] - x0[0], x1[1] - x0[1]};
	float d = sqrt(pow(v[0],2) + pow(v[1],2));
	float th = atan2(v[1], v[0]);
	float a = mod2pi(x0[3] - th);
	float b = mod2pi(x1[3] - th);
	float cmin = FLT_MAX;
	float ca = cos(a); float sa = sin(a); float cb = cos(b); float sb = sin(b);

	cmin = dubinsLSLCost(d, a, ca, sa, b, cb, sb, cmin, control, controlD);
	cmin = dubinsRSRCost(d, a, ca, sa, b, cb, sb, cmin, control, controlD);
	cmin = dubinsRSLCost(d, a, ca, sa, b, cb, sb, cmin, control, controlD);
	cmin = dubinsLSRCost(d, a, ca, sa, b, cb, sb, cmin, control, controlD);
	cmin = dubinsRLRCost(d, a, ca, sa, b, cb, sb, cmin, control, controlD);
	cmin = dubinsLRLCost(d, a, ca, sa, b, cb, sb, cmin, control, controlD);
	cmin = sqrt(pow(cmin,2) + pow(x1[2]-x0[2],2));
	return cmin;
}

float dubinsLSLCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD)
{
	float tmp = 2 + d*d - 2*(ca*cb + sa*sb - d*(sa - sb));
	if (tmp < 0) { return c; }
	float th = atan2(cb - ca, d + sa - sb);
	float t = mod2pi(-a + th);
	float p = sqrt(std::max(tmp, (float) 0));
	float q = mod2pi(b - th);
	float cnew = t + p + q;
	// std::cout << "LSL: (" << t << ", " << p << ", " << q << "), c = " << cnew << std::endl;
	if (cnew <= c) {
		c = cnew; 
		control[0] = 1; 	control[1] = 0; 	control[2] = 1;
		controlD[0] = t; 	controlD[1] = p; 	controlD[2] = q;
	}
	return c;
}

float dubinsRSRCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD)
{
	float tmp = 2 + d*d - 2*(ca*cb + sa*sb - d*(sb - sa));
	if (tmp < 0) { return c; }
	float th = atan2(ca - cb, d - sa + sb);
	float t = mod2pi(a - th);
	float p = sqrt(std::max(tmp, (float) 0));
	float q = mod2pi(-b + th);
	float cnew = t + p + q;
	// std::cout << "RSR: (" << t << ", " << p << ", " << q << "), c = " << cnew << std::endl;
	if (cnew <= c) {
		c = cnew; 
		control[0] = -1; 	control[1] = 0; 	control[2] = -1;
		controlD[0] = t; 	controlD[1] = p; 	controlD[2] = q;
	}
	return c;
}

float dubinsRSLCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD)
{
	float tmp = d * d - 2 + 2 * (ca*cb + sa*sb - d * (sa + sb));
	if (tmp < 0) { return c; }
	float p = sqrt(std::max(tmp, (float) 0));
	float th = atan2(ca + cb, d - sa - sb) - atan2(2, p);
	float t = mod2pi(a - th);
	float q = mod2pi(b - th);
	float cnew = t + p + q;
	// std::cout << "RSL: (" << t << ", " << p << ", " << q << "), c = " << cnew << std::endl;
	if (cnew <= c) {
		c = cnew; 
		control[0] = -1; 	control[1] = 0; 	control[2] = 1;
		controlD[0] = t; 	controlD[1] = p; 	controlD[2] = q;
	}
	return c;
}

float dubinsLSRCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD)
{
	float tmp = -2 + d * d + 2 * (ca*cb + sa*sb + d * (sa + sb));
	if (tmp < 0) { return c; }
	float p = sqrt(std::max(tmp, (float) 0));
	float th = atan2(-ca - cb, d + sa + sb) - atan2(-2, p);
	float t = mod2pi(-a + th);
	float q = mod2pi(-b + th);
	float cnew = t + p + q;
	// std::cout << "LSR: (" << t << ", " << p << ", " << q << "), c = " << cnew << std::endl;
	if (cnew <= c) {
		c = cnew; 
		control[0] = 1; 	control[1] = 0; 	control[2] = -1;
		controlD[0] = t; 	controlD[1] = p; 	controlD[2] = q;
	}
	return c;
}

float dubinsRLRCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD)
{
	float tmp = (6 - d * d  + 2 * (ca*cb + sa*sb + d * (sa - sb))) / 8;
	if (abs(tmp) >= 1) { return c; }
	float p = 2*M_PI - acos(tmp);
	float th = atan2(ca - cb, d - sa + sb);
	float t = mod2pi(a - th + p/2);
	float q = mod2pi(a - b - t + p);
	float cnew = t + p + q;
	// std::cout << "RLR: (" << t << ", " << p << ", " << q << "), c = " << cnew << std::endl;
	if (cnew <= c) {
		c = cnew; 
		control[0] = -1; 	control[1] = 1; 	control[2] = -1;
		controlD[0] = t; 	controlD[1] = p; 	controlD[2] = q;
	}
	return c;
}

float dubinsLRLCost(float d, float a, float ca, float sa, float b, float cb, float sb, float c, int* control, float* controlD)
{
	float tmp = (6 - d * d  + 2 * (ca*cb + sa*sb - d * (sa - sb))) / 8;
	if (abs(tmp) >= 1) { return c; }
	float p = 2*M_PI - acos(tmp);
	float th = atan2(-ca + cb, d + sa - sb);
	float t = mod2pi(-a + th + p/2);
	float q = mod2pi(b - a - t + p);
	float cnew = t + p + q;
	// std::cout << "LRL: (" << t << ", " << p << ", " << q << "), c = " << cnew << std::endl;
	if (cnew <= c) {
		c = cnew; 
		control[0] = 1; 	control[1] = -1; 	control[2] = 1;
		controlD[0] = t; 	controlD[1] = p; 	controlD[2] = q;
	}
	return c;
}

float mod2pi(float ang)
{
    float twoPi = 2.0 * M_PI;
    return ang - twoPi * floor( ang / twoPi );
}

void dubinsAirplanePath(float *x0, float *x1, int* control, float* controlD, float* path, int 
	numDisc)
{
	// load first control segment and transform
	carControl(control[0], controlD[0], &path[0], numDisc);
	transformPath(x0, &path[0], numDisc);

	// load second control segment and transform
	carControl(control[1], controlD[1], &path[DIM*numDisc*1], numDisc);
	transformPath(&path[DIM*numDisc*1-DIM], &path[DIM*numDisc*1], numDisc);

	// load third control segment and transform
	carControl(control[2], controlD[2], &path[DIM*numDisc*2], numDisc);
	transformPath(&path[DIM*numDisc*2-DIM], &path[DIM*numDisc*2], numDisc);

	// add the z component
	float localCumSum[numDisc*3];
	float normLocalCumSum[numDisc*3];
	float cumSum = 0;
	float dz = x1[2] - x0[2];

	localCumSum[0] = 0;
	for (int i = 1; i < numDisc*3; ++i) {
		float dx = path[i*DIM] - path[(i-1)*DIM];
		float dy = path[i*DIM+1] - path[(i-1)*DIM+1];
		cumSum += sqrt(pow(dx,2) + pow(dy,2));
		localCumSum[i] = cumSum;
	}
	for (int i = 1; i < numDisc*3; ++i) 
		normLocalCumSum[i] = localCumSum[i]/cumSum;
	path[2] = x0[2];
	for (int i = 1; i < numDisc*3; ++i) 
		path[i*DIM + 2] = x0[2] + normLocalCumSum[i]*dz;
}


void carControl(int control, float d, float* path, int numDisc)
{
	float dt = d/((float) numDisc-1);
	for (int i = 0; i < numDisc; ++i) {
		if (control < 0) { // right turn
			path[DIM*i + 0] = sin(dt*i); // x component
			path[DIM*i + 1] = cos(dt*i)-1; // y component
			path[DIM*i + 3] = -dt*i; // theta component
		} else if (control == 0) { // straight line
			path[DIM*i + 0] = dt*i; // x component
			path[DIM*i + 1] = 0; // y component
			path[DIM*i + 3] = 0; // theta component
		} else { // left turn
			path[DIM*i + 0] = sin(dt*i); // x component
			path[DIM*i + 1] = 1-cos(dt*i); // y component
			path[DIM*i + 3] = dt*i; // theta component
		}
	}
}

void transformPath(float *xinit, float* path, int numDisc)
{
	float xi = xinit[0];
	float yi = xinit[1];
	float thi = xinit[3]; 
	for (int i = 0; i < numDisc; ++i) {
		float x = path[i*DIM + 0];
		float y = path[i*DIM + 1];
		path[i*DIM + 0] = cos(thi)*x - sin(thi)*y + xi;
		path[i*DIM + 1] = sin(thi)*x + cos(thi)*y + yi; 
		path[i*DIM + 3] += thi;
	}
}
