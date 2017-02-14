/*
Author: brian ichter
Solve the planning problem with probabilistic roadmaps.
*/

#include "PRM.cuh"


void PRM(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	std::vector<float> adjCosts, std::vector<int> nnGoEdges, std::vector<int> nnComeEdges, int nnSize, float *d_discMotions, std::vector<int> nnIdxs,
	float *d_samples, int samplesCount, bool *d_isFreeSamples, float r, int numDisc, int numEdges,
	std::vector<float> costs, int *d_edges, int initial_idx, int goal_idx,
	std::vector<float> c2g)
{
	// instantiate memory for online comp
	double t_totalStart = std::clock();

	thrust::device_vector<bool> d_isFreeEdges_thrust(numEdges);
	bool* d_isFreeEdges = thrust::raw_pointer_cast(d_isFreeEdges_thrust.data());
	bool isFreeEdges[numEdges];
	bool isFreeSamples[NUM];

	// remove edges in collision (fast)
	// track these edges and potentially remove from storage?
	double t_edgesValidStart = std::clock();
	const int blockSizeEdgesValid = 192;
	const int gridSizeEdgesValid = std::min((numEdges + blockSizeEdgesValid - 1) / blockSizeEdgesValid, 2147483647);
	if (gridSizeEdgesValid == 2147483647)
		std::cout << "...... ERROR: increase grid size for freeEdges" << std::endl;
	float d_debugOutput[1]; d_debugOutput[0] = 0;
	freeEdges<<<gridSizeEdgesValid,blockSizeEdgesValid>>>(
		d_obstacles, obstaclesCount, d_samples, 
		d_isFreeSamples, numDisc, d_discMotions, 
		d_isFreeEdges, numEdges, d_debugOutput);
	cudaDeviceSynchronize();
	float t_edgesValid = (std::clock() - t_edgesValidStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Edge validity check took: " << t_edgesValid << " s";

	double t_memTransStart = std::clock();
	cudaMemcpy(isFreeEdges, d_isFreeEdges, sizeof(bool)*numEdges, cudaMemcpyDeviceToHost);
	cudaMemcpy(isFreeSamples, d_isFreeSamples, sizeof(bool)*NUM, cudaMemcpyDeviceToHost);
	
	// initialCondition
	std::vector<bool> visit(NUM,false); // tracks if a node has been visited

	costs[initial_idx] = 0;
	std::vector<int> pathIdx(NUM,-1);

	double t_memTrans = (std::clock() - t_memTransStart) / (double) CLOCKS_PER_SEC;
	double t_sel = 0;
	double t_search = 0;
	double t_conns = 0;
	double t_searchStart = std::clock();

	std::set<std::pair<float, int> > myset;
	std::set<std::pair<float, int> >::iterator it;

	std::pair<float, int> mypair(c2g[initial_idx], initial_idx);
	myset.insert(mypair);

	for (int i = 0; i < NUM; ++i) {
		// select node to visit
		double t_selStart = std::clock();
		// int cur = -1;
		// for (int j = 0; j < NUM; ++j) {
		// 	if (visit[j]) continue;
		// 	if (cur == -1 || costs[j]+c2g[j] < costs[cur]+c2g[cur]) {
		// 		cur = j;
		// 	}
		// }
		it = myset.begin();
		int cur = it->second;
		myset.erase(it);
		t_sel += (std::clock() - t_selStart) / (double) CLOCKS_PER_SEC;
		if (cur < 0) break; // none left to visit, i.e. failure
		if (cur == goal_idx) break; // we have a solution

		// std::cout << "node idx, cur = " << cur << " cost = " << costs[cur] << ", ";

		double t_connsStart = std::clock();

		visit[cur] = true;
		for (int j = 0; j < nnSize; ++j) {
			int nodeIdx = nnGoEdges[cur*nnSize + j];
			if (nodeIdx < 0 || nodeIdx > NUM) break; // out of neighbors (or overreaching for some bug reason)
			if (visit[nodeIdx]) continue; // this node is already set
			if (isFreeEdges[nnIdxs[NUM*nodeIdx + cur]]) { // there is an edge and it is free
				float pathCost = costs[cur] + adjCosts[cur*NUM + nodeIdx];
				if (pathCost < costs[nodeIdx]) {
					// reset entry in set for visiting
					std::pair<float, int> mypair(costs[nodeIdx] + c2g[nodeIdx], nodeIdx);
					it = myset.find(mypair);
					myset.erase(mypair);
					std::pair<float, int> mypairNew(pathCost + c2g[nodeIdx], nodeIdx);
					myset.insert(mypairNew);

					costs[nodeIdx] = pathCost;
					pathIdx[nodeIdx] = cur;
				}
			}
		}
		t_conns += (std::clock() - t_connsStart) / (double) CLOCKS_PER_SEC;

	}
	t_search += (std::clock() - t_searchStart) / (double) CLOCKS_PER_SEC;
	double t_total = (std::clock() - t_totalStart) / (double) CLOCKS_PER_SEC;

	std::cout << "timing, selection = " << t_sel << ", connection = " << t_conns << ", and search = " << t_search << ", mem = " << t_memTrans << ", total = " << t_total <<std::endl;

	// std::cout << std::endl << "PRM Costs: " << std::endl;
	// for (std::vector<float>::const_iterator i = costs.begin(); i != costs.end(); ++i)
    //    	std::cout << *i << ' ';
    //    std::cout << std::endl;

	// std::cout << std::endl << "PRM Tree: " << std::endl;
	// for (std::vector<int>::const_iterator i = pathIdx.begin(); i != pathIdx.end(); ++i)
 //    	std::cout << *i << ' ';
 //    std::cout << std::endl;

	std::cout << "PRM final cost was " << costs[goal_idx] << std::endl;
}



