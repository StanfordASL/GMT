/*
Author: brian ichter
Runs the Fast marching tree algorithm.
*/
#include "FMT.cuh"


void FMT(float *initial, float *goal, float *obstacles, int obstaclesCount,
	std::vector<float> adjCosts, std::vector<int> nnGoEdges, std::vector<int> nnComeEdges, 
	int nnSize, std::vector<float> discMotions, std::vector<int> nnIdxs,
	int samplesCount, bool *d_isFreeSamples, float r, int numDisc, int numEdges,
	std::vector<float> costs, int initial_idx, int goal_idx,
	std::vector<float> samplesAll, std::vector<float> times)
{
	// instantiate memory for online comp
	double t_totalStart = std::clock();

	bool isFreeSamples[NUM];
	cudaMemcpy(isFreeSamples, d_isFreeSamples, sizeof(bool)*NUM, cudaMemcpyDeviceToHost);

	// timing 	
	double t_sel = 0;
	double t_search = 0;
	double t_conns = 0;
	double t_cc = 0;
	double t_oneStep = 0;
	double t_searchStart = std::clock();

	// initialCondition
	std::vector<bool> closed(NUM,false); // tracks if a node has been visited
	std::vector<bool> open(NUM,false); // tracks if a node is open
	std::vector<bool> unvisited(NUM,true); // tracks if a node has been visited
	std::set<std::pair<float, int> > openSet;
	std::set<std::pair<float, int> >::iterator it;
	std::vector<int> pathIdx(NUM,-1);

	for (int i = 0; i < NUM; ++i) { // remove visited nodes
		if(!isFreeSamples[i]) {
			closed[i] = true;
			unvisited[i] = false;
		}
	}
	
	costs[initial_idx] = 0;
	open[initial_idx] = true;
	unvisited[initial_idx] = false;
	
	std::pair<float, int> nodePair(0, initial_idx);
	openSet.insert(nodePair);

	for (int i = 0; i < NUM; ++i) {
		// select node to closed
		double t_selStart = std::clock();
		if (openSet.empty()) break; // open set is empty, i.e. failure

		// find min node to expand from
		it = openSet.begin();
		int cur = it->second;
		float curCost = it->first;
		openSet.erase(it);
		t_sel += (std::clock() - t_selStart) / (double) CLOCKS_PER_SEC;

		// std::cout << "selecting (" << cur << ", " << curCost << ")" << std::endl;
		if (cur < 0) break; // none left to closed, i.e. failure
		if (cur == goal_idx) break; // we have a solution

		double t_connsStart = std::clock();

		for (int j = 0; j < nnSize; ++j) { // select node to expand to
			int nextIdx = nnGoEdges[cur*nnSize + j];

			if (nextIdx < 0 || nextIdx > NUM) break; // out of neighbors or overreach bug
			if (closed[nextIdx]) continue; // this node has been closed
			if (!unvisited[nextIdx]) continue; // this node has been set
			
			double t_oneStepStart = std::clock();
			// determine optimal one step connection
			int oneStepIdx = -1;
			int oneStepCost = 10000;
			for (int k = 0; k < nnSize; ++k) { 
				int prevIdx = nnComeEdges[nextIdx*nnSize + k];
				if (prevIdx < 0 || prevIdx > NUM) break; // out of neighbors or overreach bug

				if (!open[prevIdx]) continue; // node to connect to has been closed or unvisited

				if (costs[prevIdx] + adjCosts[prevIdx*NUM + nextIdx] < oneStepCost) {
					oneStepCost = costs[prevIdx] + adjCosts[prevIdx*NUM + nextIdx];
					oneStepIdx = prevIdx;
				}
			}
			t_oneStep += (std::clock() - t_oneStepStart) / (double) CLOCKS_PER_SEC;
			// std::cout << "connecting (" << oneStepIdx << ", " << oneStepCost << ")" << std::endl;
			if (oneStepIdx == -1) continue; // nothing to connect to

			// creating the path online as the memory access was slower
			std::vector<float> discMotion(DIM*(numDisc+1));
			findDiscretizedPath(discMotion.data(), &(samplesAll[oneStepIdx*DIM]), &(samplesAll[nextIdx*DIM]), numDisc);

			double t_ccStart = std::clock();
			int edgeIdx = nnIdxs[NUM*nextIdx + oneStepIdx];
			if (isFreeEdge_h(0, obstacles, obstaclesCount, numDisc, discMotion, NULL)) {
				float pathCost = costs[oneStepIdx] + adjCosts[oneStepIdx*NUM + nextIdx];

				// place in open set
				open[nextIdx] = true; 
				unvisited[nextIdx] = false; // has been visited
				std::pair<float, int> nodePairNew(pathCost, nextIdx);
				openSet.insert(nodePairNew);

				costs[nextIdx] = pathCost;
				pathIdx[nextIdx] = oneStepIdx;
			}
			t_cc += (std::clock() - t_ccStart) / (double) CLOCKS_PER_SEC;
		}
		closed[cur] = true; // place in closed set
		open[cur] = false; // remove from open set

		t_conns += (std::clock() - t_connsStart) / (double) CLOCKS_PER_SEC;
		// in loop timing
		// std::cout << "FMT timing, selection = " << t_sel << ", and search = " << t_search << ", cc = " << t_cc << ", onestep = " << t_oneStep <<std::endl;
	}
	t_search += (std::clock() - t_searchStart) / (double) CLOCKS_PER_SEC;
	double t_total = (std::clock() - t_totalStart) / (double) CLOCKS_PER_SEC;

	// std::cout << std::endl << "FMT Costs: " << std::endl;
	// for (std::vector<float>::const_iterator i = costs.begin(); i != costs.end(); ++i)
 //       	std::cout << *i << ' ';
 //       std::cout << std::endl;

	// std::cout << std::endl << "FMT Tree: " << std::endl;
	// for (std::vector<int>::const_iterator i = pathIdx.begin(); i != pathIdx.end(); ++i)
	//    	std::cout << *i << ' ';
	// std::cout << std::endl;

	std::cout << "FMT timing, selection = " << t_sel << ", connection = " << t_conns << ", and search = " << t_search << ", cc = " << t_cc << ", onestep = " << t_oneStep << ", total = " << t_total <<std::endl;

	std::cout << "FMT final cost was " << costs[goal_idx] << std::endl;
}



void FMTdub(float *initial, float *goal, float *obstacles, int obstaclesCount,
	std::vector<float> adjCosts, std::vector<int> nnGoEdges, std::vector<int> nnComeEdges, 
	int nnSize, std::vector<float> discMotions, std::vector<int> nnIdxs,
	int samplesCount, bool *d_isFreeSamples, float r, int numDisc, int numEdges,
	std::vector<float> costs, int initial_idx, int goal_idx,
	std::vector<float> samplesAll, std::vector<float> times,
	int numControls, std::vector<int> controlEdge, std::vector<float> controlDEdge)
{
	// instantiate memory for online comp
	double t_totalStart = std::clock();

	bool isFreeSamples[NUM];
	cudaMemcpy(isFreeSamples, d_isFreeSamples, sizeof(bool)*NUM, cudaMemcpyDeviceToHost);

	// timing 	
	double t_sel = 0;
	double t_search = 0;
	double t_conns = 0;
	double t_cc = 0;
	double t_oneStep = 0;
	double t_searchStart = std::clock();

	// initialCondition
	std::vector<bool> closed(NUM,false); // tracks if a node has been visited
	std::vector<bool> open(NUM,false); // tracks if a node is open
	std::vector<bool> unvisited(NUM,true); // tracks if a node has been visited
	std::set<std::pair<float, int> > openSet;
	std::set<std::pair<float, int> >::iterator it;
	std::vector<int> pathIdx(NUM,-1);

	for (int i = 0; i < NUM; ++i) { // remove visited nodes
		if(!isFreeSamples[i]) {
			closed[i] = true;
			unvisited[i] = false;
		}
	}
	
	costs[initial_idx] = 0;
	open[initial_idx] = true;
	unvisited[initial_idx] = false;
	
	std::pair<float, int> nodePair(0, initial_idx);
	openSet.insert(nodePair);

	for (int i = 0; i < NUM; ++i) {
		// select node to closed
		double t_selStart = std::clock();
		if (openSet.empty()) break; // open set is empty, i.e. failure

		// find min node to expand from
		it = openSet.begin();
		int cur = it->second;
		float curCost = it->first;
		openSet.erase(it);
		t_sel += (std::clock() - t_selStart) / (double) CLOCKS_PER_SEC;

		// std::cout << "selecting (" << cur << ", " << curCost << ")" << std::endl;
		if (cur < 0) break; // none left to closed, i.e. failure
		if (cur == goal_idx) break; // we have a solution

		double t_connsStart = std::clock();

		for (int j = 0; j < nnSize; ++j) { // select node to expand to
			int nextIdx = nnGoEdges[cur*nnSize + j];

			if (nextIdx < 0 || nextIdx > NUM) break; // out of neighbors or overreach bug
			if (closed[nextIdx]) continue; // this node has been closed
			if (!unvisited[nextIdx]) continue; // this node has been set
			
			double t_oneStepStart = std::clock();
			// determine optimal one step connection
			int oneStepIdx = -1;
			int oneStepCost = 10000;
			for (int k = 0; k < nnSize; ++k) { 
				int prevIdx = nnComeEdges[nextIdx*nnSize + k];
				if (prevIdx < 0 || prevIdx > NUM) break; // out of neighbors or overreach bug

				if (!open[prevIdx]) continue; // node to connect to has been closed or unvisited

				if (costs[prevIdx] + adjCosts[prevIdx*NUM + nextIdx] < oneStepCost) {
					oneStepCost = costs[prevIdx] + adjCosts[prevIdx*NUM + nextIdx];
					oneStepIdx = prevIdx;
				}
			}
			t_oneStep += (std::clock() - t_oneStepStart) / (double) CLOCKS_PER_SEC;
			// std::cout << "connecting (" << oneStepIdx << ", " << oneStepCost << ")" << std::endl;
			if (oneStepIdx == -1) continue; // nothing to connect to
			int edgeIdx = nnIdxs[NUM*nextIdx + oneStepIdx];

			std::vector<float> discMotion(DIM*(numDisc*numControls));

			// copy in discretized motion			
			// for (int i = 0; i < DIM*numDisc*numControls; ++i)
				// discMotion[i] = discMotions[edgeIdx*numControls*numDisc*DIM + i];

			// compute discretized motion from scratch
	 		// std::vector<int> control(numControls);
	 		// std::vector<float> controlD(numControls);
			// float cmin = dubinsAirplaneCost(&(samplesAll[oneStepIdx*DIM]), &(samplesAll[nextIdx*DIM]), control.data(), controlD.data());
			// dubinsAirplanePath(&(samplesAll[oneStepIdx*DIM]), &(samplesAll[nextIdx*DIM]), 
			// control.data(), controlD.data(), 
			// discMotion.data(), numDisc);

 			// compute discretized motion with controls set
 			// computing online as the memory access was slower
 			dubinsAirplanePath(&(samplesAll[oneStepIdx*DIM]), &(samplesAll[nextIdx*DIM]), 
				&(controlEdge[edgeIdx*numControls]), &(controlDEdge[edgeIdx*numControls]),
				discMotion.data(), numDisc);

			double t_ccStart = std::clock();
			if (isFreeEdge_h(0, obstacles, obstaclesCount, numDisc*numControls-1, 
				discMotion, NULL)) {
				float pathCost = costs[oneStepIdx] + adjCosts[oneStepIdx*NUM + nextIdx];

				// place in open set
				open[nextIdx] = true; 
				unvisited[nextIdx] = false; // has been visited
				std::pair<float, int> nodePairNew(pathCost, nextIdx);
				openSet.insert(nodePairNew);

				costs[nextIdx] = pathCost;
				pathIdx[nextIdx] = oneStepIdx;
			}
			t_cc += (std::clock() - t_ccStart) / (double) CLOCKS_PER_SEC;
		}
		closed[cur] = true; // place in closed set
		open[cur] = false; // remove from open set

		t_conns += (std::clock() - t_connsStart) / (double) CLOCKS_PER_SEC;
		// in loop timing
		// std::cout << "FMT timing, selection = " << t_sel << ", and search = " << t_search << ", cc = " << t_cc << ", onestep = " << t_oneStep <<std::endl;
	}
	t_search += (std::clock() - t_searchStart) / (double) CLOCKS_PER_SEC;
	double t_total = (std::clock() - t_totalStart) / (double) CLOCKS_PER_SEC;

	// std::cout << std::endl << "FMT Costs: " << std::endl;
	// for (std::vector<float>::const_iterator i = costs.begin(); i != costs.end(); ++i)
 //       	std::cout << *i << ' ';
 //       std::cout << std::endl;

	// std::cout << std::endl << "FMT Tree: " << std::endl;
	// for (std::vector<int>::const_iterator i = pathIdx.begin(); i != pathIdx.end(); ++i)
	//    	std::cout << *i << ' ';
	// std::cout << std::endl;

	std::cout << "FMT timing, selection = " << t_sel << ", connection = " << t_conns << ", and search = " << t_search << ", cc = " << t_cc << ", onestep = " << t_oneStep << ", total = " << t_total <<std::endl;

	std::cout << "FMT final cost was " << costs[goal_idx] << std::endl;
}



	