DIM ?= 6
NUM ?= 1000

CXX=nvcc
CXXFLAGS=-O3 -arch=sm_61 -rdc=true -D DIM=$(DIM) -D NUM=$(NUM)
# compute capability is sm_30 for macbook, sm_52 for phobos, sm_53 for tx1, sm_61 for 1080

SRC=*.cu

EXPERIMENT=mainExperiment.cu motionPlanningProblem.cu motionPlanningSolution.cu collisionCheck.cu obstacles.cu helper.cu 2pBVP.cu sampler.cu GMT.cu
DUBINS=mainDubins.cu motionPlanningProblem.cu motionPlanningSolution.cu collisionCheck.cu obstacles.cu helper.cu 2pBVP.cu sampler.cu GMT.cu PRM.cu FMT.cu dubinsAirplane.cu
GROUND=mainGround.cu motionPlanningProblem.cu motionPlanningSolution.cu collisionCheck.cu obstacles.cu helper.cu 2pBVP.cu sampler.cu GMT.cu PRM.cu FMT.cu dubinsAirplane.cu
SIM=mainSim.cu motionPlanningProblem.cu motionPlanningSolution.cu collisionCheck.cu obstacles.cu helper.cu 2pBVP.cu sampler.cu GMT.cu PRM.cu FMT.cu dynamics.cu

main: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ 

experiment: $(EXPERIMENT)
	$(CXX) $(CXXFLAGS) $(EXPERIMENT) -o $@ 

dubins: $(DUBINS)
	$(CXX) $(CXXFLAGS) $(DUBINS) -o $@ 

ground: $(GROUND)
	$(CXX) $(CXXFLAGS) $(GROUND) -o $@ 

sim: $(SIM)
	$(CXX) $(CXXFLAGS) $(SIM) -o $@ 

clean:
	rm -f *.o *~ *~ main ground sim dubins experiment
	rm -rf *.dSYM
