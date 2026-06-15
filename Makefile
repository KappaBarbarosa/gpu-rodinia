include common/make.config

RODINIA_BASE_DIR := $(shell pwd)

CUDA_BIN_DIR := $(RODINIA_BASE_DIR)/bin/linux/cuda
OMP_BIN_DIR := $(RODINIA_BASE_DIR)/bin/linux/omp

CUDA_DIRS := backprop bfs hotspot kmeans lavaMD lud nn nw particlefilter srad
OMP_DIRS  := backprop bfs hotspot kmeans lavaMD lud nn nw particlefilter srad

all: CUDA OMP

CUDA:
	cd cuda/backprop;		make;	cp backprop $(CUDA_BIN_DIR)
	cd cuda/bfs;			make;	cp bfs $(CUDA_BIN_DIR)
	cd cuda/hotspot;		make;	cp hotspot $(CUDA_BIN_DIR)
	cd cuda/kmeans;			make;	cp kmeans $(CUDA_BIN_DIR)
	cd cuda/lavaMD;			make;	cp lavaMD $(CUDA_BIN_DIR)
	cd cuda/lud;			make;	cp cuda/lud_cuda $(CUDA_BIN_DIR)
	cd cuda/nn;				make;	cp nn $(CUDA_BIN_DIR)
	cd cuda/nw;			make;	cp needle $(CUDA_BIN_DIR)
	cd cuda/particlefilter;		make;	cp particlefilter_naive particlefilter_float $(CUDA_BIN_DIR)
	cd cuda/srad/srad_v1;		make;	cp srad $(CUDA_BIN_DIR)/srad_v1
	cd cuda/srad/srad_v2;		make;   cp srad $(CUDA_BIN_DIR)/srad_v2


OMP:
	cd openmp/backprop;				make;	cp backprop $(OMP_BIN_DIR)
	cd openmp/bfs;					make;	cp bfs $(OMP_BIN_DIR)
	cd openmp/hotspot;				make;	cp hotspot $(OMP_BIN_DIR)
	cd openmp/kmeans/kmeans_openmp;			make;	cp kmeans $(OMP_BIN_DIR)
	cd openmp/lavaMD;				make;	cp lavaMD $(OMP_BIN_DIR)
	cd openmp/lud;					make;	cp omp/lud_omp $(OMP_BIN_DIR)
	cd openmp/nn;					make;	cp nn $(OMP_BIN_DIR)
	cd openmp/nw;					make;	cp needle $(OMP_BIN_DIR)
	cd openmp/particlefilter;			make;	cp particle_filter $(OMP_BIN_DIR)
	cd openmp/srad/srad_v1;				make;	cp srad $(OMP_BIN_DIR)/srad_v1
	cd openmp/srad/srad_v2;				make;   cp srad $(OMP_BIN_DIR)/srad_v2

clean: CUDA_clean OMP_clean

CUDA_clean:
	cd $(CUDA_BIN_DIR); rm -f *
	for dir in $(CUDA_DIRS) ; do cd cuda/$$dir ; make clean ; cd ../.. ; done

OMP_clean:
	cd $(OMP_BIN_DIR); rm -f *
	for dir in $(OMP_DIRS) ; do cd openmp/$$dir ; make clean ; cd ../.. ; done
