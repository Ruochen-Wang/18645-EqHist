
CC = g++
MPIC = mpic++

CFLAGS = -mavx2 -g3 -fopenmp -O2

compile:
	$(CC) $(CFLAGS) eq_hist_opt.cpp -o eq_hist
	$(MPIC) -g3 -mavx2 eq_hist_mpi.cpp -o mpi_eq_hist

run_mpi:
	mpiexec -n 2 ./mpi_eq_hist
