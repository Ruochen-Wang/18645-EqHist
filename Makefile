
CC=g++
CFLAGS=-mavx2

compile:
	$(CC) $(CFLAGS) eq_hist_opt.cpp -o eq_hist 
