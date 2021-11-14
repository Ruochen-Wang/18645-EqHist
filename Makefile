
CC = g++
CFLAGS = -mavx2 -g3

compile:
	$(CC) $(CFLAGS) eq_hist_opt.cpp -o eq_hist 
