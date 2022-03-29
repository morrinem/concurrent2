all:
	gcc -fopenmp -O3 -msse4 -o conv conv-harness.c

clean:
	rm conv
