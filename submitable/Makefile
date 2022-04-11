all:
	gcc -fopenmp -O3 -msse4 -o conv conv-harness.c

test:
	gcc -fopenmp -O3 -msse4 -o test conv-test.c
	gcc -fopenmp -O3 -msse4 -o conv conv-harness.c
  
clean:
	rm conv test
