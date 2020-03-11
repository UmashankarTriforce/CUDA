#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "my_sort.cuh"

void printArray(int* arr, long long int n, int offset = 0) {
	for (long long int i = offset; i < n; ++i) {
		std::cout << arr[i] << std::endl;
	}
}

void initArray(int* arr, long long int n) {
	srand(time(0));
	for (long long int i = 0; i < n; ++i) {
		arr[i] = rand();
	}
}

void checkArray(int* arr, long long int n) {
	int flag = 0;

	for (long long int i = 0; i < n - 1; ++i) {
		if (arr[i] < arr[i + 1]) {
			std::cout << "Not sorted Properly"<<std::endl;
			flag = i;
			break;
		}
	}

	if (flag) {
		std::cout << flag << std::endl;
		printArray(arr, n, flag);
	}
}

int main(int argc, char **argv) {

	long long int n = atoi(argv[1]);
	int* arr;

	arr = new int[n];
	initArray(arr, n);
	sort(arr, n);
	checkArray(arr, n);

	return 0;
}