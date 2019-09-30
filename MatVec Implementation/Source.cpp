#include "Base.h"
#include "Matrix.h"
#include "Vector.h"
#include <iostream>

using namespace mat_vec;

int main() {
	Matrix test(3, 10, -2.5);

	Matrix copyTest = Matrix(test);

	Matrix unit = Matrix::eye(4);

	test.print(), copyTest.print(), unit.print();
	
	test.reshape(10, 4);
	test.print();
	std::cout << test.shape().first << " " << test.shape().second;

	return 0;
}