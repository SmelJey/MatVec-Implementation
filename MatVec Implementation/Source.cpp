#define CATCH_CONFIG_MAIN
#pragma warning(push, 0)   
#include "catch.hpp"
#pragma warning(pop)

#include "Base.h"
#include "Matrix.h"
#include "Vector.h"
#include <iostream>

using namespace mat_vec;
using namespace std;

TEST_CASE("Matrix eye test", "[Matrix]") {
	auto testMatrix = Matrix::eye(3);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			if (i == j)
				CHECK(testMatrix(i, j) == 1);
			else
				CHECK(testMatrix(i, j) == 0);
}

TEST_CASE("Matrix copy test", "[Matrix]") {
	Matrix testMatrix1(4, 5, 2.0);
	Matrix testMatrix2(1, 1, -1.0);
	testMatrix2 = testMatrix1;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 5; j++)
			CHECK(testMatrix1(i, j) == testMatrix2(i, j));
}

TEST_CASE("Matrix reshape & shape test", "[Matrix]") {
	Matrix testMatrix(3, 3, 3.4);
	Matrix copyMatrix(testMatrix);
	copyMatrix.reshape(2, 4);

	pair<size_t, size_t> sizes1 = testMatrix.shape();
	pair<size_t, size_t> sizes2 = copyMatrix.shape();
	CHECK(sizes1.first == 3);
	CHECK(sizes1.second == 3);
	CHECK(sizes2.first == 2);
	CHECK(sizes2.second == 4);

	size_t it1 = -1; size_t it2 = 0;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 4; j++, it2 = (++it2) % 3) {
			if (it2 == 0)
				it1++;
			CHECK(testMatrix(it1, it2) == copyMatrix(i, j));
		}
}

TEST_CASE("Matrix get / index get", "[Matrix]") {
	Matrix testMatrix(2, 4, -1);
	const Matrix constMatrix(3, 5.0);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 4; j++) {
			CHECK(testMatrix.get(i, j) == testMatrix(i, j));
			CHECK(testMatrix(i, j) == -1);
		}

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) {
			CHECK(constMatrix.get(i, j) == constMatrix(i, j));
			CHECK(constMatrix(i, j) == 5);
		}

	CHECK_THROWS(testMatrix.get(10, 10));
	CHECK_THROWS(testMatrix(10, 10));
	CHECK_THROWS(constMatrix(10, 10));
}

TEST_CASE("Matrix sum", "[Matrix]") {
	Matrix testMatrix1(3, 4, 2);
	Matrix testMatrix2(3, 4, 3);
	Matrix testMatrix3(5, -1.0);

	CHECK_THROWS(testMatrix3 + testMatrix1);

	auto sumMatrix = testMatrix1 + testMatrix2;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			CHECK(sumMatrix(i, j) == 5);
}

TEST_CASE("Matrix subtraction", "[Matrix]") {
	Matrix testMatrix1(3, 4, 2);
	Matrix testMatrix2(3, 4, 3);
	Matrix testMatrix3(5, -1.0);

	CHECK_THROWS(testMatrix3 - testMatrix1);

	auto sumMatrix = testMatrix1 - testMatrix2;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			CHECK(sumMatrix(i, j) == -1);
}

TEST_CASE("Matrix multiplication", "[Matrix]") {
	Matrix testMatrix1(3, 4, 2);
	Matrix testMatrix2(4, 2, 4);
	
	CHECK_THROWS(testMatrix2 *= testMatrix1);

	Matrix prodMatrix = testMatrix1 * testMatrix2;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 2; j++)
			CHECK(prodMatrix(i, j) == 32);
}

TEST_CASE("Matrix and scalar multiplication", "[Matrix]") {
	Matrix testMatrix(3, 4, 5);
	int k = -3;
	Matrix prodMatrix = testMatrix * k;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			CHECK(prodMatrix(i, j) == -15);
}

TEST_CASE("Matrix and scalar division", "[Matrix]") {
	Matrix testMatrix(3, 4, 45);
	int k = -3;
	Matrix divMatrix = testMatrix / k;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			REQUIRE(divMatrix(i, j) == -15);
}

TEST_CASE("Matrix transpose", "[Matrix]") {
	Matrix testMatrix(2, 5, -1);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 5; j++)
			testMatrix(i, j) = 5 * (double)i + j;
	Matrix transposedMatrix = testMatrix.transposed();

	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 2; j++)
			CHECK(transposedMatrix(i, j) == testMatrix(j, i));
}

TEST_CASE("Determinant calculation", "[Matrix]") {
	Matrix testMatrix1(3, 4, 3);
	CHECK_THROWS(testMatrix1.det());
	
	Matrix testMatrix2(3, 3.0);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			testMatrix2(i, j) = (double)i * 3 + j + 1;
	testMatrix2(2, 2) = 0;
	REQUIRE(abs(testMatrix2.det() - 27) < numeric_limits<float>::epsilon());
}

TEST_CASE("Inverted matrix", "[Matrix]") {
	Matrix testMatrix1(3, 4, 3);
	CHECK_THROWS(testMatrix1.inv());
	CHECK_THROWS(Matrix(4, 0.0).inv());
	CHECK_THROWS(Matrix(0, 4.0).inv());

	Matrix testMatrix2(3, 3.0);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			testMatrix2(i, j) =	(double)i * 3 + j + 1;
	testMatrix2(2, 2) = 0;
	Matrix invMatrix = testMatrix2.inv();
	double invArr[5][5] = { { -48.0 / 27, 24.0 / 27, -3.0 / 27},
							{ 42.0 / 27, -21.0 / 27, 6.0 / 27},
							{ -3.0 / 27, 6.0 / 27, -3.0 / 27} };

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			CHECK(abs(invMatrix(i, j) - invArr[i][j]) < numeric_limits<float>::epsilon());
}

TEST_CASE("Matrix-Vector multiplication", "[Vector][Matrix]") {
	Matrix testMatrix(3, 4, -1);
	Vector testVector1(4, 5);
	Vector testVector2(3, 5);

	CHECK_THROWS(testVector1 * testMatrix);

	auto prodVector1 = testMatrix * testVector1;
	REQUIRE(prodVector1.size() == 3);
	for (int i = 0; i < prodVector1.size(); i++)
		CHECK(prodVector1[i] == -20);

	auto prodVector2 = testVector2 * testMatrix;
	REQUIRE(prodVector2.size() == 4);
	for (int i = 0; i < prodVector2.size(); i++)
		REQUIRE(prodVector2[i] == -15);
}

TEST_CASE("Matrix equals", "[Matrix]") {
	Matrix testMatrix1(3, 4.0);
	Matrix testMatrix2(4, 4.0);
	Matrix testMatrix3(3, 4.0);
	Matrix copyMatrix(testMatrix1);
	testMatrix3(2, 2) = -4.0;

	CHECK(testMatrix1 != testMatrix2);
	CHECK(!(testMatrix1 == testMatrix3));
	REQUIRE(copyMatrix == testMatrix1);
}

TEST_CASE("Matrix print", "[Matrix]") {
	Matrix testMatrix(3, 4, 5.0);
	CHECK_NOTHROW(testMatrix.print());
}

TEST_CASE("Vector assignment", "[Vector]") {
	Vector testVector1(3, -4.0);
	Vector testVector2(10, 3);

	testVector2 = testVector1;
	REQUIRE(testVector1.size() == testVector2.size());

	for (int i = 0; i < testVector1.size(); i++)
		CHECK(testVector1[i] == testVector2[i]);
}

TEST_CASE("Vector index", "[Vector]") {
	Vector testVector(5, -1.0);
	CHECK(testVector[3] == -1.0);
	CHECK_THROWS(testVector[5]);

	const Vector constVector(5, -2.0);
	CHECK(constVector[3] == -2.0);
	CHECK_THROWS(constVector[5]);
}

TEST_CASE("Vector norm", "[Vector]") {
	Vector testVector(4, 4.0);
	CHECK(abs(testVector.norm() - 8) < std::numeric_limits<double>::epsilon());
}

TEST_CASE("Vector normalization", "[Vector]") {
	Vector zeroVector(3, 0);
	CHECK_THROWS(zeroVector.normalized());

	Vector testVector(2, 3);
	testVector[1] = 4;
	Vector normVector = testVector.normalized();

	CHECK(abs(normVector[0] - 0.6) < numeric_limits<double>::epsilon());
	CHECK(abs(normVector[1] - 0.8) < numeric_limits<double>::epsilon());
}

TEST_CASE("Vector sum", "[Vector]") {
	Vector testVector1(5, 2.0);
	Vector testVector2(5, 3.0);
	Vector testVector3(3, 3.0);

	CHECK_THROWS(testVector3 + testVector2);

	Vector sumVector = testVector1 + testVector2;
	for (int i = 0; i < 5; i++)
		CHECK(abs(sumVector[i] - 5.0) < std::numeric_limits<double>::epsilon());
}

TEST_CASE("Vector subtraction", "[Vector]") {
	Vector testVector1(5, 2.0);
	Vector testVector2(5, 3.0);
	Vector testVector3(3, 3.0);

	CHECK_THROWS(testVector3 - testVector2);

	Vector subtractVector = testVector1 - testVector2;
	for (int i = 0; i < 5; i++)
		CHECK(abs(subtractVector[i] - (-1.0)) < std::numeric_limits<double>::epsilon());
}

TEST_CASE("Vector multiplication", "[Vector]") {
	Vector testVector1(5, 2.0);
	Vector testVector2(5, 3.0);
	Vector testVector3(3, 3.0);

	CHECK_THROWS(testVector3 ^ testVector2);

	Vector prodVector = testVector1 ^ testVector2;
	for (int i = 0; i < 5; i++)
		CHECK(abs(prodVector[i] - 6) < std::numeric_limits<double>::epsilon());
}

TEST_CASE("Vector dot product", "[Vector]") {
	Vector testVector1(4, 3.0);
	Vector testVector2(4, 4.0);
	Vector testVector3(2, 3.0);

	CHECK_THROWS(testVector1 * testVector3);

	CHECK(abs(testVector1 * testVector2 - 48) < std::numeric_limits<double>::epsilon());
}

TEST_CASE("Vector & scalar product, equals", "[Vector]") {
	Vector testVector1(5, 4.0);
	double k = -3.0;

	Vector prodVector1 = testVector1 * k;
	Vector prodVector2 = k * testVector1;
	
	for (int i = 0; i < 5; i++)
		CHECK(abs(prodVector1[i] - (-12.0)) < numeric_limits<double>::epsilon());
	for (int i = 0; i < 5; i++)
		CHECK(abs(prodVector2[i] - (-12.0)) < numeric_limits<double>::epsilon());

	Vector testVector2(3, -1.0);

	CHECK(prodVector1 == prodVector2);
	CHECK(testVector2 != prodVector1);
	CHECK(prodVector1 != testVector1);
}

TEST_CASE("Vector and scalar division", "[Vector]") {
	Vector testVector(4, 10.0);
	double k = 2;

	Vector divVector = testVector / k;
	for (int i = 0; i < divVector.size(); i++)
		CHECK(abs(divVector[i] - 5.0) < numeric_limits<double>::epsilon());
}

TEST_CASE("Vector print test", "[Vector]") {
	Vector testVector(10, 3.0);
	CHECK_NOTHROW(testVector.print());
}
