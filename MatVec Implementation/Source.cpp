#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "Base.h"
#include "Matrix.h"
#include "Vector.h"
#include "SparseMatrix.h"
#include <iostream>

using namespace mat_vec;
using namespace std;

TEST_CASE("Matrix eye test", "[Matrix]") {
	auto testMatrix = Matrix::eye(3);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			if (i == j)
				CHECK(abs(testMatrix(i, j) - 1) < numeric_limits<float>::epsilon());
			else
				CHECK(abs(testMatrix(i, j)) < numeric_limits<float>::epsilon());
}

TEST_CASE("Matrix copy test", "[Matrix]") {
	Matrix testMatrix1(4, 5, 2.0);
	Matrix testMatrix2(1, 1, -1.0);
	testMatrix2 = testMatrix1;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 5; j++)
			CHECK(abs(testMatrix1(i, j) - testMatrix2(i, j)) < numeric_limits<float>::epsilon());
}

TEST_CASE("Matrix reshape & shape test", "[Matrix]") {
	Matrix testMatrix(3, 4, 3.4);
	Matrix copyMatrix(testMatrix);
    CHECK_THROWS(copyMatrix.reshape(2, 5));
	copyMatrix.reshape(2, 6);

	pair<size_t, size_t> sizes1 = testMatrix.shape();
	pair<size_t, size_t> sizes2 = copyMatrix.shape();

	CHECK(sizes1.first == 3);
	CHECK(sizes1.second == 4);
	CHECK(sizes2.first == 2);
	CHECK(sizes2.second == 6);

	size_t it1 = -1; size_t it2 = 0;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 5; j++, it2 = (++it2) % 4) {
			if (it2 == 0)
				it1++;
			CHECK(abs(testMatrix(it1, it2) - copyMatrix(i, j)) < numeric_limits<float>::epsilon());
		}
}

TEST_CASE("Matrix get / index get", "[Matrix]") {
	Matrix testMatrix(2, 4, -1);
	const Matrix constMatrix(3, 5.0);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 4; j++) {
			CHECK(abs(testMatrix.get(i, j) - testMatrix(i, j)) < numeric_limits<float>::epsilon());
			CHECK(abs(testMatrix(i, j) - -1) < numeric_limits<float>::epsilon());
		}

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) {
			CHECK(abs(constMatrix.get(i, j) - constMatrix(i, j)) < numeric_limits<float>::epsilon());
			CHECK(abs(constMatrix(i, j) - 5) < numeric_limits<float>::epsilon());
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
			CHECK(abs(sumMatrix(i, j) - 5) < numeric_limits<float>::epsilon());
}

TEST_CASE("Matrix subtraction", "[Matrix]") {
	Matrix testMatrix1(3, 4, 2);
	Matrix testMatrix2(3, 4, 3);
	Matrix testMatrix3(5, -1.0);

	CHECK_THROWS(testMatrix3 - testMatrix1);

	auto sumMatrix = testMatrix1 - testMatrix2;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			CHECK(abs(sumMatrix(i, j) - -1) < numeric_limits<float>::epsilon());
}

TEST_CASE("Matrix multiplication", "[Matrix]") {
	Matrix testMatrix1(3, 4, 2);
	Matrix testMatrix2(4, 2, 4);
	
	CHECK_THROWS(testMatrix2 *= testMatrix1);

	Matrix prodMatrix = testMatrix1 * testMatrix2;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 2; j++)
			CHECK(abs(prodMatrix(i, j) - 32) < numeric_limits<float>::epsilon());
}

TEST_CASE("Matrix and scalar multiplication", "[Matrix]") {
	Matrix testMatrix(3, 4, 5);
	int k = -3;
	Matrix prodMatrix = testMatrix * k;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			CHECK(abs(prodMatrix(i, j) - -15) < numeric_limits<float>::epsilon());
}

TEST_CASE("Matrix and scalar division", "[Matrix]") {
	Matrix testMatrix(3, 4, 45);
	int k = -3;
    CHECK_THROWS(testMatrix / 0);
	Matrix divMatrix = testMatrix / k;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			CHECK(abs(divMatrix(i, j) - -15) < numeric_limits<float>::epsilon());
}

TEST_CASE("Matrix transpose", "[Matrix]") {
	Matrix testMatrix(2, 5, -1);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 5; j++)
			testMatrix(i, j) = 5 * (double)i + j;
	Matrix transposedMatrix = testMatrix.transposed();

	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 2; j++)
			CHECK(abs(transposedMatrix(i, j) - testMatrix(j, i)) < numeric_limits<float>::epsilon());
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
	for (size_t i = 0; i < prodVector1.size(); i++)
		CHECK(prodVector1[i] == -20);

	auto prodVector2 = testVector2 * testMatrix;
	REQUIRE(prodVector2.size() == 4);
	for (size_t i = 0; i < prodVector2.size(); i++)
		REQUIRE(abs(prodVector2[i] - -15) < numeric_limits<float>::epsilon());
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

	for (size_t i = 0; i < testVector1.size(); i++)
		CHECK(abs(testVector1[i] - testVector2[i]) < numeric_limits<float>::epsilon());
}

TEST_CASE("Vector index", "[Vector]") {
	Vector testVector(5, -1.0);
	CHECK(abs(testVector[3] - -1.0) < numeric_limits<float>::epsilon());
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

    CHECK_THROWS(testVector / 0);

	Vector divVector = testVector / k;
	for (size_t i = 0; i < divVector.size(); i++)
		CHECK(abs(divVector[i] - 5.0) < numeric_limits<double>::epsilon());
}

TEST_CASE("Vector print test", "[Vector]") {
	Vector testVector(10, 3.0);
	CHECK_NOTHROW(testVector.print());
}

TEST_CASE("Sparse matrix get, set", "[SparseMatrix]") {
    auto matrix = SparseMatrix::eye(5);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (i == j)
                CHECK(abs(matrix.get(i, j) - 1) < numeric_limits<float>::epsilon());
            else
                CHECK(abs(matrix.get(i, j)) < numeric_limits<float>::epsilon());
        }
    }

    CHECK_THROWS(matrix.get(5, 4));
    CHECK_THROWS(matrix.set(0, 5, 4));

    matrix.set(2, 1, 1);
    matrix.set(0, 0, 0);
    matrix.set(0, 1, 0);

    CHECK(abs(matrix.get(1, 1) - 2) < numeric_limits<float>::epsilon());
    CHECK(abs(matrix.get(0, 0)) < numeric_limits<float>::epsilon());
    CHECK(abs(matrix.get(1, 0)) < numeric_limits<float>::epsilon());
}

TEST_CASE("Sparse matrix copy constructor, assign operator and shape", "[SparseMatrix]") {
    auto testMatrix = SparseMatrix::eye(4);
    testMatrix.set(3, 1, 1);
    SparseMatrix copyMatrix(testMatrix);
    SparseMatrix newMatrix(4, 4);
    newMatrix = testMatrix;

    auto size1 = testMatrix.shape();
    auto size2 = copyMatrix.shape();
    CHECK(size1 == size2);
    CHECK(size1.first == 4);
    CHECK(size1.second == 4);

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            CHECK(abs(copyMatrix.get(i, j) - testMatrix.get(i, j)) < numeric_limits<float>::epsilon());
            CHECK(abs(newMatrix.get(i, j) - testMatrix.get(i, j)) < numeric_limits<float>::epsilon());
        }
    }
}

TEST_CASE("Sparse matrix to Matrix and back, density and sparsity", "[SparseMatrix]") {
    Matrix testMatrix1 = Matrix::eye(5);
    Matrix testMatrix2(3, 4, -1.0);

    SparseMatrix spMatrix1(testMatrix1);

    auto size1 = testMatrix1.shape();
    auto size2 = spMatrix1.shape();

    REQUIRE(size1 == size2);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            REQUIRE(abs(spMatrix1.get(i, j) - testMatrix1.get(i, j)) < numeric_limits<float>::epsilon());
        }
    }

    CHECK(abs(spMatrix1.density() - 0.2) < numeric_limits<float>::epsilon());
    CHECK(abs(spMatrix1.sparsity() - 0.8) < numeric_limits<float>::epsilon());
    Matrix copyMatrix1 = spMatrix1.denseMatrix();
    CHECK(copyMatrix1 == testMatrix1);

    SparseMatrix spMatrix2(testMatrix2);

    size1 = testMatrix2.shape();
    size2 = spMatrix2.shape();

    REQUIRE(size1 == size2);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            REQUIRE(abs(spMatrix2.get(i, j) - testMatrix2.get(i, j)) < numeric_limits<float>::epsilon());
        }
    }

    CHECK(abs(spMatrix2.density() - 1) < numeric_limits<float>::epsilon());
    CHECK(abs(spMatrix2.sparsity()) < numeric_limits<float>::epsilon());

    Matrix copyMatrix2 = spMatrix2.denseMatrix();
    CHECK(copyMatrix2 == testMatrix2);
}

TEST_CASE("Sparse matrix sum", "[SparseMatrix]") {
    SparseMatrix testMatrix1(3, 4);
    SparseMatrix testMatrix2(4, 4);
    for (int i = 0; i < 4; i++) {
        testMatrix2.set(3, 0, i);
    }
    testMatrix2.set(1, 3, 3);

    CHECK_THROWS(testMatrix1 + testMatrix2);

    SparseMatrix testMatrix3 = SparseMatrix::eye(4);

    double checkSum[4][4] = { { 4, 3, 3, 3 },
                              { 0, 1, 0, 0 },
                              { 0, 0, 1, 0 },
                              { 0, 0, 0, 2 } };

    auto resMatrix = testMatrix2 + testMatrix3;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            CHECK(abs(resMatrix.get(i, j) - checkSum[i][j]) < numeric_limits<float>::epsilon());
        }
    }
}

TEST_CASE("Sparse matrix subtraction", "[SparseMatrix]") {
    SparseMatrix testMatrix1(3, 4);
    SparseMatrix testMatrix2(4, 4);
    for (int i = 0; i < 4; i++) {
        testMatrix2.set(3, 0, i);
    }
    testMatrix2.set(1, 3, 3);

    CHECK_THROWS(testMatrix1 - testMatrix2);

    SparseMatrix testMatrix3 = SparseMatrix::eye(4);

    double checkSum[4][4] = { { 2, 3, 3, 3 },
                              { 0, -1, 0, 0 },
                              { 0, 0, -1, 0 },
                              { 0, 0, 0, 0 } };

    auto resMatrix = testMatrix2 - testMatrix3;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            CHECK(abs(resMatrix.get(i, j) - checkSum[i][j]) < numeric_limits<float>::epsilon());
        }
    }
}

TEST_CASE("Sparse matrix transpose", "[SparseMatrix]") {
    SparseMatrix test(3, 4);
    test.set(3, 2, 3);
    test.set(-1, 0, 0);
    test.set(-3, 2, 2);
    test.set(-10, 0, 2);
    test.set(4, 1, 0);

    SparseMatrix transposedMatrix = test.transposed();
    std::pair<size_t, size_t> size1 = test.shape();
    std::pair<size_t, size_t> size2 = test.shape();

    REQUIRE(size1 == size2);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            CHECK(abs(test.get(i, j) - transposedMatrix.get(j, i)) < numeric_limits<float>::epsilon());
        }
    }
}

TEST_CASE("Sparse matrix multiplication", "[SparseMatrix]") {
    double sp1Vals[3][4] = { { 1, 0, 4, 5 },
                             { 3, 0, 2, -1 },
                             { 0, 0, 0, 0 } };
    SparseMatrix sp1(3, 4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            sp1.set(sp1Vals[i][j], i, j);
        }
    }

    double sp2Vals[4][5] = { { 10, 0, 0, 0, 6 },
                             { 0, 0, -8, 0, 0 },
                             { 0, 0, 0, 0, 0 },
                             { 0, -7, 0, 0, 5 } };
    SparseMatrix sp2(4, 5);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            sp2.set(sp2Vals[i][j], i, j);
        }
    }

    SparseMatrix sp3(3, 3);
    CHECK_THROWS(sp1 * sp3);

    double checkVals[3][5] = { { 10, -35, 0, 0, 31 },
                               { 30, 7, 0, 0, 13 },
                               { 0, 0, 0, 0, 0 } };
    SparseMatrix res = sp1 * sp2;
    std::pair<size_t, size_t> size = res.shape();
    REQUIRE(size.first == 3);
    REQUIRE(size.second == 5);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            CHECK(abs(res.get(i, j) - checkVals[i][j]) < numeric_limits<float>::epsilon());
        }
    }
}

TEST_CASE("Sparse Matrix scalar division and multiplication", "[SparseMatrix]") {
    SparseMatrix testMatrix(4, 5);
    double vals[4][5] = { { 10, -35, 0, 0, 31 },
                          { 30, 7, 0, 0, 13 },
                          { 0, 0, 0, -7, 0 },
                          { -1.5, 1, -2, 0, 0 } };
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            testMatrix.set(vals[i][j], i, j);
        }
    }

    double k1 = -2.5;

    SparseMatrix test1 = testMatrix * k1;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            CHECK(abs(test1.get(i, j) - vals[i][j] * k1) < std::numeric_limits<float>::epsilon());
        }
    }

    CHECK_THROWS(test1 / 0);

    test1 = test1 / k1;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            CHECK(abs(test1.get(i, j) - vals[i][j]) < std::numeric_limits<float>::epsilon());
        }
    }
}

TEST_CASE("Sparse matrix equality check", "[SparseMatrix]") {
    SparseMatrix testMatrix1(2, 3);
    SparseMatrix testMatrix2(3, 2);

    double vals[2][3] = { { -1, 0, 4 }, 
                          { 0, 0, -3} };
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            testMatrix1.set(vals[i][j], i, j);
            testMatrix2.set(vals[i][j], j, i);
        }
    }

    CHECK(testMatrix1 != testMatrix2);

    SparseMatrix testMatrix3(testMatrix1);
    testMatrix3.set(0, 1, 2);

    CHECK(testMatrix1 != testMatrix3);

    SparseMatrix testMatrix4(testMatrix1);
    testMatrix4.set(5, 1, 2);
    CHECK(testMatrix1 != testMatrix4);

    SparseMatrix copyMatrix(testMatrix1);
    CHECK(copyMatrix == testMatrix1);
}

TEST_CASE("Sparse matrix and vector multiplication", "[SparseMatrix][Vector]") {
    Vector testVector(3, 1);
    testVector[1] = -4;
    testVector[2] = 3;

    SparseMatrix testMatrix(3, 2);
    testMatrix.set(-2, 0, 0);
    testMatrix.set(6, 0, 1);
    testMatrix.set(-1, 1, 1);
    testMatrix.set(5, 2, 0);

    Vector testVector2(4);
    CHECK_THROWS(testVector2 * testMatrix);

    Vector res = testVector * testMatrix;
    REQUIRE(res.size() == 2);
    CHECK(abs(res[0] - 13) < numeric_limits<float>::epsilon());
    CHECK(abs(res[1] - 10) < numeric_limits<float>::epsilon());

    Vector testVector3(2, -1);
    testVector3[1] = 5;

    res = testMatrix * testVector3;
    REQUIRE(res.size() == 3);
    CHECK(abs(res[0] - 32) < numeric_limits<float>::epsilon());
    CHECK(abs(res[1] - -5) < numeric_limits<float>::epsilon());
    CHECK(abs(res[2] - -5) < numeric_limits<float>::epsilon());
}

TEST_CASE("Sparse Matrix print", "[SparseMatrix]") {
    SparseMatrix testMatrix = SparseMatrix::eye(4);
    CHECK_NOTHROW(testMatrix.print());
}

TEST_CASE("Sparse Matrix determinant and inverted matrix") {
    SparseMatrix testMatrix1(3, 4);
    CHECK_THROWS(testMatrix1.det());
    CHECK_THROWS(testMatrix1.inv());

    SparseMatrix testMatrix2 = SparseMatrix::eye(4);
    CHECK(abs(testMatrix2.det() - 1) < std::numeric_limits<float>::epsilon());
    CHECK(testMatrix2.inv() == testMatrix2);

    SparseMatrix testMatrix3(3);
    testMatrix3.set(1, 0, 0);
    testMatrix3.set(-2, 0, 1);
    testMatrix3.set(5, 1, 2);
    testMatrix3.set(-3, 2, 0);
    testMatrix3.set(6, 2, 2);
    CHECK(abs(testMatrix3.det() - 30) < std::numeric_limits<float>::epsilon());
    SparseMatrix invMatrix = testMatrix3.inv();

    double checkVals[3][3] = { { 0, 12.0 / 30, -10.0 / 30 },
                               { -15.0 / 30, 6.0 / 30, -5.0 / 30},
                               { 0, 6.0 / 30, 0} };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(abs(invMatrix.get(i, j) - checkVals[i][j]) < numeric_limits<float>::epsilon());
        }
    }

    SparseMatrix testMatrix4(Matrix(3, -1.0));
    CHECK_THROWS(testMatrix4.inv());
}
