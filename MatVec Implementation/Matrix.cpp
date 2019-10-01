#include "Matrix.h"
#include <iostream>
#include <algorithm>
#include <numeric>

using namespace mat_vec;

Matrix::Matrix(size_t size, double value) : Matrix(size, size, value) {}

Matrix Matrix::eye(size_t size) {
	Matrix unit(size);
	for (int i = 0; i < size; i++) 
		unit.vals[i][i] = 1;
	return unit;
}

Matrix::Matrix(size_t rows, size_t cols, double value) {
	this->rows = rows; this->cols = cols;
	vals = new double* [rows]();
	for (int i = 0; i < rows; i++){
		vals[i] = new double[cols]();
		for (int j = 0; j < cols; j++) 
			vals[i][j] = value;
	}
}

Matrix::Matrix(const Matrix& src) {
	rows = src.rows; cols = src.cols;
	vals = new double* [rows]();
	for (int i = 0; i < rows; i++) {
		vals[i] = new double[cols]();
		for (int j = 0; j < cols; j++)
			vals[i][j] = src.vals[i][j];
	}
}

Matrix& Matrix::operator=(const Matrix& rhs) {
	Matrix tmp(rhs);
	swap(tmp);
	return *this;
}

Matrix::~Matrix() {
	for (int i = 0; i < rows; i++)
		delete[] vals[i];
	delete[] vals;
}

void Matrix::reshape(size_t rows, size_t cols) {
	int i1 = 0, j1 = 0;

	double** newVals = new double*[rows]();
	for (int i = 0; i < rows; i++) {
		newVals[i] = new double[cols]();
		for (int j = 0; i1 < this->rows && j < cols; j++) {
			newVals[i][j] = vals[i1][j1];
			j1++;
			if (j1 >= this->cols) {
				j1 = 0; i1++;
			}
		}
	}

	std::swap(newVals, this->vals);
	for (int i = 0; i < this->rows; i++)
		delete[] newVals[i];
	delete[] newVals;

	this->rows = rows;
	this->cols = cols;
}

std::pair<size_t, size_t> Matrix::shape() const {
	return std::pair<size_t, size_t>(rows, cols);
}

double Matrix::get(size_t row, size_t col) const {
	return this->vals[row][col];
}

Matrix Matrix::operator+(const Matrix& rhs) const {
	Matrix res = *this;
	res += rhs;
	return res;
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
	for (int i = 0; i < std::min(rows, rhs.rows); i++) {
		for (int j = 0; j < std::min(cols, rhs.cols); j++) {
			vals[i][j] += rhs.vals[i][j];
			if (abs(vals[i][j]) < std::numeric_limits<float>::epsilon())
				vals[i][j] = 0;
		}
	}
	return *this;
}

Matrix Matrix::operator-(const Matrix& rhs) const {
	Matrix res = *this;
	res -= rhs;
	return res;
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
	for (int i = 0; i < std::min(rows, rhs.rows); i++) {
		for (int j = 0; j < std::min(cols, rhs.cols); j++) {
			vals[i][j] -= rhs.vals[i][j];
			if (abs(vals[i][j]) < std::numeric_limits<float>::epsilon())
				vals[i][j] = 0;
		}
	}
	return *this;
}

Matrix Matrix::operator*(const Matrix& rhs) const {
	Matrix res = *this;
	res *= rhs;
	return res;
}

Matrix& Matrix::operator*=(const Matrix& rhs) {
	if (this->cols != rhs.rows)
		throw std::exception("Cant multiply these matrices");

	double** res = new double* [rows]();
	for (int i = 0; i < rows; i++)
		res[i] = new double[rhs.cols]();

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < rhs.cols; j++) {
			for (int k = 0; k < cols; k++) {
				res[i][j] += vals[i][k] * rhs.vals[k][j];
			}
			if (abs(res[i][j]) < std::numeric_limits<float>::epsilon())
				res[i][j] = 0;
		}
	}

	std::swap(res, this->vals);
	for (int i = 0; i < rows; i++)
		delete[] res[i];
	delete[] res;

	cols = rhs.cols;
	
	return *this;
}

Matrix Matrix::operator*(double k) const {
	Matrix res = *this;
	res *= k;
	return res;
}

Matrix& Matrix::operator*=(double k) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			vals[i][j] *= k;
			if (abs(vals[i][j]) < std::numeric_limits<float>::epsilon())
				vals[i][j] = 0;
		}
	}
	return *this;
}

Matrix Matrix::operator/(double k) const {
	Matrix res = *this;
	res /= k;
	return res;
}
Matrix& Matrix::operator/=(double k) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			vals[i][j] /= k;
			if (abs(vals[i][j]) < std::numeric_limits<float>::epsilon())
				vals[i][j] = 0;
		}
	}
	return *this;
}

Matrix Matrix::transposed() const {
	Matrix res = *this;
	res.transpose();
	return res;
}

void Matrix::transpose() {
	double** newVals = new double* [cols]();
	for (int i = 0; i < cols; i++)
		newVals[i] = new double[rows]();

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			newVals[j][i] = vals[i][j];
		}
	}

	std::swap(newVals, vals);

	for (int i = 0; i < rows; i++)
		delete[] newVals[i];
	delete[] newVals;

	std::swap(rows, cols);
}

double Matrix::det() const {
	if (rows != cols)
		throw std::exception("This matrix is not a square");

	double** tmpMatrix = new double* [rows]();
	for (int i = 0; i < rows; i++) {
		tmpMatrix[i] = new double[cols]();
		for (int j = 0; j < cols; j++)
			tmpMatrix[i][j] = vals[i][j];
	}
	
	double determinant = 1;
	for (int i = 0; i < cols; i++) {
		int maxElem = i;

		for (int j = i + 1; j < rows; j++) {
			if (abs(tmpMatrix[maxElem][i]) < abs(tmpMatrix[j][i]))
				maxElem = j;
		}

		if (abs(tmpMatrix[maxElem][i]) < std::numeric_limits<double>::epsilon()) {
			determinant = 0;
			break;
		}
		
		if (i != maxElem) {
			std::swap(tmpMatrix[i], tmpMatrix[maxElem]);
			determinant = -determinant;
		}

		determinant *= tmpMatrix[i][i];

		for (int j = i + 1; j < cols; j++) {
			tmpMatrix[i][j] /= tmpMatrix[i][i];
		}

		for (int j = 0; j < rows; j++) {
			if (abs(tmpMatrix[j][i]) > std::numeric_limits<double>::epsilon() && i != j) {
				for (int k = i + 1; k < cols; k++) {
					tmpMatrix[j][k] -= tmpMatrix[j][i] * tmpMatrix[i][k];
				}
			}
		}
	}

	for (int i = 0; i < rows; i++)
		delete[] tmpMatrix[i];
	delete[] tmpMatrix;

	return determinant;
}

Matrix Matrix::inv() const {
	if (rows != cols) 
		throw std::exception("This matrix is not a square");
	
	Matrix inverted(*this);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Matrix minor(rows - 1);

			int dy = 0;
			for (int y = 0; y < rows; y++) {
				if (y == i) {
					dy++; continue;
				}

				int dx = 0;
				for (int x = 0; x < cols; x++) {
					if (x == j) {
						dx++; continue;
					}
					minor.vals[y - dy][x - dx] = vals[y][x];
				}
			}

			inverted.vals[i][j] = ((i + j) % 2 ? -1 : 1) * minor.det();
		}
	}
	inverted.transpose();
	inverted /= this->det();
	return inverted;
}

// Additional methods

void Matrix::swap(Matrix& rhs) {
	std::swap(this->cols, rhs.cols);
	std::swap(this->rows, rhs.rows);
	double** tmp = this->vals;
	this->vals = rhs.vals;
	rhs.vals = tmp;
}

void Matrix::print() {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			std::cout << vals[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}