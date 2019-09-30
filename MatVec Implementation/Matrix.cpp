#include "Base.h"
#include "Matrix.h"
#include <iostream>

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
	vals = new double* [rows];
	for (int i = 0; i < rows; i++) {
		vals[i] = new double[cols];
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