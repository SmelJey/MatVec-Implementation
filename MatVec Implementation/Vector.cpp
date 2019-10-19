#include "Vector.h"
#include "Base.h"
#include "Matrix.h"
#include "SparseMatrix.h"
#include <algorithm>
#include <iostream>

using namespace mat_vec;

Vector mat_vec::operator*(double k, const Vector& v) {
	Vector res(v);
	for (size_t i = 0; i < res.size(); i++)
		res[i] *= k;

	return res;
}

Vector::Vector(size_t size, double value) : count(size) {
    vals = new double[count];
	for (size_t i = 0; i < count; i++)
		vals[i] = value;
}

Vector::Vector(const Vector& src) : count(src.count) {
	vals = new double[count]();
	for (size_t i = 0; i < count; i++)
		vals[i] = src.vals[i];
}

Vector& Vector::operator=(const Vector& rhs) {
	Vector tmp(rhs);
	swap(tmp);
	return *this;
}

Vector::~Vector() {
	delete[] vals;
}

size_t Vector::size() const {
	return count;
}

double Vector::operator[](size_t n) const {
	if (n >= count)
		throw std::out_of_range("Element is out of bounds");
	return vals[n];
}

double& Vector::operator[](size_t n) {
	if (n >= count)
		throw std::out_of_range("Element is out of bounds");
	return vals[n];
}

double Vector::norm() const {
	double res = 0;
	for (size_t i = 0; i < count; i++)
		res += vals[i] * vals[i];

	return sqrt(res);
}

Vector Vector::normalized() const {
	Vector res(*this);
	res.normalize();
	return res;
}

void Vector::normalize() {
	double norm = this->norm();
	if (abs(norm) < std::numeric_limits<double>::epsilon())
		throw std::invalid_argument("Zero vector can't be normalized");
	double invNorm = 1.0 / norm;
	for (size_t i = 0; i < count; i++) {
		vals[i] *= invNorm;
	}
}

Vector Vector::operator+(const Vector& rhs) const {
	Vector res(*this);
	return res += rhs;
}

Vector& Vector::operator+=(const Vector& rhs) {
	if (count != rhs.count)
		throw std::invalid_argument("Vectors must have the same size");
	for (size_t i = 0; i < count; i++)
		vals[i] += rhs.vals[i];
	return *this;
}

Vector Vector::operator-(const Vector& rhs) const {
	Vector res(*this);
	return res -= rhs;
}
Vector& Vector::operator-=(const Vector& rhs) {
	if (count != rhs.count)
		throw std::invalid_argument("Vectors must have the same size");
	for (size_t i = 0; i < count; i++)
		vals[i] -= rhs.vals[i];
	return *this;
}

Vector Vector::operator^(const Vector& rhs) const {
	Vector res(*this);
	return res ^= rhs;
}

Vector& Vector::operator^=(const Vector& rhs) {
	if (count != rhs.count)
		throw std::invalid_argument("Vectors must have the same size");
	for (size_t i = 0; i < count; i++)
		vals[i] *= rhs.vals[i];
	return *this;
}

double Vector::operator*(const Vector& rhs) const {
	if (count != rhs.count)
		throw std::invalid_argument("Vectors must have the same size");
	double res = 0;
	for (size_t i = 0; i < count; i++)
		res += vals[i] * rhs.vals[i];
	return res;
}

Vector Vector::operator*(double k) const {
	Vector res(*this);
	return res *= k;
}

Vector& Vector::operator*=(double k) {
	for (size_t i = 0; i < count; i++)
		vals[i] *= k;
	return *this;
}

Vector Vector::operator/(double k) const {
	Vector res(*this);
	return res /= k;
}
Vector& Vector::operator/=(double k) {
    if (abs(k) < std::numeric_limits<double>::epsilon())
        throw std::invalid_argument("Division by zero");
	for (size_t i = 0; i < count; i++)
		vals[i] /= k;
	return *this;
}

Vector Vector::operator*(const Matrix& mat) const {
	Vector res(*this);
	return res *= mat;
}

Vector& Vector::operator*=(const Matrix& mat) {
	auto shape = mat.shape();
	if (count != shape.first)
		throw std::invalid_argument("Vector must have the same size as matrix");

	double* newVals = new double[shape.second]();
	for (size_t i = 0; i < shape.second; i++) {
		for (size_t j = 0; j < count; j++) {
			newVals[i] += vals[j] * mat(j, i);
		}
	}

	std::swap(newVals, vals);
	delete[] newVals;

	count = shape.second;

	return *this;
}

Vector mat_vec::Vector::operator*(const SparseMatrix& mat) const {
    Vector tmp(*this);
    return tmp *= mat;
}

Vector& mat_vec::Vector::operator*=(const SparseMatrix& mat) {
    if (mat.shape().first != count)
        throw std::invalid_argument("Vector must have the same size as matrix");

    Vector res(mat.shape().second);
    size_t row = 0;
    for (size_t i = 0; i < mat.rowsOffset[mat.rows] - 1; i++) {
        while (mat.rowsOffset[row + 1] == i)
            row++;
        res[mat.colsIndxs[i]] += mat.vals[i] * (*this)[row];
    }
    
    swap(res);
    return *this;
}

bool Vector::operator==(const Vector& rhs) const {
	if (rhs.count != count)
		return false;
	for (size_t i = 0; i < count; i++)
		if (rhs.vals[i] != vals[i])
			return false;
	return true;
}
bool Vector::operator!=(const Vector& rhs) const {
	return !(*this == rhs);
}

// Additional methods

void Vector::swap(Vector& rhs) {
	std::swap(count, rhs.count);
	double* tmp = this->vals;
	this->vals = rhs.vals;
	rhs.vals = tmp;
}

void Vector::print() {
	for (size_t i = 0; i < count; i++)
		std::cout << vals[i] << " ";
	std::cout << std::endl << std::endl;
}