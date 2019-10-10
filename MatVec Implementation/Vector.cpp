#include "Vector.h"
#include "Base.h"
#include "Matrix.h"
#include <algorithm>
#include <iostream>

using namespace mat_vec;

Vector mat_vec::operator*(double k, const Vector& v) {
	Vector res(v);
	for (int i = 0; i < res.size(); i++)
		res[i] *= k;

	return res;
}

Vector::Vector(size_t size, double value) : count(size) {
	vals = new double[count]();
	for (int i = 0; i < count; i++)
		vals[i] = value;
}

Vector::Vector(const Vector& src) : count(src.count) {
	vals = new double[count]();
	for (int i = 0; i < count; i++)
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
		throw std::exception("Element is out of bounds");
	return vals[n];
}

double& Vector::operator[](size_t n) {
	if (n >= count)
		throw std::exception("Element is out of bounds");
	return vals[n];
}

double Vector::norm() const {
	double res = 0;
	for (int i = 0; i < count; i++)
		res += vals[i] * vals[i];

	return sqrt(res);
}

Vector Vector::normalized() const {
	Vector res(*this);
	res.normalize();
	return res;
}

void Vector::normalize() {
	int norm = this->norm();
	if (abs(norm) < std::numeric_limits<double>::epsilon())
		throw std::exception("Zero vector can't be normalized");
	double invNorm = 1.0 / norm;
	for (int i = 0; i < count; i++) {
		vals[i] *= invNorm;
	}
}

Vector Vector::operator+(const Vector& rhs) const {
	Vector res(*this);
	return res += rhs;
}

Vector& Vector::operator+=(const Vector& rhs) {
	if (count != rhs.count)
		throw std::exception("Can't do operation with these vectors");
	for (int i = 0; i < count; i++)
		vals[i] += rhs.vals[i];
	return *this;
}

Vector Vector::operator-(const Vector& rhs) const {
	Vector res(*this);
	return res -= rhs;
}
Vector& Vector::operator-=(const Vector& rhs) {
	if (count != rhs.count)
		throw std::exception("Can't do operation with these vectors");
	for (int i = 0; i < count; i++)
		vals[i] -= rhs.vals[i];
	return *this;
}

Vector Vector::operator^(const Vector& rhs) const {
	Vector res(*this);
	return res ^= rhs;
}

Vector& Vector::operator^=(const Vector& rhs) {
	if (count != rhs.count)
		throw std::exception("Can't do operation with these vectors");
	for (int i = 0; i < count; i++)
		vals[i] *= rhs.vals[i];
	return *this;
}

double Vector::operator*(const Vector& rhs) const {
	if (count != rhs.count)
		throw std::exception("Can't get dot product of two vectors with different dimensions");
	double res = 0;
	for (int i = 0; i < count; i++)
		res += vals[i] * rhs.vals[i];
	return res;
}

Vector Vector::operator*(double k) const {
	Vector res(*this);
	return res *= k;
}

Vector& Vector::operator*=(double k) {
	for (int i = 0; i < count; i++)
		vals[i] *= k;
	return *this;
}

Vector Vector::operator/(double k) const {
	Vector res(*this);
	return res /= k;
}
Vector& Vector::operator/=(double k) {
	for (int i = 0; i < count; i++)
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
		throw std::exception("Can't get a product of matrix with this size");

	double* newVals = new double[shape.second]();
	for (int i = 0; i < shape.second; i++) {
		for (int j = 0; j < count; j++) {
			newVals[i] += vals[j] * mat(j, i);
		}
	}

	std::swap(newVals, vals);
	delete[] newVals;

	count = shape.second;

	return *this;
}

bool Vector::operator==(const Vector& rhs) const {
	if (rhs.count != count)
		return false;
	for (int i = 0; i < count; i++) 
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
	for (int i = 0; i < count; i++)
		std::cout << vals[i] << " ";
	std::cout << std::endl;
}