#include "Matrix.h"
#include "Vector.h"
#include <iostream>
#include <algorithm>
#include <numeric>

using namespace mat_vec;

Matrix::Matrix(size_t size, double value) : Matrix(size, size, value) {}

Matrix Matrix::eye(size_t size) {
    Matrix unit(size);
    for (size_t i = 0; i < size; i++)
        unit.vals[i][i] = 1;
    return unit;
}

Matrix::Matrix(size_t rows, size_t cols, double value) : rows(rows), cols(cols),
                                                        vals(new double* [rows]) {
    for (size_t i = 0; i < rows; i++) {
        vals[i] = new double[cols];
        for (size_t j = 0; j < cols; j++)
            vals[i][j] = value;
    }
}

Matrix::Matrix(const Matrix& src) : rows(src.rows), cols(src.cols),
                                    vals(new double* [src.rows]){;
    for (size_t i = 0; i < rows; i++) {
        vals[i] = new double[cols];
        for (size_t j = 0; j < cols; j++)
            vals[i][j] = src.vals[i][j];
    }
}

Matrix& Matrix::operator=(const Matrix& rhs) {
    Matrix tmp(rhs);
    swap(tmp);
    return *this;
}

Matrix::~Matrix() {
    for (size_t i = 0; i < rows; i++)
        delete[] vals[i];
    delete[] vals;
}

void Matrix::reshape(size_t rows, size_t cols) {
    if (rows * cols != this->rows * this->cols)
        throw std::invalid_argument("New capacity of matrix must be the same");
    size_t i1 = 0, j1 = 0;

    double** newVals = new double*[rows];
    for (size_t i = 0; i < rows; i++) {
        newVals[i] = new double[cols]();
        for (size_t j = 0; i1 < this->rows && j < cols; j++) {
            newVals[i][j] = vals[i1][j1];
            j1++;
            if (j1 >= this->cols) {
                j1 = 0; i1++;
            }
        }
    }

    std::swap(newVals, this->vals);
    for (size_t i = 0; i < this->rows; i++)
        delete[] newVals[i];
    delete[] newVals;

    this->rows = rows;
    this->cols = cols;
}

std::pair<size_t, size_t> Matrix::shape() const {
    return std::pair<size_t, size_t>(rows, cols);
}

double Matrix::get(size_t row, size_t col) const {
    if (row >= rows || col >= cols)
        throw std::out_of_range("Out of bounds");
    return this->vals[row][col];
}

Matrix Matrix::operator+(const Matrix& rhs) const {
    Matrix res(*this);
    return res += rhs;
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
    if (rows != rhs.rows || cols != rhs.cols)
        throw std::invalid_argument("Matrices must have the same shape");
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            vals[i][j] += rhs.vals[i][j];
        }
    }
    return *this;
}

Matrix Matrix::operator-(const Matrix& rhs) const {
    Matrix res(*this);
    return res -= rhs;;
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
    if (rows != rhs.rows || cols != rhs.cols)
        throw std::invalid_argument("Matrices must have the same shape");
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            vals[i][j] -= rhs.vals[i][j];
        }
    }
    return *this;
}

Matrix Matrix::operator*(const Matrix& rhs) const {
    Matrix res(*this);
    return res *= rhs;
}

Matrix& Matrix::operator*=(const Matrix& rhs) {
    if (this->cols != rhs.rows)
        throw std::invalid_argument("Can't do multiplication with matrices of these sizes");

    double** res = new double* [rows]();
    for (size_t i = 0; i < rows; i++)
        res[i] = new double[rhs.cols]();

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < rhs.cols; j++) {
            for (size_t k = 0; k < cols; k++) {
                res[i][j] += vals[i][k] * rhs.vals[k][j];
            }
        }
    }

    std::swap(res, this->vals);
    for (size_t i = 0; i < rows; i++)
        delete[] res[i];
    delete[] res;

    cols = rhs.cols;

    return *this;
}

Matrix Matrix::operator*(double k) const {
    Matrix res(*this);
    return res *= k;
}

Matrix& Matrix::operator*=(double k) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            vals[i][j] *= k;
        }
    }
    return *this;
}

Matrix Matrix::operator/(double k) const {
    Matrix res(*this);
    return res /= k;
}
Matrix& Matrix::operator/=(double k) {
    if (abs(k) < std::numeric_limits<double>::epsilon())
        throw std::invalid_argument("Division by zero");
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            vals[i][j] /= k;
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
    for (size_t i = 0; i < cols; i++)
        newVals[i] = new double[rows]();

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            newVals[j][i] = vals[i][j];
        }
    }

    std::swap(newVals, vals);

    for (size_t i = 0; i < rows; i++)
        delete[] newVals[i];
    delete[] newVals;

    std::swap(rows, cols);
}

double Matrix::det() const {
    if (rows != cols)
        throw std::invalid_argument("This matrix is not a square");

    double** tmpMatrix = new double* [rows]();
    for (size_t i = 0; i < rows; i++) {
        tmpMatrix[i] = new double[cols]();
        for (size_t j = 0; j < cols; j++)
            tmpMatrix[i][j] = vals[i][j];
    }

    double determinant = 1;
    for (size_t i = 0; i < cols; i++) {
        int maxElem = i;

        for (size_t j = i + 1; j < rows; j++) {
            if (abs(tmpMatrix[maxElem][i]) < abs(tmpMatrix[j][i]))
                maxElem = j;
        }

        if (abs(tmpMatrix[maxElem][i]) < std::numeric_limits<float>::epsilon()) {
            determinant = 0;
            break;
        }
        
        if (i != maxElem) {
            std::swap(tmpMatrix[i], tmpMatrix[maxElem]);
            determinant = -determinant;
        }

        determinant *= tmpMatrix[i][i];

        for (size_t j = i + 1; j < cols; j++) {
            tmpMatrix[i][j] /= tmpMatrix[i][i];
        }

        for (size_t j = 0; j < rows; j++) {
            if (abs(tmpMatrix[j][i]) > std::numeric_limits<float>::epsilon() && i != j) {
                for (size_t k = i + 1; k < cols; k++) {
                    tmpMatrix[j][k] -= tmpMatrix[j][i] * tmpMatrix[i][k];
                }
            }
        }
    }

    for (size_t i = 0; i < rows; i++)
        delete[] tmpMatrix[i];
    delete[] tmpMatrix;

    if (cols == 0 || rows == 0)
        return 0;

    return determinant;
}

Matrix Matrix::inv() const {
    if (rows != cols) 
        throw std::invalid_argument("This matrix is not a square");

    double det = this->det();

    if (abs(det) < std::numeric_limits<float>::epsilon())
        throw std::invalid_argument("Determinant is 0");
    
    Matrix inverted(*this);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            Matrix minor(rows - 1);

            int dy = 0;
            for (size_t y = 0; y < rows; y++) {
                if (y == i) {
                    dy++; continue;
                }

                int dx = 0;
                for (size_t x = 0; x < cols; x++) {
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

    inverted /= det;
    return inverted;
}

Vector Matrix::operator*(const Vector& vec) const {
    return vec * this->transposed();
}

bool Matrix::operator==(const Matrix& rhs) const {
    if (rows != rhs.rows || cols != rhs.cols)
        return false;
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            if (abs(vals[i][j] - rhs.vals[i][j]) > std::numeric_limits<float>::epsilon())
                return false;

    return true;
}

bool Matrix::operator!=(const Matrix& rhs) const {
    return !(*this == rhs);
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
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << vals[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols)
        throw std::out_of_range("Out of bounds");
    return vals[row][col];
}
double Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols)
        throw std::out_of_range("Out of bounds");
    return vals[row][col];
}