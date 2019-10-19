#include "SparseMatrix.h"
#include "Matrix.h"
#include "Vector.h"

#include <algorithm>
#include <numeric>
#include <iostream>

using namespace mat_vec;

SparseMatrix::SparseMatrix(size_t size) : SparseMatrix(size, size) {}

SparseMatrix::SparseMatrix(const Matrix& src) : rows(src.shape().first), cols(src.shape().second),
                                                        rowsOffset(new size_t[src.shape().first + 1]) {
    size_t count = 0;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (src(i, j) != 0)
                count++;
        }
    }

    rowsOffset[rows] = count + 1;
    colsIndxs = new size_t[count];
    vals = new double[count];

    size_t it = 0;
    for (size_t i = 0; i < rows; i++) {
        rowsOffset[i] = it;
        for (size_t j = 0; j < cols; j++) {          
            if (src(i, j) != 0) {
                vals[it] = src(i, j);
                colsIndxs[it] = j;
                it++;
            } 
        }
    }
}

SparseMatrix SparseMatrix::eye(size_t size) {
    SparseMatrix matrix(size);

    for (int i = 0; i < size; i++) {
        matrix.set(1, i, i);
    }

    return matrix;
}

SparseMatrix::SparseMatrix(size_t rows, size_t cols) : rows(rows), cols(cols),
                                    rowsOffset(new size_t[rows + 1]()),
                                    colsIndxs(new size_t[0]), vals(new double[0]) {
    rowsOffset[rows] = 1;
}

SparseMatrix::SparseMatrix(const SparseMatrix& src) : rows(src.rows), cols(src.cols),
                                    rowsOffset(new size_t[src.rows + 1]), 
                                    colsIndxs(new size_t[src.rowsOffset[src.rows] - 1]),
                                    vals(new double[src.rowsOffset[src.rows] - 1]) {
    for (size_t i = 0; i <= rows; i++)
        rowsOffset[i] = src.rowsOffset[i];

    for (size_t i = 0; i < rowsOffset[rows] - 1; i++) {
        vals[i] = src.vals[i];
        colsIndxs[i] = src.colsIndxs[i];
    }
}

SparseMatrix& mat_vec::SparseMatrix::operator=(const SparseMatrix& rhs) {
    SparseMatrix tmp(rhs);
    swap(tmp);
    return *this;
}

SparseMatrix::~SparseMatrix() {
    delete[] vals;
    delete[] colsIndxs;
    delete[] rowsOffset;
}

std::pair<size_t, size_t> mat_vec::SparseMatrix::shape() const {
    return std::pair<size_t, size_t>(rows, cols);
}

double mat_vec::SparseMatrix::density() const {
    return (double)(rowsOffset[rows] - 1) / ((double)rows * cols);
}

double mat_vec::SparseMatrix::sparsity() const {
    return 1 - density();
}

void SparseMatrix::set(double val, size_t row, size_t col) {
    if (row >= rows || col >= cols)
        throw std::out_of_range("Out of bounds");

    int elIndx = -1;
    for (size_t startIndx = rowsOffset[row]; startIndx < std::min(rowsOffset[row + 1], rowsOffset[rows] - 1); startIndx++) {
        if (colsIndxs[startIndx] == col) {
            elIndx = startIndx;
            break;
        }
    }

    if (abs(val) < std::numeric_limits<float>::epsilon()) {

        // erase
        if (elIndx != -1) {
            double* newVals = new double[rowsOffset[rows] - 2];
            size_t* newCols = new size_t[rowsOffset[rows] - 2];

            memcpy(newVals, vals, elIndx * sizeof(double));
            memcpy(newCols, colsIndxs, elIndx * sizeof(size_t));

            memcpy(newVals + elIndx, vals + elIndx + 1, (rowsOffset[rows] - 2 - elIndx) * sizeof(double));
            memcpy(newCols + elIndx, colsIndxs + elIndx + 1, (rowsOffset[rows] - 2 - elIndx) * sizeof(size_t));

            std::swap(newVals, vals);
            std::swap(newCols, colsIndxs);

            delete[] newVals;
            delete[] newCols;

            for (size_t i = row + 1; i <= rows; i++) {
                rowsOffset[i]--;
            }
        }

    // insert
    } else if (elIndx == -1) {
        double* newVals = new double[rowsOffset[rows]];
        size_t* newCols = new size_t[rowsOffset[rows]];

        memcpy(newVals, vals, rowsOffset[row] * sizeof(double));
        memcpy(newCols, colsIndxs, rowsOffset[row] * sizeof(size_t));

        newVals[rowsOffset[row]] = val;
        newCols[rowsOffset[row]] = col;

        memcpy(newVals + (rowsOffset[row] + 1), vals + rowsOffset[row],
            (rowsOffset[rows] - 1 - rowsOffset[row]) * sizeof(double));
        memcpy(newCols + (rowsOffset[row] + 1), colsIndxs + rowsOffset[row],
            (rowsOffset[rows] - 1 - rowsOffset[row]) * sizeof(size_t));

        std::swap(newVals, vals);
        std::swap(newCols, colsIndxs);

        delete[] newVals;
        delete[] newCols;

        for (size_t r = row + 1; r <= rows; r++)
            rowsOffset[r]++;

    // rewrite
    } else {
        vals[elIndx] = val;
    }

}

double SparseMatrix::get(size_t row, size_t col) const {
    if (row >= rows || col >= cols)
        throw std::out_of_range("Out of bounds");

    for (size_t i = rowsOffset[row]; i < rowsOffset[row + 1]; i++) {
        if (colsIndxs[i] == col)
            return vals[i];
    }

    return 0.0;
}

SparseMatrix mat_vec::SparseMatrix::operator+(const SparseMatrix& rhs) const {
    SparseMatrix tmp(*this);
    return tmp += rhs;
}

SparseMatrix& mat_vec::SparseMatrix::operator+=(const SparseMatrix& rhs) {
    if (rows != rhs.rows || cols != rhs.cols)
        throw std::invalid_argument("Matrices must have the same shape");

    size_t row = 0;
    for (size_t i = 0; i < rhs.rowsOffset[rows] - 1; i++) {
        while (i == rhs.rowsOffset[row + 1])
            row++;
        double val = get(row, rhs.colsIndxs[i]) + rhs.vals[i];
        set(val, row, rhs.colsIndxs[i]);
    }

    return *this;
}

SparseMatrix mat_vec::SparseMatrix::operator-(const SparseMatrix& rhs) const {
    SparseMatrix tmp(*this);
    return tmp -= rhs;
}

SparseMatrix& mat_vec::SparseMatrix::operator-=(const SparseMatrix& rhs) {
    if (rows != rhs.rows || cols != rhs.cols)
        throw std::invalid_argument("Matrices must have the same shape");

    size_t row = 0;
    for (size_t i = 0; i < rhs.rowsOffset[rows] - 1; i++) {
        while (i == rhs.rowsOffset[row + 1])
            row++;
        double val = get(row, rhs.colsIndxs[i]) - rhs.vals[i];
        set(val, row, rhs.colsIndxs[i]);
    }

    return *this;
}

SparseMatrix mat_vec::SparseMatrix::operator*(const SparseMatrix& rhs) const {
    SparseMatrix tmp(*this);
    return tmp *= rhs;
}

SparseMatrix& mat_vec::SparseMatrix::operator*=(const SparseMatrix& rhs) {
    if (this->cols != rhs.rows)
        throw std::invalid_argument("Can't do multiplication with matrices of these sizes");

    SparseMatrix res(this->rows, rhs.cols);
    SparseMatrix transpRhs = rhs.transposed();

    double curVal = 0;
    size_t row1 = 0, row2 = 0;
    for (size_t row2 = 0; row2 < transpRhs.rows; row2++) {
        for (size_t i = 0; i < rowsOffset[rows] - 1; i++) {
            while (i == rowsOffset[row1 + 1]) {
                if (abs(curVal) > std::numeric_limits<double>::epsilon()) {
                    res.set(curVal, row1, row2);
                }
                row1++;
                curVal = 0;
            }

            curVal += vals[i] * transpRhs.get(row2, colsIndxs[i]);
        }

        if (abs(curVal) > std::numeric_limits<double>::epsilon()) {
            res.set(curVal, row1, row2);
            curVal = 0;
        }
        row1 = 0;
    }
    

    swap(res);
    return *this;
}

SparseMatrix mat_vec::SparseMatrix::operator*(double k) const {
    SparseMatrix tmp(*this);
    return tmp *= k;
}

SparseMatrix& mat_vec::SparseMatrix::operator*=(double k) {
    for (size_t i = 0; i < rowsOffset[rows] - 1; i++)
        vals[i] *= k;
    return *this;
}

SparseMatrix mat_vec::SparseMatrix::operator/(double k) const {
    SparseMatrix tmp(*this);
    return tmp /= k;
}

SparseMatrix& mat_vec::SparseMatrix::operator/=(double k) {
    if (abs(k) < std::numeric_limits<double>::epsilon())
        throw std::invalid_argument("Division by zero");
    for (size_t i = 0; i < rowsOffset[rows] - 1; i++)
        vals[i] /= k;
    return *this;
}

Vector mat_vec::SparseMatrix::operator*(const Vector& vec) const {
    return vec * this->transposed();
}

Matrix SparseMatrix::denseMatrix() const {
    Matrix res(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = rowsOffset[i]; j < std::min(rowsOffset[i + 1], rowsOffset[rows] - 1); j++)
            res(i, colsIndxs[j]) = vals[j];
    }

    return res;
}

double mat_vec::SparseMatrix::det() const {
    return this->denseMatrix().det();
}

SparseMatrix mat_vec::SparseMatrix::inv() const {
    return SparseMatrix(this->denseMatrix().inv());
}

void SparseMatrix::swap(SparseMatrix& rhs) {
    std::swap(this->cols, rhs.cols);
    std::swap(this->rows, rhs.rows);
    std::swap(this->vals, rhs.vals);
    std::swap(this->rowsOffset, rhs.rowsOffset);
    std::swap(this->colsIndxs, rhs.colsIndxs);
}

void SparseMatrix::print() const {
    std::cout << "SparseMatrix print" << std::endl;
    std::cout << "Sizes: " << rows << " " << cols << std::endl;
    std::cout << "Real size " << rowsOffset[rows] << std::endl;
    std::cout << "Rows Offsets:" << std::endl;
    for (size_t i = 0; i <= rows; i++)
        std::cout << rowsOffset[i] << " ";
    std::cout << std::endl;

    std::cout << "Cols Indexes:" << std::endl;
    for (size_t i = 0; i < rowsOffset[rows] - 1; i++)
        std::cout << colsIndxs[i] << " ";
    std::cout << std::endl;

    std::cout << "Vals:" << std::endl;
    for (size_t i = 0; i < rowsOffset[rows] - 1; i++)
        std::cout << vals[i] << " ";
    std::cout << std::endl << std::endl;
}

SparseMatrix mat_vec::SparseMatrix::transposed() const {
    SparseMatrix res(*this);
    res.transpose();
    return res;
}

void mat_vec::SparseMatrix::transpose() {
    SparseMatrix tmp(cols, rows);
    size_t row = 0;
    for (size_t i = 0; i < rowsOffset[rows] - 1; i++) {
        while (i == rowsOffset[row + 1])
            row++;
        tmp.set(vals[i], colsIndxs[i], row);
    }
    swap(tmp);
}

bool mat_vec::SparseMatrix::operator==(const SparseMatrix& rhs) const {
    if (this->shape() != rhs.shape())
        return false;
    if (this->rowsOffset[rows] != rhs.rowsOffset[rhs.rows])
        return false;

    size_t row = 0;
    for (size_t i = 0; i < rowsOffset[rows] - 1; i++) {
        while (i == rowsOffset[row + 1])
            row++;
        if (abs(vals[i] - rhs.get(row, colsIndxs[i])) > std::numeric_limits<float>::epsilon())
            return false;
    }

    return true;
}

bool mat_vec::SparseMatrix::operator!=(const SparseMatrix& rhs) const {
    return !(*this == rhs);
}
