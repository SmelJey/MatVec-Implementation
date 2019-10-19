#include "SparseMatrix.h"
#include "Matrix.h"
#include "Vector.h"

#include <algorithm>
#include <numeric>
#include <iostream>

using namespace mat_vec;

SparseMatrix::SparseMatrix(size_t size) : SparseMatrix(size, size) {}

SparseMatrix::SparseMatrix(const Matrix& src) : rows(src.shape().first), cols(src.shape().second),
                                                        rowsOffset(new size_t[rows + 1]()) {
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
        for (size_t j = 0; j < cols; j++) {
            if (src(i, j) != 0) {
                if (rowsOffset[i] == 0)
                    rowsOffset[i] = it;
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
        matrix.print();
    }

    return matrix;
}

SparseMatrix::SparseMatrix(size_t rows, size_t cols) : rows(rows), cols(cols)/*,
                                    rowsOffset(new int[rows + 1]()),
                                    colsIndxs(new int[cols]()), vals(new double[cols]()) */{
    rowsOffset = new size_t[rows + 1]();
    colsIndxs = new size_t[cols]();
    vals = new double[cols]();
    rowsOffset[rows] = 1;
}

SparseMatrix::SparseMatrix(const SparseMatrix& src) : rows(src.rows), cols(src.cols),
                                    rowsOffset(new size_t[src.rows + 1]()), colsIndxs(new size_t[src.rowsOffset[src.rows] - 1]),
                                    vals(new double[src.rowsOffset[src.rows] - 1]) {
    for (size_t i = 0; i <= rows; i++)
        rowsOffset[i] = src.rowsOffset[i];

    for (size_t i = 0; i < rowsOffset[rows] - 1; i++) {
        vals[i] = src.vals[i];
        colsIndxs[i] = src.colsIndxs[i];
    }
}

SparseMatrix::~SparseMatrix() {
    delete[] vals;
    delete[] colsIndxs;
    delete[] rowsOffset;
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

void SparseMatrix::swap(SparseMatrix& rhs) {
    std::swap(this->cols, rhs.cols);
    std::swap(this->rows, rhs.rows);
    std::swap(this->vals, rhs.vals);
    std::swap(this->rowsOffset, rhs.rowsOffset);
    std::swap(this->colsIndxs, rhs.colsIndxs);
}

void SparseMatrix::print() {
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
