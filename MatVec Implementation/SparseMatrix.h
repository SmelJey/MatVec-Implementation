#pragma once

#include "Base.h"
#include "Vector.h"
#include <tuple>

namespace mat_vec {

    class SparseMatrix {
    public:
        // ������������ ������� � ��������� size x size, ����������� 0
        explicit SparseMatrix(size_t size);

        // ������������ ����������� ������� �� ������ ������
        explicit SparseMatrix(const Matrix& src);

        // ���������� ��������� �������
        static SparseMatrix eye(size_t size);

        // ���������� ������� � ��������� rows x cols, ����������� 0
        SparseMatrix(size_t rows, size_t cols);

        // ����������� �����������
        SparseMatrix(const SparseMatrix& src);

        // �������� ������������
        SparseMatrix& operator=(const SparseMatrix& rhs);

        // ����������
        ~SparseMatrix();

        // ���������� ���� {rows, cols} -- ���������� ����� � �������� �������
        std::pair<size_t, size_t> shape() const;

        // ���������� ����������� ������������� �������
        double density() const;

        // ���������� ����������� ������������� �������
        double sparsity() const;

        // �������� ������� �� ������� [row, col]
        void set(double val, size_t row, size_t col);

        // ���������� ������� �� ������� [row, col]
        double get(size_t row, size_t col) const;

        // ������������ ��������
        SparseMatrix operator+(const SparseMatrix& rhs) const;
        SparseMatrix& operator+=(const SparseMatrix& rhs);

        // ������������ ���������
        SparseMatrix operator-(const SparseMatrix& rhs) const;
        SparseMatrix& operator-=(const SparseMatrix& rhs);

        // ��������� ���������
        SparseMatrix operator*(const SparseMatrix& rhs) const;
        SparseMatrix& operator*=(const SparseMatrix& rhs);

        // ��������� ���� ��������� ������� �� ���������
        SparseMatrix operator*(double k) const;
        SparseMatrix& operator*=(double k);

        // ������� ���� ��������� ������� �� ���������
        SparseMatrix operator/(double k) const;
        SparseMatrix& operator/=(double k);

        Vector operator*(const Vector& vec) const;
        friend Vector& Vector::operator*=(const SparseMatrix& mat);

        // ���������� ����� �������, ���������� ����������������� ������� (this)
        SparseMatrix transposed() const;

        // ������������� ������� �������
        void transpose();

        // ������������ ���������
        bool operator==(const SparseMatrix& rhs) const;
        bool operator!=(const SparseMatrix& rhs) const;

        // ���������� ������� �������, ��������������� ������
        Matrix denseMatrix() const;

        // ������������ (��������� ����������)
        double det() const;

        // �������� ������� (��������� ����������)
        SparseMatrix inv() const;

        // Additional methods
        void swap(SparseMatrix& rhs);
        void print() const;

    private:
        double* vals;
        size_t* colsIndxs;
        size_t* rowsOffset;

        size_t rows, cols;
    };
}