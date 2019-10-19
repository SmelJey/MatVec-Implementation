#pragma once

#include "Base.h"
#include "Vector.h"
#include <tuple>

namespace mat_vec {

    class SparseMatrix {
    public:
        // Конструирует матрицу с размерами size x size, заполненную 0
        explicit SparseMatrix(size_t size);

        // Конструирует разреженную матрицу на основе данной
        explicit SparseMatrix(const Matrix& src);

        // Возвращает единичную матрицу
        static SparseMatrix eye(size_t size);

        // Возвращает матрицу с размерами rows x cols, заполненную 0
        SparseMatrix(size_t rows, size_t cols);

        // Конструктор копирования
        SparseMatrix(const SparseMatrix& src);

        // Оператор присваивания
        SparseMatrix& operator=(const SparseMatrix& rhs);

        // Деструктор
        ~SparseMatrix();

        // Возвращает пару {rows, cols} -- количество строк и столбцов матрицы
        std::pair<size_t, size_t> shape() const;

        // Возвращает коэффициент заполненности матрицы
        double density() const;

        // Возвращает коэффициент разреженности матрицы
        double sparsity() const;

        // Изменяет элемент на позиции [row, col]
        void set(double val, size_t row, size_t col);

        // Возвращает элемент на позиции [row, col]
        double get(size_t row, size_t col) const;

        // Поэлементное сложение
        SparseMatrix operator+(const SparseMatrix& rhs) const;
        SparseMatrix& operator+=(const SparseMatrix& rhs);

        // Поэлементное вычитание
        SparseMatrix operator-(const SparseMatrix& rhs) const;
        SparseMatrix& operator-=(const SparseMatrix& rhs);

        // Матричное умножение
        SparseMatrix operator*(const SparseMatrix& rhs) const;
        SparseMatrix& operator*=(const SparseMatrix& rhs);

        // Умножение всех элементов матрицы на константу
        SparseMatrix operator*(double k) const;
        SparseMatrix& operator*=(double k);

        // Деление всех элементов матрицы на константу
        SparseMatrix operator/(double k) const;
        SparseMatrix& operator/=(double k);

        Vector operator*(const Vector& vec) const;
        friend Vector& Vector::operator*=(const SparseMatrix& mat);

        // Возвращает новую матрицу, полученную транспонированием текущей (this)
        SparseMatrix transposed() const;

        // Транспонирует текущую матрицу
        void transpose();

        // Поэлементное сравнение
        bool operator==(const SparseMatrix& rhs) const;
        bool operator!=(const SparseMatrix& rhs) const;

        // Возвращает обычную матрицу, соответствующую данной
        Matrix denseMatrix() const;

        // Определитель (Медленная реализация)
        double det() const;

        // Обратная матрица (Медленная реализация)
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