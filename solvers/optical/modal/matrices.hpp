/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
/**
 *  \file   matrix.h Definitions of matrices and matrix operations
 */
#ifndef PLASK__SOLVER_VSLAB_MATRIX_H
#define PLASK__SOLVER_VSLAB_MATRIX_H

#include <cstring>
#include <cmath>

#include <plask/plask.hpp>
#include "fortran.hpp"

namespace plask { namespace optical { namespace modal {

template <typename T> class MatrixDiagonal;

/// General simple matrix template
template <typename T>
class Matrix {
  protected:
    size_t r, c;

    T* data_;               ///< The data of the matrix
    std::atomic<int>* gc;   ///< the reference count for the garbage collector

    void dec_ref() {    // see http://www.boost.org/doc/libs/1_53_0/doc/html/atomic/usage_examples.html "Reference counting" for optimal memory access description
        if (gc && gc->fetch_sub(1, std::memory_order_release) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            delete gc;
            aligned_free(data_);
            write_debug("freeing matrix {:d}x{:d} ({:.3f} MB) at {:p}", r, c, double(r*c*sizeof(T))/1048576., (void*)data_);
        }
    }

    void inc_ref() {
        if (gc) gc->fetch_add(1, std::memory_order_relaxed);
    }

  public:

    typedef T DataType;

    Matrix() : r(0), c(0), data_(nullptr), gc(nullptr) {}

    Matrix(size_t m, size_t n) : r(m), c(n),
        data_(aligned_malloc<T>(m*n)), gc(new std::atomic<int>(1)) {
        write_debug("allocating matrix {:d}x{:d} ({:.3f} MB) at {:p}", r, c, double(r*c*sizeof(T))/1048576., (void*)data_);
    }

    Matrix(size_t m, size_t n, T val) : r(m), c(n),
        data_(aligned_malloc<T>(m*n)), gc(new std::atomic<int>(1)) {
        write_debug("allocating matrix {:d}x{:d} ({:.3f} MB) at {:p}", r, c, double(r*c*sizeof(T))/1048576., (void*)data_);
        std::fill_n(data_, m*n, val);
    }

    Matrix(const Matrix<T>& M) : r(M.r), c(M.c), data_(M.data_), gc(M.gc) {
        inc_ref();
    }

    Matrix<T>& operator=(const Matrix<T>& M) {
        const_cast<Matrix<T>&>(M).inc_ref();    // must be called before dec_ref in case of self-asigment with *gc==1!
        this->dec_ref();
        r = M.r; c = M.c; data_ = M.data_; gc = M.gc;
        return *this;
    }

    Matrix(const MatrixDiagonal<T>& M): r(M.size()), c(M.size()),
        data_(aligned_malloc<T>(M.size()*M.size())), gc(new std::atomic<int>(1))  {
        write_debug("allocating matrix {:d}x{:d} ({:.3f} MB) at {:p} (from diagonal)", r, c, double(r*c*sizeof(T))/1048576., (void*)data_);
        std::fill_n(data_, r*c, 0);
        for (int j = 0, n = 0; j < r; j++, n += c+1) data_[n] = M[j];
    }

    Matrix(size_t m, size_t n, T* existing_data) : r(m), c(n), data_(existing_data), gc(nullptr) {
        // Create matrix using exiting data. This data is never garbage-collected
    }

    Matrix(size_t m, size_t n, std::initializer_list<T> data): r(m), c(n),
        data_(aligned_malloc<T>(m*n)), gc(new std::atomic<int>(1)) {
        std::copy(data.begin(), data.end(), data_);
    }

    ~Matrix() {
        dec_ref();
    }

    void reset() {
        dec_ref();
        r = c = 0;
        data_ = nullptr;
        gc = nullptr;
    }

    void reset(size_t m, size_t n) {
        dec_ref();
        r = m; c = n;
        data_ = aligned_malloc<T>(m*n);
        gc = new std::atomic<int>(1);
        write_debug("allocating matrix {:d}x{:d} ({:.3f} MB) at {:p}", r, c, double(r*c*sizeof(T))/1048576., (void*)data_);
    }

    void reset(size_t m, size_t n, T val) {
        dec_ref();
        r = m; c = n;
        data_ = aligned_malloc<T>(m*n);
        gc = new std::atomic<int>(1);
        write_debug("allocating matrix {:d}x{:d} ({:.3f} MB) at {:p}", r, c, double(r*c*sizeof(T))/1048576., (void*)data_);
        std::fill_n(data_, m*n, val);
    }

    void reset(size_t m, size_t n, T* existing_data) {
        dec_ref();
        r = m; c = n;
        data_ = existing_data;
        gc = nullptr;
    }

    inline const T* data() const { return data_; }
    inline T* data() { return data_; }

    inline const T& operator[](size_t i) const {
        assert(i < r*c);
        return data_[i];
    }
    inline T& operator[](size_t i) {
        assert(i < r*c);
        return data_[i];
    }

    inline const T& operator()(size_t m, size_t n) const {
        assert(m < r);
        assert(n < c);
        return data_[n*r + m];
    }
    inline T& operator()(size_t m, size_t n) {
        assert(m < r);
        assert(n < c);
        return data_[n*r + m];
    }

    inline size_t rows() const { return r; }
    inline size_t cols() const { return c; }

    Matrix<T> copy() const {
        Matrix<T> copy_(r, c);
        std::copy_n(data_, r*c, copy_.data());
        return copy_;
    }

    void copyto(Matrix<T>& dst) {
        assert(dst.rows() == rows() && dst.cols() == cols());
        assert(dst.rows() == rows() && dst.cols() == cols());
        std::copy(begin(), end(), dst.begin());
    }

    Matrix<T>& operator*=(T a) {
        size_t size = r*c; for (size_t i = 0; i < size; i++) data_[i] *= a;
        return *this;
    }
    Matrix<T>& operator/=(T a) {
        size_t size = r*c; for (size_t i = 0; i < size; i++) data_[i] /= a;
        return *this;
    }

    /// Check if the matrix contains any NaN
    inline bool isnan() const {
        const size_t n = r * c;
        for (size_t i = 0; i < n; ++i)
            if (std::isnan(real(data_[i])) || std::isnan(imag(data_[i]))) return true;
        return false;
    }

    /// Check if the matrix has any data
    bool empty() const {
        return !data_;
    }

    T* begin() const {
        return data_;
    }

    T* end() const {
        return data_ + r*c;
    }
};


/// General simple diagonal matrix template
template <typename T>
class MatrixDiagonal {
  protected:
    size_t siz;

    T* data_;               //< The data of the matrix
    std::atomic<int>* gc;   //< the reference count for the garbage collector

    void dec_ref() {    // see http://www.boost.org/doc/libs/1_53_0/doc/html/atomic/usage_examples.html "Reference counting" for optimal memory access description
        if (gc && gc->fetch_sub(1, std::memory_order_release) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            delete gc;
            aligned_free(data_);
            write_debug("freeing diagonal matrix {0}x{0} ({1:.3f} MB) at {2}", siz, double(siz*sizeof(T))/1048576., (void*)data_);
        }
    }

    void inc_ref() {
        if (gc) gc->fetch_add(1, std::memory_order_relaxed);
    }

  public:

    typedef T DataType;

    MatrixDiagonal() : siz(0), gc(nullptr) {}

    MatrixDiagonal(size_t n) : siz(n), data_(aligned_malloc<T>(n)), gc(new std::atomic<int>(1)) {
        write_debug("allocating diagonal matrix {0}x{0} ({1:.3f} MB) at {2}", siz, double(siz*sizeof(T))/1048576., (void*)data_);
    }

    MatrixDiagonal(size_t n, T val) : siz(n), data_(aligned_malloc<T>(n)), gc(new std::atomic<int>(1)) {
        write_debug("allocating and filling diagonal matrix {0}x{0} ({1:.3f} MB) at {2}", siz, double(siz*sizeof(T))/1048576., (void*)data_);
        std::fill_n(data_, n, val);
    }

    MatrixDiagonal(const MatrixDiagonal<T>& M) : siz(M.siz), data_(M.data_), gc(M.gc) {
        inc_ref();
    }

    MatrixDiagonal<T>& operator=(const MatrixDiagonal<T>& M) {
        const_cast<MatrixDiagonal<T>&>(M).inc_ref();    // must be called before dec_ref in case of self-asigment with *gc==1!
        this->dec_ref();
        siz = M.siz; data_ = M.data_; gc = M.gc;
        return *this;
    }

    ~MatrixDiagonal() {
        dec_ref();
    }

    inline void reset() {
        dec_ref();
        siz = 0;
        data_ = nullptr;
        gc = nullptr;
    }

    void reset(size_t n) {
        dec_ref();
        siz = n;
        data_ = aligned_malloc<T>(n);
        gc = new std::atomic<int>(1);
        write_debug("allocating diagonal matrix {0}x{0} ({1:.3f} MB) at {2}", siz, double(siz*sizeof(T))/1048576., (void*)data_);
    }

    void reset(size_t n, T val) {
        dec_ref();
        siz = n;
        data_ = aligned_malloc<T>(n);
        gc = new std::atomic<int>(1);
        write_debug("allocating diagonal matrix {0}x{0} ({1:.3f} MB) at {2}", siz, double(siz*sizeof(T))/1048576., (void*)data_);
        std::fill_n(data_, n, val);
    }

    inline const T* data() const { return data_; }
    inline T* data() { return data_; }

    inline const T& operator[](size_t n) const {
        assert(n < siz);
        return data_[n];
    }
    inline T& operator[](size_t n) {
        assert(n < siz);
        return data_[n];
    }

    inline const T& operator()(size_t m, size_t n) const {
        assert(m < siz);
        assert(n < siz);
        return (m == n)? data_[n] : 0;
    }
    inline T& operator()(size_t m, size_t n) {
        assert(m < siz);
        assert(n < siz);
        assert(m == n);
        return data_[n];
    }

    inline size_t size() const { return siz; }

    MatrixDiagonal<T> copy() const {
        MatrixDiagonal<T> C(siz);
        std::copy_n(data_, siz, C.data());
        return C;
    }

    MatrixDiagonal<T>& operator*=(T a) { for (int i = 0; i < siz; i++) data_[i] *= a; return *this; }
    MatrixDiagonal<T>& operator/=(T a) { for (int i = 0; i < siz; i++) data_[i] /= a; return *this; }

    /// Check if the matrix contains any NaN
    inline bool isnan() const {
        for (size_t i = 0; i != siz; ++i)
            if (std::isnan(real(data_[i])) || std::isnan(imag(data_[i]))) return true;
        return false;
    }

    /// Check if the matrix has any data
    bool empty() const {
        return !data_;
    }

    T* begin() const {
        return data_;
    }

    T* end() const {
        return data_ + siz;
    }
};

//**************************************************************************
// Rectangular matrix of real and complex numbers
typedef Matrix<double> dmatrix;
typedef Matrix<dcomplex> cmatrix;

// Column vector and diagonal matrix
typedef DataVector<dcomplex> cvector;
typedef DataVector<double> dvector;
typedef DataVector<const dcomplex> const_cvector;
typedef MatrixDiagonal<dcomplex> cdiagonal;

//**************************************************************************
/// Summation operator of the matrices
template <typename T>
inline Matrix<T> operator+(const Matrix<T>& A, const Matrix<T>& B) {
    // if (A.rows() != B.rows() || A.cols() != B.cols()) throw ComputationError("operator*<cmatrix,cmatrix>", "cannot add: dimensions does not match");
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    Matrix<T> C(A.rows(), A.cols());
    T* c = C.data();
    for (const T *a = A.data(), *b = B.data(), *end = A.data() + A.rows()*A.cols(); a != end; ++a, ++b, ++c)
        *c = *a + *b;
    return C;
}

/// Difference operator of the matrices
template <typename T>
inline Matrix<T> operator-(const Matrix<T>& A, const Matrix<T>& B) {
    // if (A.rows() != B.rows() || A.cols() != B.cols()) throw ComputationError("operator*<cmatrix,cmatrix>", "cannot add: dimensions does not match");
    assert(A.rows() == B.rows() && A.cols() == B.cols());
    Matrix<T> C(A.rows(), A.cols());
    T* c = C.data();
    for (const T *a = A.data(), *b = B.data(), *end = A.data() + A.rows()*A.cols(); a != end; ++a, ++b, ++c)
        *c = *a - *b;
    return C;
}

/// Negation operator of the matrix
template <typename T>
inline Matrix<T> operator-(const Matrix<T>& A) {
    Matrix<T> C(A.rows(), A.cols());
    for (T *a = A.data(), *c = C.data(), *end = A.data() + A.rows()*A.cols(); a != end; ++a, ++c)
        *c = - *a;
    return C;
}

/// Multiplication operator of the matrices (using BLAS level3)
inline cmatrix operator*(const cmatrix& A, const cmatrix& B) {
    // if (A.cols() != B.rows()) throw ComputationError("operator*<cmatrix,cmatrix>", "cannot multiply: A.cols != B.rows");
    assert(A.cols() == B.rows());
    cmatrix C(A.rows(), B.cols());
    zgemm('n', 'n', int(A.rows()), int(B.cols()), int(A.cols()), 1., A.data(), int(A.rows()), B.data(), int(B.rows()), 0., C.data(), int(C.rows()));
    return C;
}

/// Multiplication operator of the matrix-vector product (using BLAS level3)
inline cvector operator*(const cmatrix& A, const cvector& v) {
    size_t n = A.cols(), m = A.rows();
    // if (n != v.size()) throw ComputationError("mult_matrix_by_vector", "A.cols != v.size");
    assert(n == v.size());
    cvector dst(m);
    zgemv('n', int(m), int(n), 1., A.data(), int(m), v.data(), 1, 0., dst.data(), 1);
    return dst;
}

/// Multiplication by the diagonal matrix (right)
template <typename T>
inline cmatrix operator*(const Matrix<T>& A, const MatrixDiagonal<T>& B) {
    // if (A.cols() != B.size()) throw ComputationError("operator*<cmatrix,cdiagonal>", "cannot multiply: A.cols != B.size");
    assert(A.cols() == B.size());
    cmatrix C(A.rows(), B.size());
    size_t n = 0;
    const size_t r = A.rows(), c = A.cols();
    for (size_t j = 0; j < c; j++)
    {
        T b = B[j];
        for (size_t i = 0; i < r; i++, n++)
            C[n] = A[n] * b;
    }
    return C;
}

/// Multiplication by the diagonal matrix (left)
template <typename T>
inline cmatrix operator*(const MatrixDiagonal<T>& A, const Matrix<T>& B) {
    // if (A.size() != B.rows()) throw ComputationError("operator*<cdiagonal,cmatrix>", "cannot multiply: A.size != B.rows");
    assert(A.size() == B.rows());
    cmatrix C(A.size(), B.cols());
    size_t n = 0;
    const size_t r = B.rows(), c = B.cols();
    for (size_t j = 0; j < c; j++)
        for (size_t i = 0; i < r; i++, n++)
            C[n] = A[i] * B[n];
    return C;
}

/// Multiplication of matrix by diagonal in-place (replacing A)
template <typename T>
inline void mult_matrix_by_diagonal(Matrix<T>& A, const MatrixDiagonal<T>& B) {
    // if (A.cols() != B.size()) throw ComputationError("mult_matrix_by_diagonal", "cannot multiply: A.cols != B.size");
    assert(A.cols() == B.size());
    size_t n = 0;
    const size_t r = A.rows(), c = A.cols();
    for (size_t j = 0; j < c; j++) {
        T b = B[j];
        for (size_t i = 0; i < r; i++, n++)
            A[n] *= b;
    }
}

/// Multiplication of diagonal by matrix in-place (replacing B)
template <typename T>
inline void mult_diagonal_by_matrix(const MatrixDiagonal<T>& A, Matrix<T>& B) {
    // if (A.size() != B.rows()) throw ComputationError("mult_diagonal_by_matrix", "cannot multiply: A.size != B.rows");
    assert(A.size() == B.rows());
    size_t n = 0;
    const size_t r = B.rows(), c = B.cols();
    for (size_t j = 0; j < c; j++)
        for (size_t i = 0; i < r; i++, n++)
            B[n] *= A[i];
}


// BLAS wrappers for multiplications without allocating additional storage
inline void mult_matrix_by_vector(const cmatrix& A, const const_cvector& v, cvector& dst) {
    const size_t m = A.rows(),
                 n = A.cols();
    // if (n != v.size()) throw ComputationError("mult_matrix_by_vector", "A.cols != v.size");
    // if (m != dst.size()) throw ComputationError("mult_matrix_by_vector", "A.rows != dst.size");
    assert(n == v.size());
    assert(m == dst.size());
    zgemv('n', int(m), int(n), 1., A.data(), int(m), v.data(), 1, 0., dst.data(), 1);
}

inline void mult_matrix_by_matrix(const cmatrix& A, const cmatrix& B, cmatrix& dst) {
    const size_t k = A.cols(),
                 m = A.rows(),
                 n = B.cols();
    // if (k != B.rows()) throw ComputationError("mult_matrix_by_matrix", "cannot multiply: A.cols != B.rows");
    // if (m != dst.rows()) throw ComputationError("mult_matrix_by_matrix", "A.rows != dst.rows");
    // if (n != dst.cols()) throw ComputationError("mult_matrix_by_matrix", "B.cols != dst.cols");
    assert(k == B.rows());
    assert(m == dst.rows());
    assert(n == dst.cols());
    zgemm('n', 'n', int(m), int(n), int(k), 1., A.data(), int(m), B.data(), int(k), 0., dst.data(), int(m));
}

inline void add_mult_matrix_by_vector(const cmatrix& A, const cvector& v, cvector& dst) {
    const size_t m = A.rows(),
                 n = A.cols();
    // if (n != v.size()) throw ComputationError("add_mult_matrix_by_vector", "A.cols != v.size");
    // if (m != dst.size()) throw ComputationError("add_mult_matrix_by_vector", "A.rows != dst.size");
    assert(n == v.size());
    assert(m == dst.size());
    zgemv('n', int(m), int(n), 1., A.data(), int(m), v.data(), 1, 1., dst.data(), 1);
}

inline void add_mult_matrix_by_matrix(const cmatrix& A, const cmatrix& B, cmatrix& dst) {
    const size_t k = A.cols(),
                 m = A.rows(),
                 n = B.cols();
    // if (k != B.rows()) throw ComputationError("add_mult_matrix_by_matrix", "cannot multiply: A.cols != B.rows");
    // if (m != dst.rows()) throw ComputationError("add_mult_matrix_by_matrix", "A.rows != dst.rows");
    // if (n != dst.cols()) throw ComputationError("add_mult_matrix_by_matrix", "B.cols != dst.cols");
    assert(k == B.rows());
    assert(m == dst.rows());
    assert(n == dst.cols());
    zgemm('n', 'n', int(m), int(n), int(k), 1., A.data(), int(m), B.data(), int(k), 1., dst.data(), int(m));
}


// Some LAPACK wrappers
cmatrix invmult(cmatrix& A, cmatrix& B);
cvector invmult(cmatrix& A, cvector& B);
cmatrix inv(cmatrix& A);
dcomplex det(cmatrix& A);
int eigenv(cmatrix& A, cdiagonal& vals, cmatrix* rightv=NULL, cmatrix* leftv=NULL);

// r-value wrappers
inline cmatrix invmult(cmatrix&& A, cmatrix& B) { return invmult(A, B); }
inline cvector invmult(cmatrix&& A, cvector& B) { return invmult(A, B); };
inline cmatrix inv(cmatrix&& A) { return inv(A); }
inline dcomplex det(cmatrix&& A) { return det(A); }
inline int eigenv(cmatrix&& A, cdiagonal& vals, cmatrix* rightv=NULL, cmatrix* leftv=NULL) {
    return eigenv(A, vals, rightv, leftv);
}


/// Zero matrix
template <typename T>
void zero_matrix(Matrix<T>& A, size_t m, size_t n) {
    if (A.rows() != m || A.cols() != n) A.reset(m, n, 0.);
    else std::fill_n(A.data(), m*n, T(0.));
}

/// Zero matrix
template <typename T>
void zero_matrix(Matrix<T>& A) {
    std::fill_n(A.data(), A.rows()*A.cols(), T(0.));
}

/// Create unit matrix
template <typename T>
void make_unit_matrix(Matrix<T>& A, size_t n) {
    zero_matrix(A, n, n);
    constexpr T one(1.);
    for (int i = 0; i < n; ++i) A(i,i) = one;
}

/// Create unit matrix
template <typename T>
void make_unit_matrix(Matrix<T>& A) {
    assert(A.rows() == A.cols());
    zero_matrix(A);
    constexpr T one(1.);
    for (int i = 0; i < A.rows(); ++i) A(i,i) = one;
}

}}}
#endif // PLASK__SOLVER_VSLAB_MATRIX_H
