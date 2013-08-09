/**
 *  \file fortran.h Headers for foran BLAS i LAPACK functions
 */
#ifndef PLASK__SOLVER_VSLAB_FORTRAN_H
#define PLASK__SOLVER_VSLAB_FORTRAN_H

#include <plask/plask.hpp>
using plask::dcomplex;

#define F(fun) F77_##fun

// BLAS subroutines and functions

// perform one of the matrix-matrix operations   B := alpha*op(A)*B, or B := alpha*B*op(A)
// where alpha is a scalar, B is an m by n matrix, A is a unit, or non-unit, upper or
// lower triangular matrix and op(A) is one of op(A) = A or A' or conjg(A')
#define F77_ztrmm F77_GLOBAL(ztrmm,ZTRMM)
F77SUB F77_ztrmm(const char& side, const char& uplo, const char& transa, const char& diag,
                 const int& m, const int& n, const dcomplex& alpha, dcomplex* a, const int& lda,
                 dcomplex* b, const int& ldb);

// solve  one  of the matrix equations op(A)*X = alpha*B, or X*op(A) = alpha*B,
// where alpha is a scalar, X and B are m by n matrix, A is a unit, or non-unit, upper or
// lower triangular matrix and op(A) is one of op(A) = A or A' or conjg(A')
// the matrix X is overwritten on
#define F77_ztrsm F77_GLOBAL(ztrsm,ZTRSM)
F77SUB F77_ztrsm(const char& side, const char& uplo, const char& transa, const char& diag,
                 const int& m, const int& n, const dcomplex& alpha, dcomplex* a, const int& lda,
                 dcomplex* b, const int& ldb);

// perform  one  of  the  matrix-vector  operations  y  :=  alpha*A*x  + beta*y,
// or y := alpha*A'*x + beta*y, or   y := alpha*conjg(A')*x + beta*y,
#define F77_zgemv F77_GLOBAL(zgemv,ZGEMV)
F77SUB F77_zgemv(const char& trans, const int& m, const int& n, const dcomplex& alpha,
                 const dcomplex* a, const int& lda, const dcomplex* x, const int& incx,
                 const dcomplex& beta, dcomplex* y, const int& incy);

// perform one of the matrix-matrix operations   C := alpha*op(A)*op(B) + beta*C
#define F77_zgemm F77_GLOBAL(zgemm,ZGEMM)
F77SUB F77_zgemm(const char& transa, const char& transb, const int& m, const int& n,
                 const int& k, const dcomplex& alpha, const dcomplex *a, const int& lda,
                 const dcomplex *b, const int& ldb, const dcomplex& beta, dcomplex* c,
                 const int& ldc);


// LAPACK subroutines

// compute for an N-by-N complex nonsymmetric matrix A, the eigenvalues and, optionally,
// the left and/or right eigenvectors
#define F77_zgeev F77_GLOBAL(zgeev,ZGEEV)
F77SUB F77_zgeev(const char& jobvl, const char& jobvr, const int& n, dcomplex* a,
                 const int& lda, dcomplex* w, dcomplex* vl, const int& ldvl,
                 dcomplex* vr, const int& ldvr, dcomplex* work, const int& lwork,
                 double* rwork, int& info);

// compute the solution to a complex system of linear equations A * X = B
#define F77_zgesv F77_GLOBAL(zgesv,ZGESV)
F77SUB F77_zgesv(const int& n, const int& nrsh, dcomplex* a, const int& lda,
                 int* ipiv, dcomplex* b, const int& ldb, int& info);

// compute an LU factorization of a general M-by-N matrix A using partial pivoting
// with row interchanges
#define F77_zgetrf F77_GLOBAL(zgetrf,ZGETRF)
F77SUB F77_zgetrf(const int& m, const int& n, dcomplex* a, const int& lda,
                  int* ipiv, int& info);

// solve a system of linear equations A * X = B, A**T * X = B, or A**H * X = B
// with a general N-by-N matrix A using the LU factorization computed by ZGETRF
#define F77_zgetrs F77_GLOBAL(zgetrs,ZGETRS)
F77SUB F77_zgetrs(const char& trans, const int& n, const int& nrhs, const dcomplex* a,
                  const int& lda, const int* ipiv, dcomplex* b, const int& ldb, int& info);

// compute some or all of the right and/or left eigenvectors of a complex upper triangular matrix T
#define F77_ztrevc F77_GLOBAL(ztrevc,ZTREVC)
F77SUB F77_ztrevc(const char& side, const char& howmny, const int* select, const int& n,
                  dcomplex* t, const int& ldt, dcomplex* vl, const int& ldvl, dcomplex* vr,
                  const int& ldvr, const int& mm, int& m, dcomplex* work, double* rwork, int& info);

// reduce a general complex M-by-N matrix A to upper or lower bidiagonal form B by a unitary transformation
#define F77_zgebrd F77_GLOBAL(zgebrd,ZGEBRD)
F77SUB F77_zgebrd(const int& m, const int& n, dcomplex* a, const int& lda, double* d, double* e,
                  dcomplex* tauq, dcomplex* taup, dcomplex* work, const int& lwork, int& info);

// compute the singular value decomposition (SVD) of a real N-by-N (upper or lower) bidiagonal matrix B
#define F77_zbdsqr F77_GLOBAL(zbdsqr,ZBDSQR)
F77SUB F77_zbdsqr(const char &uplo, const int& n, const int& ncvt, const int& nru, const int& ncc,
                  double* d, double* e, dcomplex* vt, const int& ldvt, dcomplex* u, const int& ldu,
                  dcomplex* c, const int& ldc, double* rwork, int& info);

// compute the singular value decomposition (SVD) of a real N-by-N (upper or lower) bidiagonal matrix B
// using Divide and Conquer algorithm
#define F77_dbdsdc F77_GLOBAL(dbdsdc,DBDSDC)
F77SUB F77_dbdsdc(const char& uplo, const char& compq, const int& n, double* d, double* e,
                  double* u, const int& ldu, double* vt, const int& ldvt, double* q, int* iq,
                  double* work, int* iwork, int& info);

// compute the Cholesky factorization of a complex Hermitian positive definite matrix A stored in packed format
#define F77_zpptrf F77_GLOBAL(zpptrf,ZPPTRF)
F77SUB F77_zpptrf(const char& uplo, const int& n, dcomplex* ap, int& info);

// solve a system of linear equations A*X = B with a Hermitian positive definite matrix A in packed storage
// using the Cholesky factorization A = U**H*U or A = L*L**H computed by zpptrf
#define F77_zpptrs F77_GLOBAL(zpptrs,ZPPTRS)
F77SUB F77_zpptrs(const char& uplo, const int& n, const int& nrhs, dcomplex* ap,
                  dcomplex* b, const int& ldb, int& info);

// perform a series of row interchanges on the matrix A.
#define F77_zlaswp F77_GLOBAL(zlaswp,ZLASWP)
F77SUB F77_zlaswp(const int& n, dcomplex* a, const int& lda, const int& k1, const int& k2,
                  const int* ipiv, const int& incx);


// ARPACK subroutines

// reverse communication interface for the Implicitly Restarted Arnoldi iteration
#define F77_znaupd F77_GLOBAL(znaupd,ZNAUPD)
F77SUB F77_znaupd(int& ido, const char& bmat, const int& n, const char* which,
                  const int& nev, const double& tol, dcomplex* resid, const int& ncv,
                  dcomplex* v, const int& ldv, int* iparam, int* ipntr,
                  dcomplex* workd, dcomplex* workl, const int& lworkl,
                  double* rwork, int& info);

// this subroutine returns the converged approximations to eigenvalues
// of A*z = lambda*B*z and eigenvectors and/or an orthonormal basis
// for the associated approximate invariant subspace
#define F77_zneupd F77_GLOBAL(zneupd,ZNEUPD)
F77SUB F77_zneupd(const int& rvec, const char& howmny, const int* select, dcomplex* d,
                  dcomplex* z, const int& ldz, const double& sigma, dcomplex* workev,
                  const char& bmat, const int& n, const char* which, const int& nev,
                  const double& tol, dcomplex* resid, const int& ncv, dcomplex* v,
                  const int& ldv, int* iparam, int* ipntr, dcomplex* workd,
                  dcomplex* workl, const int& lworkl, double* rwork, int& info);


#endif // PLASK__SOLVER_VSLAB_FORTRAN_H
