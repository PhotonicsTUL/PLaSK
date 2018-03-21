#include "gauss_legendre.h"
#include "fortran.h"

#include <memory>

namespace plask { namespace optical { namespace slab {

inline static double sign(double x) { return (x < 0.)? -1. : 1.; }

/** IMTQLX diagonalizes a symmetric tridiagonal matrix.

 Discussion:

   This routine is a slightly modified version of the EISPACK routine to
   perform the implicit QL algorithm on a symmetric tridiagonal matrix.

   The authors thank the authors of EISPACK for permission to use this
   routine.

   It has been modified to produce the product Q' * Z, where Z is an input
   vector and Q is the orthogonal matrix diagonalizing the input matrix.
   The changes consist (essentialy) of applying the orthogonal transformations
   directly to Z as they are generated.

 Licensing:

   This code is distributed under the GNU LGPL license.

 Modified:

   08 January 2010

 Author:

   Original FORTRAN77 version by Sylvan Elhay, Jaroslav Kautsky.
   C++ version by John Burkardt.

 Reference:

   Sylvan Elhay, Jaroslav Kautsky,
   Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of
   Interpolatory Quadrature,
   ACM Transactions on Mathematical Software,
   Volume 13, Number 4, December 1987, pages 399-415.

   Roger Martin, James Wilkinson,
   The Implicit QL Algorithm,
   Numerische Mathematik,
   Volume 12, Number 5, December 1968, pages 377-383.

 Parameters:

   Input, int N, the order of the matrix.

   Input/output, double D(N), the diagonal entries of the matrix.
   On output, the information in D has been overwritten.

   Input/output, double E(N), the subdiagonal entries of the
   matrix, in entries E(1) through E(N-1).  On output, the information in
   E has been overwritten.

   Input/output, double Z(N).  On input, a vector.  On output,
   the value of Q' * Z, where Q is the matrix that diagonalizes the
   input symmetric tridiagonal matrix.
*/
inline static void imtqlx(int n, double d[], double e[], double z[])
{
    double b;
    double c;
    double f;
    double g;
    int i;
    int ii;
    const int itn = 30;
    int j;
    int k;
    int l;
    int m;
    int mml;
    double p;
    double prec;
    double r;
    double s;

  prec = std::numeric_limits<double>::epsilon();

  if (n == 1) return;

    e[n-1] = 0.0;

    for (l = 1; l <= n; l++) {
        j = 0;
        for (;;) {
            for (m = l; m <= n; m++) {
                if (m == n) break;
                if (fabs(e[m-1]) <= prec * (fabs(d[m-1]) + fabs(d[m])))
                    break;
            }
            p = d[l-1];
            if (m == l) break;
            if (j >= itn)
                throw "Iteration limit exceeded\n";
            j = j + 1;
            g = (d[l] - p) / (2.0 * e[l-1]);
            r = sqrt(g * g + 1.0);
            g = d[m-1] - p + e[l-1] / (g + fabs(r) * sign(g));
            s = 1.0;
            c = 1.0;
            p = 0.0;
            mml = m - l;

            for (ii = 1; ii <= mml; ii++)
            {
                i = m - ii;
                f = s * e[i-1];
                b = c * e[i-1];

                if (fabs(g) <= fabs(f)) {
                    c = g / f;
                    r =  sqrt(c * c + 1.0);
                    e[i] = f * r;
                    s = 1.0 / r;
                    c = c * s;
                } else {
                    s = f / g;
                    r =  sqrt(s * s + 1.0);
                    e[i] = g * r;
                    c = 1.0 / r;
                    s = s * c;
                }
                g = d[i] - p;
                r = (d[i-1] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i] = g + p;
                g = c * r - b;
                f = z[i];
                z[i] = s * z[i-1] + c * f;
                z[i-1] = c * z[i-1] - s * f;
            }
            d[l-1] = d[l-1] - p;
            e[l-1] = g;
            e[m-1] = 0.0;
        }
    }

    //  Sorting.
    for (ii = 2; ii <= m; ii++) {
        i = ii - 1;
        k = i;
        p = d[i-1];

        for (j = ii; j <= n; j++) {
            if (d[j-1] < p) {
                k = j;
                p = d[j-1];
            }
        }

        if (k != i) {
            d[k-1] = d[i-1];
            d[i-1] = p;
            p = z[i-1];
            z[i-1] = z[k-1];
            z[k-1] = p;
        }
    }
}


void gaussLaguerre(size_t n, std::vector<double>& abscissae, DataVector<double>& weights, double scale)
{
    std::unique_ptr<double[]> work(new double[n]);

    abscissae.resize(n);
    weights.reset(n);

    for (size_t i = 0; i != n; ++i) {
        abscissae[i] = double(2 * i + 1);
        work[i] = double(i + 1);
    }

    std::fill(weights.begin(), weights.end(), 0.);
    weights[0] = 1.0;

    imtqlx(int(n), &abscissae.front(), work.get(), weights.data());

    double iscale = 1. / scale;
    for (size_t i = 0; i < n; ++i) {
        double factor = exp(abscissae[i]);
        if (!std::isfinite(factor)) factor = 0.;
        weights[i] *= iscale * weights[i] * factor;
        abscissae[i] *= iscale;
    }
}

}}} // # namespace plask::optical::slab
