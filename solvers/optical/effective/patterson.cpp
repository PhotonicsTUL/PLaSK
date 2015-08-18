#include "patterson.h"
#include "patterson-data.h"

namespace plask { namespace solvers { namespace effective {

template <typename R, typename T>
R patterson(const std::function<R(T)>& fun, T a, T b, double& err)
{
    double eps = err;
    err *= 2.;

    R result, result2;
    T D = (b - a) / 2., Z = (a + b) / 2.;

    R values[256]; // std::fill_n(values, 256, 0.);
    values[0] = fun(Z);
    result = (b - a) * values[0];

    unsigned n;

    for (n = 1; err > eps && n < 9; ++n) {
        unsigned N = 1 << n; // number of points in current iteration on one side of zero
        unsigned stp = 256 >> n; // step in x array
        result2 = result;
        // Compute integral on scaled range [-1, +1]
        result = patterson_weights[n][0] * values[0];
        for (unsigned i = stp, j = 1; j < N; i += stp, ++j) {
            if (j % 2) { // only odd points are new ones
                double x = patterson_points[i];
                T z1 = Z - D * x;
                T z2 = Z + D * x;
                values[i] = fun(z1) + fun(z2);
            }
            result += patterson_weights[n][j] * values[i];
        }
        // Rescale integral and compute error
        result *= D;
        err = abs(1. - result2 / result);
    }

#ifndef NDEBUG
    writelog(LOG_DEBUG, "Patterson quadrature for %1% points, error = %2%", (2<<n)-1, err);
#endif

    return result;
}

template double patterson<double,double>(const std::function<double(double)>& fun, double a, double b, double& err);

}}} // namespace plask::solvers::effective
