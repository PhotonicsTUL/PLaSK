#include "patterson.h"
#include "patterson-data.h"

namespace plask { namespace optical { namespace effective {

template <typename S, typename T>
S patterson(const std::function<S(T)>& fun, T a, T b, double& err)
{
    double eps = err;
    err *= 2.;

    S result, result2;
    T D = (b - a) / 2., Z = (a + b) / 2.;

    S values[256]; 
    values[0] = fun(Z);
    result = (b - a) * values[0];

    unsigned n;

    for (n = 1; err > eps && n < 9; ++n) {
        unsigned N = 1 << n; // number of points in current iteration on one side of zero
        unsigned N0 = N >> 1; // number of points in previous iteration on one side of zero
        result2 = result;
        result = 0.;
        // Add previously computed points
        for (unsigned i = 0; i < N0; ++i) {
            result +=  patterson_weights[n][i] * values[i];
        }
        // Compute and add new points
        for (unsigned i = N0; i < N; ++i) {
            double x = patterson_points[i];
            T z1 = Z - D * x;
            T z2 = Z + D * x;
            values[i] = fun(z1) + fun(z2);
            result +=  patterson_weights[n][i] * values[i];
        }
        // Rescale integral and compute error
        result *= D;
        err = abs(1. - result2 / result);
    }

#ifndef NDEBUG
    writelog(LOG_DEBUG, "Patterson quadrature for {0} points, error = {1}", (2<<n)-1, err);
#endif

    return result;
}

template double patterson<double,double>(const std::function<double(double)>& fun, double a, double b, double& err);

}}} // namespace plask::optical::effective
