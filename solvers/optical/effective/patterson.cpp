#include "patterson.h"
#include "patterson-data.h"

namespace plask { namespace solvers { namespace effective {

dcomplex patterson(const std::function<dcomplex(dcomplex)>& fun, dcomplex a, dcomplex b, double& err)
{
    double eps = err;
    err *= 2.;

    dcomplex result, result2;
    dcomplex D = (b - a) / 2., Z = (a + b) / 2.;

    dcomplex values[511]; std::fill_n(values, 511, 0.);
    values[255] = fun(Z);
    result = (b - a) * values[0];

    for (unsigned n = 1; err > eps && n < 9; ++n) {
        unsigned N = 1 << n; // number of point in current iteration on one size of zero
        unsigned stp = 256 >> n; // step in x array
        result2 = result;
        // Compute integral on scaled range [-1, +1]
        result = patterson_weights[n][0] * values[255];
        for (unsigned i = stp, j = 1; j < N; i += stp, ++j) {
            if (j % 2) { // only odd points are new ones
                double x = patterson_points[i];
                dcomplex z1 = Z - D * x;
                dcomplex z2 = Z + D * x;
                values[255-i] = fun(z1);
                values[255+i] = fun(z2);
            }
            result += patterson_weights[n][j] * (values[255-i] + values[255+i]);
        }
        // Rescale integral and compute error
        result *= D;
        err = abs(1. - result2 / result);
    }

    return result;
}

}}} // namespace plask::solvers::effective
