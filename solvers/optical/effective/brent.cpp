#include "brent.h"
using namespace std;

namespace plask { namespace solvers { namespace effective {


double RootBrent::axisBrent(double a, double b, std::function<double(double)>fun, double& fx)
{
    //  Reference:
    //    Richard Brent,
    //    Algorithms for Minimization Without Derivatives,
    //    Dover, 2002,
    //    ISBN: 0-486-41998-3,
    //    LC: QA402.5.B74.

    const double C = 0.5 * (3. - sqrt(5.));

    double sa = a, sb = b, x = sa + C * (b - a), w = x, v = w, e = 0.0;
    fx = fun(x);
    double fw = fx, fv = fx;

    double d, u;

    for (unsigned i = 0; i < params.maxiter; ++i) {
        double m = 0.5 * (sa + sb) ;
        double tol = SMALL * abs(x) + params.tolx;
        double t2 = 2.0 * tol;

        // Check the stopping criterion.
        if (abs(x - m) <= t2 - 0.5 * (sb - sa)) return x;

        // Fit a parabola.
        double r = 0., q = 0., p = 0.;
        if (tol < abs(e)) {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (0.0 < q) p = - p;
            q = abs(q);
            r = e;
            e = d;
        }
        if (abs(p) < abs(0.5 * q * r) && q * (sa - x) < p && p < q * (sb - x)) {
            // Take the parabolic interpolation step
            d = p / q;
            u = x + d;
            // F must not be evaluated too close to a or b
            if ((u - sa) < t2 || (sb - u) < t2) {
                if (x < m) d = tol;
                else d = - tol;
            }
        } else {
            // A golden-section step.
            if (x < m) e = sb - x;
            else e = sa - x;
            d = C * e;
        }

        // F must not be evaluated too close to x
        if (tol <= abs(d)) u = x + d;
        else if (0.0 < d) u = x + tol;
        else u = x - tol;

        // Update a, b, v, w, and x
        double fu = fun(u);
        if (fu <= fx) {
            if (u < x) sb = x;
            else sa = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) sa = u;
            else sb = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }
    throw ComputationError(solver.getId(), "Brent: %1%: maximum number of iterations reached", log_value.chart_name);
    return 0;
}


//**************************************************************************
/// Search for a single mode starting from the given point: point
dcomplex RootBrent::find(dcomplex start) const
{
    dcomplex first = start - 0.5 * params.initial_dist;
    dcomplex second = start + 0.5 * params.initial_dist;

    writelog(LOG_DETAIL, "Searching for the root with two-step Brent method between %1% and %2%", str(first), str(second));
    log_value.resetCounter();
}

}}} // namespace plask::solvers::effective
