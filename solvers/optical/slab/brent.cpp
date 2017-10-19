#include "brent.h"

namespace plask { namespace optical { namespace slab {

#define carg(x) realaxis? dcomplex(x, imag(start)) : dcomplex(real(start), x)
#define fun(x) abs(val_function(carg(x)))

double RootBrent::axisBrent(dcomplex start, double& fx, bool realaxis, int& counter)
{
    const double C = 0.5 * (3. - sqrt(5.));
    const double G = 2. / (sqrt(5.) - 1.);

    unsigned i = 0;
    double x, dist;

    if (realaxis) {
        x = real(start);
        dist = real(params.initial_dist);
    } else {
        x = imag(start);
        if (imag(params.initial_dist) == 0) dist = real(params.initial_dist);
        else dist = imag(params.initial_dist);
    }

    // Exponential search
    double a = x - dist,
           b = x + dist;

    writelog(LOG_DETAIL, "Searching for the root with Brent method between {0} and {1} in the {2} axis",
                         str(a), str(b), realaxis?"real":"imaginary");

    if (isnan(fx)) fx = fun(x);

    double fw = fun(a),
           fv = fun(b);
    log_value(carg(a), fw);
    log_value(carg(x), fx);
    log_value(carg(b), fv);

    double d, u;

    log_value.resetCounter();

    bool bounded = false;
    if (fw <= fx && fx <= fv) {
        writelog(LOG_DETAIL, "Extending search range to lower values");
        for (; i < params.maxiter; ++i) {
            u = a - G * (x - a);
            b = x; fv = fx;
            x = a; fx = fw;
            a = u; fw = fun(a);
            counter++;
            log_value.count(carg(b), fw);
            if (fw > fx) {
                bounded = true;
                break;
            }
        }
        if (bounded) writelog(LOG_DETAIL, "Searching minimum in range between {0} and {1}", a, b);
    } else if (fw >= fx && fx >= fv) {
        writelog(LOG_DETAIL, "Extending search range to higher values");
        for (; i < params.maxiter; ++i) {
            u = b + G * (b - x);
            a = x; fw = fx;
            x = b; fx = fv;
            b = u; fv = fun(b);
            counter++;
            log_value.count(carg(b), fv);
            if (fv > fx) {
                bounded = true;
                break;
            }
        }
        if (bounded) writelog(LOG_DETAIL, "Searching minimum in range between {0} and {1}", a, b);
    } else bounded = true;

    if (!bounded)
        throw ComputationError(solver.getId(),
                               "Brent: {0}: minimum still unbounded after maximum number of iterations",
                               log_value.chart_name);

    double sa = a, sb = b, w = x, v = x, e = 0.0;
    fw = fv = fx;

    //  Reference:
    //    Richard Brent,
    //    Algorithms for Minimization Without Derivatives,
    //    Dover, 2002,
    //    ISBN: 0-486-41998-3,
    //    LC: QA402.5.B74.

    double tol = SMALL * abs(x) + params.tolx;
    double tol2 = 2.0 * tol;

    for (; i < params.maxiter; ++i) {
        double m = 0.5 * (sa + sb) ;

        // Check the stopping criterion.
        if (abs(x - m) <= tol2 - 0.5 * (sb - sa)) return x;

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
            if ((u - sa) < tol2 || (sb - u) < tol2) {
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
        counter++;
        log_value.count(carg(x), fx);
    }
    throw ComputationError(solver.getId(), "Brent: {0}: maximum number of iterations reached", log_value.chart_name);
    return 0;
}


//**************************************************************************
/// Search for a single mode starting from the given point: point
dcomplex RootBrent::find(dcomplex xstart)
{
    double f0 = NAN;
    dcomplex xprev = NAN;
    double tolx2 = params.tolx * params.tolx;

    int counter = 0;

    while (counter < params.maxiter && !(f0 <= params.tolf_min || abs2(xstart - xprev) <= tolx2)) {
        xprev = xstart;
        xstart.real(axisBrent(xstart, f0, true, counter));
        xstart.imag(axisBrent(xstart, f0, false, counter));
    }

    if (f0 > params.tolf_max)
        ComputationError(solver.getId(),
                         "Brent: {0}: After real and imaginary minimum search, determinant still not small enough",
                         log_value.chart_name);
    return xstart;
}

}}} // namespace plask::optical::slab
