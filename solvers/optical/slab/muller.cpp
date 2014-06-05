#include "muller.h"
#include "slab_base.h"
using namespace std;

namespace plask { namespace solvers { namespace slab {

//**************************************************************************
/// Search for a single mode starting from the given point: point
dcomplex RootMuller::find(dcomplex start) const
{
    dcomplex first = start - 0.5 * params.initial_dist;
    dcomplex second = start + 0.5 * params.initial_dist;

    writelog(LOG_DETAIL, "Searching for the root with Muller method between %1% and %2%", str(first), str(second));
    log_value.resetCounter();

    double xtol2 = params.tolx * params.tolx;
    double fmin2 = params.tolf_min * params.tolf_min;
    double fmax2 = params.tolf_min * params.tolf_min;

    dcomplex x2 = first, x1 = second, x0 = start;

    dcomplex f2 = val_function(x2); log_value(x2, f2);
    dcomplex f1 = val_function(x1); log_value(x1, f1);
    dcomplex f0 = val_function(x0); log_value.count(x0, f0);

    for (unsigned i = 0; i < params.maxiter; ++i) {
        if (isnan(real(f0)) || isnan(imag(f0)))
            throw ComputationError(solver.getId(), "Computed value is NaN");
        dcomplex q = (x0 - x1) / (x1 - x2);
        dcomplex A = q * f0 - q*(q+1.) * f1 + q*q * f2;
        dcomplex B = (2.*q+1.) * f0 - (q+1.)*(q+1.) * f1 + q*q * f2;
        dcomplex C = (q+1.) * f0;
        dcomplex S = sqrt(B*B - 4.*A*C);
        x2 = x1; f2 = f1;
        x1 = x0; f1 = f0;
        x0 = x1 - (x1-x2) * ( 2*C / std::max(B+S, B-S, [](const dcomplex& a, const dcomplex& b){return abs2(a) < abs2(b);}) );
        f0 = val_function(x0); log_value.count(x0, f0);
        if (abs2(f0) < fmin2 || (abs2(x0-x1) < xtol2 && abs2(f0) < fmax2)) {
            writelog(LOG_RESULT, "Found root at " + str(x0));
            return x0;
        }
    }

    throw ComputationError(solver.getId(), "Muller: %1%: maximum number of iterations reached", log_value.chart_name);
}

}}} // namespace plask::solvers::slab
