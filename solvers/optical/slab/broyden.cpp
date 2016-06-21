#include "broyden.h"
using namespace std;

namespace plask { namespace solvers { namespace slab {

//**************************************************************************
/// Search for a single mode starting from the given point: point
dcomplex RootBroyden::find(dcomplex start)
{
    writelog(LOG_DETAIL, "Searching for the root with Broyden method starting from " + str(start));
    log_value.resetCounter();
    dcomplex x = Broyden(start);
    writelog(LOG_RESULT, "Found root at " + str(x));
    return x;
}

//**************************************************************************
//**************************************************************************
//******** The Broyden globally convergent method for root finding *********
//**************************************************************************

//**************************************************************************
// Return Jacobian of F(x)
void RootBroyden::fdjac(dcomplex x, dcomplex F, dcomplex& Jr, dcomplex& Ji)
{
    double xr0 = real(x), xi0 = imag(x);
    double hr = EPS*abs(xr0), hi = EPS*abs(xi0);
    if (hr == 0.0) hr = EPS; if (hi == 0.0) hi = EPS;

    double xr1 = xr0 + hr, xi1 = xi0 + hi;
    hr = xr1 - xr0; hi = xi1 - xi0;             // trick to reduce finite precision error

    dcomplex xr = dcomplex(xr1, xi0), xi = dcomplex(xr0, xi1);
    dcomplex Fr = val_function(xr); log_value(xr, Fr);
    dcomplex Fi = val_function(xi); log_value(xi, Fi);

    Jr = (Fr - F) / hr;
    Ji = (Fi - F) / hi;
}

//**************************************************************************
// Search for the new point x along direction p for which
// functional f decreased sufficiently
// g - (approximate) gradient of 1/2(F*F), stpmax - maximum allowed step
// return true if performed step or false if could not find sufficient function decrease
bool RootBroyden::lnsearch(dcomplex& x, dcomplex& F, dcomplex g, dcomplex p, double stpmax)
{
    if (double absp=abs(p) > stpmax) p *= stpmax/absp; // Ensure step <= stpmax

    double slope = real(g)*real(p) + imag(g)*imag(p);   // slope = grad(f)*p

    // Compute the functional
    double f = 0.5 * (real(F)*real(F) + imag(F)*imag(F));

    // Remember original values
    dcomplex x0 = x;
    double f0 = f;

    double lambda = 1.0;                        // lambda parameter x = x0 + lambda*p
    double lambda1, lambda2 = 0., f2 = 0.;

    bool first = true;

    while(true) {
        if (lambda < params.lambda_min) {              // we have (possible) convergence of x
            x = x0; // f = f0;
            return false;
        }

        x = x0 + lambda*p;
        F = val_function(x);
        log_value.count(x, F);

        f = 0.5 * (real(F)*real(F) + imag(F)*imag(F));
        if (std::isnan(f)) throw ComputationError(solver.getId(), "Computed value is NaN");

        if (f < f0 + params.alpha*lambda*slope)    // sufficient function decrease
            return true;

        lambda1 = lambda;

        if (first) {                            // first backtrack
            lambda = -slope / (2. * (f-f0-slope));
            first = false;
        } else {                                // subsequent backtracks
            double rsh1 = f - f0 - lambda1*slope;
            double rsh2 = f2 - f0 - lambda2*slope;
            double a = (rsh1 / (lambda1*lambda1) - rsh2 / (lambda2*lambda2)) / (lambda1-lambda2);
            double b = (-lambda2 * rsh1 / (lambda1*lambda1) + lambda1 * rsh2 / (lambda2*lambda2))
                       / (lambda1-lambda2);

            if (a == 0.0)
                lambda = -slope/(2.*b);
            else {
                double delta = b*b - 3.*a*slope;
                if (delta < 0.0) throw ComputationError(solver.getId(), "Broyden lnsearch: roundoff problem");
                lambda = (-b + sqrt(delta)) / (3.0*a);
            }
        }

        lambda2 = lambda1; f2 = f;              // store the second last parameters

        lambda = max(lambda, 0.1*lambda1);      // guard against too fast decrease of lambda

        writelog(LOG_DETAIL, "Broyden step decreased to the fraction " + str(lambda) + " of the original step");
    }

}

//**************************************************************************
// Search for the root of char_val using globally convergent Broyden method
// starting from point x
dcomplex RootBroyden::Broyden(dcomplex x)
{
    // Compute the initial guess of the function (and check for the root)
    dcomplex F = val_function(x);
    double absF = abs(F);
    log_value.count(x, F);
    if (absF < params.tolf_min) return x;

    bool restart = true;                    // do we have to recompute Jacobian?
    bool trueJacobian;                      // did we recently update Jacobian?

    dcomplex Br, Bi;                        // Broyden matrix columns
    dcomplex dF, dx;                        // Computed dist

    dcomplex oldx, oldF;

    // Main loop
    for (int i = 0; i < params.maxiter; i++) {
        oldx = x; oldF = F;

        if (restart) {                      // compute Broyden matrix as a Jacobian
            fdjac(x, F, Br, Bi);
            restart = false;
            trueJacobian = true;
        } else {                            // update Broyden matrix
            dcomplex dB = dF - dcomplex(real(Br)*real(dx)+real(Bi)*imag(dx), imag(Br)*real(dx)+imag(Bi)*imag(dx));
            double m = (real(dx)*real(dx) + imag(dx)*imag(dx));
            Br += (dB * real(dx)) / m;
            Bi += (dB * imag(dx)) / m;
            trueJacobian = false;
        }

        // compute g ~= B^T * F
        dcomplex g = dcomplex(real(Br)*real(F)+imag(Br)*imag(F), real(Bi)*real(F)+imag(Bi)*imag(F));

        // compute p = - B**(-1) * F
        double M = real(Br)*imag(Bi) - imag(Br)*real(Bi);
        if (M == 0) throw ComputationError(solver.getId(), "Broyden matrix singular");
        dcomplex p = - dcomplex(real(F)*imag(Bi)-imag(F)*real(Bi), real(Br)*imag(F)-imag(Br)*real(F)) / M;

        // find the right step
        if (lnsearch(x, F, g, p, params.maxstep)) {   // found sufficient functional decrease
            dx = x - oldx;
            dF = F - oldF;
            if ((abs(dx) < params.tolx && abs(F) < params.tolf_max) || abs(F) < params.tolf_min)
                return x;                       // convergence!
        } else {
            if (abs(F) < params.tolf_max)       // convergence!
                return x;
            else if (!trueJacobian) {           // first try reinitializing the Jacobian
                 writelog(LOG_DETAIL, "Reinitializing Jacobian");
                restart = true;
            } else {                            // either spurious convergence (local minimum) or failure
                throw ComputationError(solver.getId(), "Broyden method failed to converge");
            }
        }
    }

    throw ComputationError(solver.getId(), "Broyden: maximum number of iterations reached");
}

}}} // namespace plask::solvers::slab
