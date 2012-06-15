#include "broyden.h"
#include "eim.h"
using namespace std;

namespace plask { namespace modules { namespace eim {

vector<dcomplex> RootDigger::findMap(vector<double> repoints, vector<double> impoints) const
{

    // The number of points in each direction
    int NR = repoints.size();
    int NI = impoints.size();

    module.writelog(LOG_DETAIL, "Searching for the solutions map using %1% points", NR*NI);

    // Handle situations with inconvenient number of points in some direction
    // (this is not perfect but we must handle it somehow)
    if (NR == 0) throw BadInput(module.getId(), "At least one point in real domain required to browse for a map");
    if (NI == 0) { impoints = vector<double>(1, 0.0); NI = 1; }
    else if (NI  == 2) { impoints = vector<double>(1, 0.5*(impoints[0]+impoints[1])); NI = 1; }
    if (NR  == 2) { repoints = vector<double>(1, 0.5*(repoints[0]+repoints[1])); NR = 1; }
    if (NR == 1 && NI == 1) return vector<dcomplex>(1, repoints[0] + impoints[0]*I);

    // Create space for char_val values in points
    double** values = new double*[NR];
    for (int i = 0; i < NR; i++)
        values[i] = new double[NI];

    // Compute the values at points
    for (int r = 0; r < NR; r++)
        for (int i = 0; i < NI; i++) {
        try {
                dcomplex x = dcomplex(repoints[r], impoints[i]);
                dcomplex y = val_function(x);
                log_value(x, values[r][i] = abs(y));
        } catch(...) {
            values[r][i] = NAN;
            //TODO: print warning on screen and handle it in minima search
        }
        if (std::isnan(values[r][i])) writelog(LOG_WARNING, "Computed value is NaN in map search");
    }

    vector<dcomplex> results;

    // Now browse them to find local minima
    // (this method never will find anything at the boundaries)
    if (NR == 1) {
        for (int i = 0; i < NI; i++)
            if (values[0][i] < values[0][i-1] && values[0][i] < values[0][i+1])
                results.push_back(repoints[0] + impoints[i]*I);
    } else if (NI == 1) {
        for (int r = 1; r < NR-1; r++)
            if (values[r][0] < values[r-1][0] && values[r][0] < values[r+1][0])
                results.push_back(repoints[r] + impoints[0]*I);
    } else {
        for (int r = 1; r < NR-1; r++)
            for (int i = 1; i < NI-1; i++)
                if (values[r][i] < values[r-1][i] && values[r][i] < values[r+1][i] &&
                       values[r][i] < values[r][i-1] && values[r][i] < values[r][i+1])
                    results.push_back(repoints[r] + impoints[0]*I);
    }

    // Free space for char_val values in points
    for (int r = 0; r < NR; r++)
        delete[] values[r];
    delete[] values;

    // Log found results
    std::stringstream resultsrt;
    resultsrt << "Found map values: ";
    for (auto map = results.begin(); map != results.end(); map++) {
        if (map != results.begin()) resultsrt << ", ";
        resultsrt << str(*map);
    }
    module.writelog(LOG_RESULT, resultsrt.str());

    return results;
}

//**************************************************************************
/// Look for the minima map browsing through given points
std::vector<dcomplex> RootDigger::findMap(plask::dcomplex start, plask::dcomplex end, int replot, int implot)
{
    if (imag(start) == imag(end)) implot = 0;
    if (real(start) == real(end)) replot = 0;

    vector<double> repoints(replot+1), impoints(implot+1);
    double restep = 0, imstep = 0;

    // Set points for map search
    if (replot != 0) {
        double restart = real(start); restep = (real(end) - restart) / replot;
        for (int i = 0; i <= replot; i++)
            repoints[i] = restart + i*restep;
    } else {
        repoints[0] = 0.5 * (real(start) + real(end));
    }
    if (implot != 0) {
        double imstart = imag(start); imstep = (imag(end) - imstart) / implot;
        for (int i = 0; i <= implot; i++)
            impoints[i] = imstart + i*imstep;
    } else {
        impoints[0] = 0.5 * (imag(start) + imag(end));
    }

    return findMap(repoints, impoints);
}


//**************************************************************************
/// Search for solutions within the region real(start) - real(end),
std::vector< dcomplex > RootDigger::searchSolutions(plask::dcomplex start, plask::dcomplex end, int replot, int implot, int num_modes)
{
    vector<dcomplex> modes;

    // Determine map
    vector<dcomplex> map = findMap(start, end, replot, implot);

    // Find solutions starting from the map points
    int iend = min(int(map.size()), num_modes);
    for (int i = 0; i < iend; i++) {
        try {
            dcomplex mode = getSolution(map[i]);
            modes.push_back(mode);
        } catch (runtime_error err) {
            module.writelog(LOG_ERROR, "Failed to get solution around " + str(map[i]) + " (" + err.what() + ")");
        };
    }

    return modes;
}

//**************************************************************************
/// Search for a single mode starting from the given point: point
dcomplex RootDigger::getSolution(dcomplex point) const
{
    module.writelog(LOG_DETAIL, "Searching for the solution with Broyden method starting from " + str(point));
    log_value.resetCounter();
    dcomplex x = Broyden(point);
    module.writelog(LOG_RESULT, "Found solution at " + str(x));
    return x;
}




//**************************************************************************
//**************************************************************************
//******** The Broyden globally convergent method for root finding *********
//**************************************************************************

//**************************************************************************
// Return Jacobian of F(x)
void RootDigger::fdjac(dcomplex x, dcomplex F, dcomplex& Jr, dcomplex& Ji) const
{
    double xr0 = real(x), xi0 = imag(x);
    double hr = EPS*abs(xr0), hi = EPS*abs(xi0);
    if (hr == 0.0) hr = EPS; if (hi == 0.0) hi = EPS;

    double xr1 = xr0 + hr, xi1 = xi0 + hi;
    hr = xr1 - xr0; hi = xi1 - xi0;             // trick to reduce finite precision error

    dcomplex xr = dcomplex(xr1, xi0), xi = dcomplex(xr0, xi1);
    dcomplex Fr = val_function(xr),
             Fi = val_function(xi);
    log_value(xr, abs(Fr));
    log_value(xi, abs(Fi));

    Jr = (Fr - F) / hr;
    Ji = (Fi - F) / hi;
}

//**************************************************************************
// Search for the new point x along direction p for which
// functional f decreased sufficiently
// g - (approximate) gradient of 1/2(F*F), stpmax - maximum allowed step
// return true if performed step or false if could not find sufficient function decrease
bool RootDigger::lnsearch(dcomplex& x, dcomplex& F, dcomplex g, dcomplex p, double stpmax) const
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
        if (lambda < lambda_min) {              // we have (possible) convergence of x
            x = x0; f = f0;
            return false;
        }

        x = x0 + lambda*p;
        F = val_function(x);

        f = 0.5 * (real(F)*real(F) + imag(F)*imag(F));
        if (std::isnan(f)) throw ComputationError(module.getId(), "Computed value is NaN");

        if (f < f0 + alpha*lambda*slope) {      // sufficient function decrease
            log_value.count(x, abs(F));
            return true;
        }
        log_value(x, abs(F));

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
                if (delta < 0.0) throw ComputationError(module.getId(), "Broyden lnsearch: roundoff problem");
                lambda = (-b + sqrt(delta)) / (3.0*a);
            }
        }


        lambda2 = lambda1; f2 = f;              // store the second last parameters

        lambda = max(lambda, 0.1*lambda1);      // guard against too fast decrease of lambda

        module.writelog(LOG_DETAIL, "Broyden step decreased to the fraction " + str(lambda) + " of the original step");
    }

}

//**************************************************************************
// Search for the root of char_val using globally convergent Broyden method
// starting from point x
dcomplex RootDigger::Broyden(dcomplex x) const
{
    // Compute the initial guess of the function (and check for the root)
    dcomplex F = val_function(x);
    double absF = abs(F);
    log_value.count(x, absF);
    if (absF < tolf_min) return x;

    bool restart = true;                    // do we have to recompute Jacobian?
    bool trueJacobian;                      // did we recently update Jacobian?

    dcomplex Br, Bi;                        // Broyden matrix columns
    dcomplex dF, dx;                        // Computed shift

    dcomplex oldx, oldF;

    // Main loop
    for (int i = 0; i < maxiterations; i++) {
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
        if (M == 0) throw ComputationError(module.getId(), "Broyden matrix singular");
        dcomplex p = - dcomplex(real(F)*imag(Bi)-imag(F)*real(Bi), real(Br)*imag(F)-imag(Br)*real(F)) / M;

        // find the right step
        if (lnsearch(x, F, g, p, maxstep)) {   // found sufficient functional decrease
            dx = x - oldx;
            dF = F - oldF;
            if ((abs(dx) < tolx && abs(F) < tolf_max) || abs(F) < tolf_min)
                return x;                       // convergence!
        } else {
            if (abs(F) < tolf_max)       // convergence!
                return x;
            else if (!trueJacobian) {           // first try reinitializing the Jacobian
                 module.writelog(LOG_DETAIL, "Reinitializing Jacobian");
                restart = true;
            } else {                            // either spurious convergence (local minimum) or failure
                throw ComputationError(module.getId(), "Broyden method failed to converge");
            }
        }
    }

    throw ComputationError(module.getId(), "Broyden: maximum number of iterations reached");
}

}}} // namespace plask::modules::eim
