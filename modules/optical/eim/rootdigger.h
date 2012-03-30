#ifndef ROOTDIGGER_H
#define ROOTDIGGER_H

#include <plask/plask.hpp>
#include "eim.hpp"

namespace plask { namespace eim {

class RootDigger {

    EffectiveIndex2dModule& module;

  protected:
    // Parameters for Broyden algorithm
    const double alpha, lambda_min, EPS;

    // Return Jacobian of F(x)
    void fdjac(dcomplex x, dcomplex F, dcomplex& Jr, dcomplex& Ji) const;

    // Search for the new point x along direction p for which f decreased sufficiently
    bool lnsearch(dcomplex& x, dcomplex& F, dcomplex g, dcomplex p, double stpmax) const;

    // Search for the root of char_val using globally convergent Broyden method
    dcomplex Broyden(dcomplex x) const;

    // Look for map browsing through given points
    std::vector<dcomplex> find_map(std::vector<double> repoints, std::vector<double> impoints) const;

    // Compute the function value
    dcomplex value(dcomplex x, bool count=true) const {
        dcomplex y = module.char_val(x);
        double ay = abs(y);
        if (count) module.log_value.count(x, ay);
        else module.log_value(x, ay);
        return y;
    }

  public:
    /// Parameters for Broyden algorithm
    double tolx, tolf_min, tolf_max, maxstep;

    /// Maximum number of iterations
    int maxiterations;

    // Constructors
    RootDigger(EffectiveIndex2dModule& module) : module(module),
        maxiterations(500),                              // maximum number of iterations
        tolx(1.0e-07),                              // absolute tolerance on the argument
        tolf_min(1.0e-12),                          // sufficient tolerance on the function value
        tolf_max(1.0e-10),                          // required tolerance on the function value
        maxstep(0.1),                              // maximum step in one iteration
        alpha(1.0e-4),                              // ensures sufficient decrease of charval in each step
        lambda_min(1.0e-7),                         // minimum decreased ratio of the step (lambda)
        EPS(sqrt(std::numeric_limits<double>::epsilon()))// square root of machine precission
    {};

    RootDigger(const RootDigger& d) = default;

    /// Search for modes within the region real(start) - real(end),
    //   imag(start) - imag(end), divided on: replot for real direction and implot for imaginary one
    //   search for single mode defined by the number: modeno and allmodes==false
    //   search for the given number of modes: number of the searched modes is modeno and allmodes==true
    //   return vector of complex coordinates of the modes
    //   return 0 size vector if the mode has not been found
    std::vector<dcomplex> searchModes(dcomplex start, dcomplex end, int replot, int implot, int num_modes);

    /// Search for a single mode starting from the given point: point
    // return complex coordinate of the mode
    // return 0 if the mode has not been found
    dcomplex getMode(dcomplex point) const;
};

}} // namespace plask::eim
#endif // ROTDIGGER_H
