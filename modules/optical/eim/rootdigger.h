#ifndef ROOTDIGGER_H
#define ROOTDIGGER_H

#include <plask/plask.hpp>

namespace plask { namespace eim {

class EffectiveIndex2dModule;

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

    // Compute the function value
    dcomplex value(dcomplex x, bool count=true) const;

  public:

    // Constructors
    RootDigger(EffectiveIndex2dModule& module) : module(module),
        alpha(1.0e-4),                                          // ensures sufficient decrease of charval in each step
        lambda_min(1.0e-7),                                     // minimum decreased ratio of the step (lambda)
        EPS(sqrt(std::numeric_limits<double>::epsilon()))       // square root of machine precission
    {};

    RootDigger(const RootDigger& d) = default;

    /**
     * Look for the minima map browsing through given points
     *
     * \param repoints list of points to browse in real domain
     * \param impoints list of points to browse in imaginary domain
     * \return potential values close to minima
     */
    std::vector<dcomplex> findMap(std::vector<double> repoints, std::vector<double> impoints) const;

    /**
     * Search for zeros within the region between \a start and \a end
     *
     * \param start start of the range
     * \param end end of the range
     * \param replot number of segments in real domain to divide the range into
     * \param implot number of segments in imaginary domain to divide the range into
     * \param num_modes maximum number of modes to look for
     * \return list of found solutions
     */
    std::vector<dcomplex> searchSolutions(dcomplex start, dcomplex end, int replot, int implot, int num_modes);

    /**
     * Search for a single zero starting from the given point
     *
     * \param point initial point to start search from
     * \return found solution
     */
    dcomplex getSolution(dcomplex point) const;
};

}} // namespace plask::eim
#endif // ROTDIGGER_H
