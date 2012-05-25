#ifndef ROOTDIGGER_H
#define ROOTDIGGER_H

#include <functional>
#include <plask/plask.hpp>

namespace plask { namespace modules { namespace eim {

class EffectiveIndex2dModule;

struct RootDigger {

    typedef std::function<dcomplex(const dcomplex&)> function_type;

  private:

    // Module
    EffectiveIndex2dModule& module;

    // Module method computing the value to zero
    function_type val_function;

    // Value logger
    Data2dLog<dcomplex,double>& log_value;

    // Parameters for Broyden algorithm
    constexpr static double EPS         ///< Square root of machine precission
                 = sqrt(std::numeric_limits<double>::epsilon());

    // Return Jacobian of F(x)
    void fdjac(dcomplex x, dcomplex F, dcomplex& Jr, dcomplex& Ji) const;

    // Search for the new point x along direction p for which f decreased sufficiently
    bool lnsearch(dcomplex& x, dcomplex& F, dcomplex g, dcomplex p, double stpmax) const;

    // Search for the root of char_val using globally convergent Broyden method
    dcomplex Broyden(dcomplex x) const;

    // Compute the function value
    dcomplex value(dcomplex x, bool count=true) const;

  public:

    double tolx,        ///< Absolute tolerance on the argument
           tolf_min,    ///< Sufficient tolerance on the function value
           tolf_max,    ///< Required tolerance on the function value
           maxstep;     ///< Maximum step in one iteration
    int maxiterations;  ///< Maximum number of iterations
    double alpha,       ///< Ensures sufficient decrease of charval in each step
           lambda_min;  ///< Minimum decreased ratio of the step (lambda)

    // Constructor
    RootDigger(EffectiveIndex2dModule& module, const function_type& val_fun, Data2dLog<dcomplex,double>& log_value,
               double tolx=1.0e-07, double tolf_min=1.0e-10, double tolf_max=1.0e-8, double maxstep=0.1, int maxiterations=500,
               double alpha=1.0e-6, double lambda_min=1.0e-7) :
        module(module),
        val_function(val_fun),
        log_value(log_value),
        tolx(tolx),
        tolf_min(tolf_min),
        tolf_max(1.0e-8),
        maxstep(maxstep),
        maxiterations(maxiterations),
        alpha(alpha),
        lambda_min(lambda_min)
    {};

    /**
     * Look for the minima map browsing through given points
     *
     * \param repoints list of points to browse in real domain
     * \param impoints list of points to browse in imaginary domain
     * \return potential values close to minima
     */
    std::vector<dcomplex> findMap(std::vector<double> repoints, std::vector<double> impoints) const;

    /**
     * Look for the minima map browsing through given points
     *
     * \param start start of the range
     * \param end end of the range
     * \param replot number of segments in real domain to divide the range into
     * \param implot number of segments in imaginary domain to divide the range into
     * \param num_modes maximum number of modes to look for
     * \return potential values close to minima
     */
    std::vector<dcomplex> findMap(dcomplex start, dcomplex end, int replot, int implot);

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

}}} // namespace plask::modules::eim
#endif // ROTDIGGER_H
