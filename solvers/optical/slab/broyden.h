#ifndef PLASK__OPTICAL_SLAB_BROYDEN_H
#define PLASK__OPTICAL_SLAB_BROYDEN_H

#include <functional>
#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace slab {

struct RootDigger {

    typedef std::function<dcomplex(dcomplex)> function_type;

    struct Params {
        double tolx,        ///< Absolute tolerance on the argument
            tolf_min,       ///< Sufficient tolerance on the function value
            tolf_max,       ///< Required tolerance on the function value
            maxstep;        ///< Maximum step in one iteration
        int maxiter;        ///< Maximum number of iterations
        double alpha,       ///< Ensures sufficient decrease of charval in each step
            lambda_min;     ///< Minimum decreased ratio of the step (lambda)

        Params() :
            tolx(1.0e-07), tolf_min(1.0e-10), tolf_max(1.0e-8), maxstep(0.1), maxiter(500),
            alpha(1.0e-6), lambda_min(1.0e-7) {}
    };

  private:

    // Solver
    Solver& solver;

    // Solver method computing the value to zero
    function_type val_function;

    // Value writelog
    Data2DLog<dcomplex,dcomplex>& log_value;

    // Parameters for Broyden algorithm
    static constexpr double EPS = 1e6 * SMALL; ///< precision for computing Jacobian

    // Return Jacobian of F(x)
    void fdjac(dcomplex x, dcomplex F, dcomplex& Jr, dcomplex& Ji) const;

    // Search for the new point x along direction p for which f decreased sufficiently
    bool lnsearch(dcomplex& x, dcomplex& F, dcomplex g, dcomplex p, double stpmax) const;

    // Search for the root of char_val using globally convergent Broyden method
    dcomplex Broyden(dcomplex x) const;

    // Write log message
    template <typename... Args>
    void writelog(LogLevel level, const std::string& msg, Args&&... args) const {
        std::string prefix = solver.getId(); prefix += ": "; prefix += log_value.chart_name; prefix += ": ";
        plask::writelog(level, prefix + msg, std::forward<Args>(args)...);
    }

  public:

    Params par;

    // Constructor
    RootDigger(Solver& solver, const function_type& val_fun, Data2DLog<dcomplex,dcomplex>& log_value,
               const Params& par) :
        solver(solver),
        val_function(val_fun),
        log_value(log_value),
        par(par)
    {};

    /**
     * Search for a single zero starting from the given point
     *
     * \param point initial point to start search from
     * \return found solution
     */
    dcomplex operator()(dcomplex point) const;
};

}}} // namespace plask::solvers::slab
#endif // PLASK__OPTICAL_SLAB_BROYDEN_H
