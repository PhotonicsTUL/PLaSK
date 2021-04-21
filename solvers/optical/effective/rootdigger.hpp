#ifndef PLASK__OPTICAL_EFFECTIVE_ROOTDIGGER_H
#define PLASK__OPTICAL_EFFECTIVE_ROOTDIGGER_H

#include <functional>
#include <plask/plask.hpp>

namespace plask { namespace optical { namespace effective {

struct RootDigger {

    typedef std::function<dcomplex(dcomplex)> function_type;

    /// Root finding method
    enum Method {
        ROOT_MULLER,
        ROOT_BROYDEN,
        ROOT_BRENT,
    };

    struct Params {
        Method method;          ///< Root finding method
        double tolx,            ///< Absolute tolerance on the argument
            tolf_min,           ///< Sufficient tolerance on the function value
            tolf_max,           ///< Required tolerance on the function value
            maxstep;            ///< Maximum step in one iteration
        int maxiter;            ///< Maximum number of iterations
        double alpha,           ///< Ensures sufficient decrease of determinant in each step
            lambda_min;         ///< Minimum decrease ratio of the step (lambda)
        dcomplex initial_dist;  ///< Distance between initial points
        int stairs;             ///< Number of staircase iterations in Brent algorithm

        Params():
            method(ROOT_BROYDEN),
            tolx(1e-6),
            tolf_min(1e-7),
            tolf_max(1e-5),
            maxstep(0.1),
            maxiter(500),
            alpha(1e-7),
            lambda_min(1e-8),
            initial_dist(1e-3),
            stairs(3)
            {}
    };

  protected:

    // Solver
    Solver& solver;

    // Solver method computing the value to zero
    function_type val_function;

    // Value writelog
    DataLog<dcomplex,dcomplex>& log_value;

    inline dcomplex valFunction(dcomplex x) const {
        try {
            return val_function(x);
        } catch (...) {
            log_value.throwError(x);
        }
    }

  public:

    // Rootdigger parameters
    Params params;

    // Constructor
    RootDigger(Solver& solver, const function_type& val_fun, DataLog<dcomplex,dcomplex>& log_value,
               const Params& pars) :
        solver(solver),
        val_function(val_fun),
        log_value(log_value),
        params(pars)
    {};

    virtual ~RootDigger() {}

    /**
     * Search for a single zero starting from the given point
     * \param start initial point to start search from
     * \return found solution
     */
    virtual dcomplex find(dcomplex start) const = 0;

    /**
     * Get root digger for given function and params
     * \param func function to find zero of
     * \param detlog output logger
     * \param params rootdigger params
     * \return unique pointer to rootdigger
     */
    static std::unique_ptr<RootDigger> get(Solver* solver, const function_type& func, DataLog<dcomplex,dcomplex>& detlog, const Params& params);

    /// Read configuration from xml
    static void readRootDiggerConfig(XMLReader& reader, Params& params);
};

}}} // namespace plask::optical::effective

#endif // PLASK__OPTICAL_EFFECTIVE_ROOTDIGGER_H
