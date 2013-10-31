#ifndef PLASK__OPTICAL_EFFECTIVE_MULLER_H
#define PLASK__OPTICAL_EFFECTIVE_MULLER_H

#include <functional>
#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace effective {

struct RootMuller {

    typedef std::function<dcomplex(dcomplex)> function_type;

    struct Params {
        double tolx,        ///< Absolute tolerance on the argument
            tolf_min,       ///< Sufficient tolerance on the function value
            tolf_max;       ///< Required tolerance on the function value
        unsigned maxiter;   ///< Maximum number of iterations

        Params() :
            tolx(1.0e-07), tolf_min(1.0e-10), tolf_max(1.0e-8), maxiter(500) {}
    };

  private:

    // Solver
    Solver& solver;

    // Solver method computing the value to zero
    function_type val_function;

    // Value writelog
    Data2DLog<dcomplex,dcomplex>& log_value;

    // Write log message
    template <typename... Args>
    void writelog(LogLevel level, const std::string& msg, Args&&... args) const {
        std::string prefix = solver.getId(); prefix += ": "; prefix += log_value.chart_name; prefix += ": ";
        plask::writelog(level, prefix + msg, std::forward<Args>(args)...);
    }

  public:

    Params params;

    // Constructor
    RootMuller(Solver& solver, const function_type& val_fun, Data2DLog<dcomplex,dcomplex>& log_value,
               const Params& params) :
        solver(solver),
        val_function(val_fun),
        log_value(log_value),
        params(params)
    {};

    /**
     * Search for a single zero starting from the given point
     *
     * \param first first initial point to start search from
     * \param second second initial point to start search from
     * \return found solution
     */
    dcomplex operator()(dcomplex first, dcomplex second) const;
};

}}} // namespace plask::solvers::effective
#endif // PLASK__OPTICAL_EFFECTIVE_MULLER_H
