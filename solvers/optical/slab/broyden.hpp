#ifndef PLASK__OPTICAL_SLAB_BROYDEN_H
#define PLASK__OPTICAL_SLAB_BROYDEN_H

#include "rootdigger.hpp"
#include "solver.hpp"

namespace plask { namespace optical { namespace slab {

class RootBroyden: public RootDigger {

    // Parameters for Broyden algorithm
    static constexpr double EPS = 1e6 * SMALL; ///< precision for computing Jacobian

    // Return Jacobian of F(x)
    void fdjac(dcomplex x, dcomplex F, dcomplex& Jr, dcomplex& Ji);

    // Search for the new point x along direction p for which f decreased sufficiently
    bool lnsearch(dcomplex& x, dcomplex& F, dcomplex g, dcomplex p, double stpmax);

    // Search for the root of char_val using globally convergent Broyden method
    dcomplex Broyden(dcomplex x);

    // Write log message
    template <typename... Args>
    void writelog(LogLevel level, const std::string& msg, Args&&... args) const {
        std::string prefix = solver.getId(); prefix += ": "; prefix += log_value.chart_name; prefix += ": ";
        plask::writelog(level, prefix + msg, std::forward<Args>(args)...);
    }

  protected:

    // Value writelog
    Data2DLog<dcomplex,dcomplex> log_value;

  public:

    // Constructor
    RootBroyden(SlabBase& solver, const function_type& val_fun, const Params& pars, const char* name):
        RootDigger(solver, val_fun, pars), log_value(solver.getId(), "modal", name, "det") {}

    dcomplex find(dcomplex start) override;
};

}}} // namespace plask::optical::slab
#endif // PLASK__OPTICAL_SLAB_BROYDEN_H
