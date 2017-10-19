#ifndef PLASK__OPTICAL_EFFECTIVE_BRENT_H
#define PLASK__OPTICAL_EFFECTIVE_BRENT_H

#include "rootdigger.h"
#include "solver.h"

namespace plask { namespace optical { namespace slab {

class RootBrent: public RootDigger {

    // Write log message
    template <typename... Args>
    void writelog(LogLevel level, const std::string& msg, Args&&... args) const {
        std::string prefix = solver.getId(); prefix += ": "; prefix += log_value.chart_name; prefix += ": ";
        plask::writelog(level, prefix + msg, std::forward<Args>(args)...);
    }

  protected:

    // Value writelog
    Data2DLog<dcomplex,double> log_value;

    double axisBrent(dcomplex start, double& fx, bool real, int& counter);

  public:

    // Constructor
    RootBrent(SlabBase& solver, const function_type& val_fun, const Params& pars, const char* name):
        RootDigger(solver, val_fun, pars), log_value(solver.getId(), "modal", name, "det") {}


    dcomplex find(dcomplex start) override;
};

}}} // namespace plask::optical::slab
#endif // PLASK__OPTICAL_EFFECTIVE_BRENT_H
