#ifndef PLASK__OPTICAL_EFFECTIVE_BRENT_H
#define PLASK__OPTICAL_EFFECTIVE_BRENT_H

#include "rootdigger.hpp"
#include "solver.hpp"

namespace plask { namespace optical { namespace slab {

class RootBrent: public RootDigger {

    // Write log message
    template <typename... Args>
    void writelog(LogLevel level, const std::string& msg, Args&&... args) const {
        std::string prefix = solver.getId(); prefix += ": "; prefix += log_value.chartName(); prefix += ": ";
        plask::writelog(level, prefix + msg, std::forward<Args>(args)...);
    }

  protected:

    double axisBrent(dcomplex start, double& fx, bool real, int& counter);

  public:

    // Constructor
    RootBrent(SlabBase& solver, const function_type& val_fun, const Params& pars, const char* name):
        RootDigger(solver, val_fun, pars, name) {}


    dcomplex find(dcomplex start) override;
};

}}} // namespace plask::optical::slab
#endif // PLASK__OPTICAL_EFFECTIVE_BRENT_H
