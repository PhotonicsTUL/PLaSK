#ifndef PLASK__OPTICAL_EFFECTIVE_BRENT_H
#define PLASK__OPTICAL_EFFECTIVE_BRENT_H

#include "rootdigger.hpp"

namespace plask { namespace optical { namespace effective {

class RootBrent: public RootDigger {

    // Write log message
    template <typename... Args>
    void writelog(LogLevel level, const std::string& msg, Args&&... args) const {
        std::string prefix = solver.getId(); prefix += ": "; prefix += log_value.chart_name; prefix += ": ";
        plask::writelog(level, prefix + msg, std::forward<Args>(args)...);
    }

  protected:

    double axisBrent(dcomplex start, double& fx, bool real) const;

  public:

    // Constructor
    RootBrent(Solver& solver, const function_type& val_fun, Data2DLog<dcomplex,dcomplex>& log_value,
               const Params& pars): RootDigger(solver, val_fun, log_value, pars) {}


    dcomplex find(dcomplex start) const override;
};

}}} // namespace plask::optical::effective
#endif // PLASK__OPTICAL_EFFECTIVE_BRENT_H