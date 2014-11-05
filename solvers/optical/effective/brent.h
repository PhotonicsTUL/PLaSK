#ifndef PLASK__OPTICAL_EFFECTIVE_MULLER_H
#define PLASK__OPTICAL_EFFECTIVE_MULLER_H

#include "rootdigger.h"

namespace plask { namespace solvers { namespace effective {

class RootBrent: public RootDigger {

    // Write log message
    template <typename... Args>
    void writelog(LogLevel level, const std::string& msg, Args&&... args) const {
        std::string prefix = solver.getId(); prefix += ": "; prefix += log_value.chart_name; prefix += ": ";
        plask::writelog(level, prefix + msg, std::forward<Args>(args)...);
    }

  protected:

    double axisBrent(double a, double b, std::function<double(double)>fun, double& fx);

  public:

    // Constructor
    RootBrent(Solver& solver, const function_type& val_fun, Data2DLog<dcomplex,dcomplex>& log_value,
               const Params& pars): RootDigger(solver, val_fun, log_value, pars) {}


    dcomplex find(dcomplex start) const;
};

}}} // namespace plask::solvers::effective
#endif // PLASK__OPTICAL_EFFECTIVE_MULLER_H
