#ifndef PLASK__OPTICAL_SIMPLE_OPTICAL_MULLER_H
#define PLASK__OPTICAL_SIMPLE_OPTICAL_MULLER_H

#include "rootdigger.h"

namespace plask { namespace optical { namespace simple_optical {

class RootMuller: public RootDigger {

    // Write log message
    template <typename... Args>
    void writelog(LogLevel level, const std::string& msg, Args&&... args) const {
        std::string prefix = solver.getId(); prefix += ": "; prefix += log_value.chart_name; prefix += ": ";
        plask::writelog(level, prefix + msg, std::forward<Args>(args)...);
    }

  public:

    // Constructor
    RootMuller(Solver& solver, const function_type& val_fun, Data2DLog<dcomplex,dcomplex>& log_value,
               const Params& pars): RootDigger(solver, val_fun, log_value, pars) {}


    dcomplex find(dcomplex start) const override;
};

}}} // namespace plask::optical::simple_optical
#endif // PLASK__OPTICAL_SIMPLE_OPTICAL_MULLER_H