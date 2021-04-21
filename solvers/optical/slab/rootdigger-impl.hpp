#ifndef PLASK__OPTICAL_SLAB_ROOTDIGGER_IMPL_H
#define PLASK__OPTICAL_SLAB_ROOTDIGGER_IMPL_H

#include "rootdigger.hpp"
#include "solverbase.hpp"

namespace plask { namespace optical { namespace slab {

template <typename... Args>
void RootDigger::writelog(LogLevel level, const std::string& msg, Args&&... args) const {
    std::string prefix = solver.getId(); prefix += ": "; prefix += log_value.chartName(); prefix += ": ";
    plask::writelog(level, prefix + msg, std::forward<Args>(args)...);
}

}}} // namespace plask::optical::slab

#endif // PLASK__OPTICAL_SLAB_ROOTDIGGER_IMPL_H
