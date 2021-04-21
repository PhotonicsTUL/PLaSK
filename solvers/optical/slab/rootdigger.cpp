#include "rootdigger.hpp"
#include "solverbase.hpp"

namespace plask { namespace optical { namespace slab {

RootDigger::RootDigger(SlabBase& solver, const function_type& val_fun, const Params& pars, const char* name) :
    solver(solver),
    val_function(val_fun),
    params(pars),
    log_value(solver.getId(), "modal", name, "det")
{}

}}} // namespace plask::optical::slab
