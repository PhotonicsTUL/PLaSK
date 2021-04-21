#ifndef PLASK__OPTICAL_SLAB_MULLER_H
#define PLASK__OPTICAL_SLAB_MULLER_H

#include "rootdigger.hpp"
#include "solver.hpp"

namespace plask { namespace optical { namespace slab {

struct RootMuller: public RootDigger {

    RootMuller(SlabBase& solver, const function_type& val_fun, const Params& pars, const char* name):
        RootDigger(solver, val_fun, pars, name) {}


    dcomplex find(dcomplex start) override;
};

}}} // namespace plask::optical::slab
#endif // PLASK__OPTICAL_SLAB_MULLER_H
