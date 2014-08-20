#include "rootdigger.h"
#include "muller.h"
#include "broyden.h"

namespace plask { namespace solvers { namespace effective {

std::unique_ptr<RootDigger> RootDigger::get(Solver* solver, const function_type& func, Data2DLog<dcomplex,dcomplex>& detlog, const Params& params) {
    typedef std::unique_ptr<RootDigger> Res;
    if (params.method == RootDigger::ROOT_MULLER) return Res(new RootMuller(*solver, func, detlog, params));
    else if (params.method == RootDigger::ROOT_BROYDEN) return Res(new RootBroyden(*solver, func, detlog, params));
    throw BadInput(solver->getId(), "Wrong root finding method");
    return Res();
}

void RootDigger::readRootDiggerConfig(XMLReader& reader, Params& params) {
    params.tolx = reader.getAttribute<double>("tolx", params.tolx);
    params.tolf_min = reader.getAttribute<double>("tolf-min", params.tolf_min);
    params.tolf_max = reader.getAttribute<double>("tolf-max", params.tolf_max);
    params.maxstep = reader.getAttribute<double>("maxstep", params.maxstep);
    params.maxiter = reader.getAttribute<int>("maxiter", params.maxiter);
    params.alpha = reader.getAttribute<double>("alpha", params.alpha);
    params.lambda_min = reader.getAttribute<double>("lambda", params.lambda_min);
    params.initial_dist = reader.getAttribute<dcomplex>("initial-range", params.initial_dist);
    params.method = reader.enumAttribute<RootDigger::Method>("method")
        .value("broyden", RootDigger::ROOT_BROYDEN)
        .value("muller", RootDigger::ROOT_MULLER)
        .get(params.method);
    reader.requireTagEnd();
}

}}} // namespace plask::solvers::effective
