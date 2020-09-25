#include "plask_material_function.hpp"

namespace plask { namespace solvers { namespace optical_fdtd {

PlaskMaterialFunction::PlaskMaterialFunction(plask::solvers::optical_fdtd::FDTDSolver* solver) : solver(solver) {}

// This is the function which returns the index
double PlaskMaterialFunction::chi1p1(meep::field_type, const meep::vec& r) { return solver->eps(r); }

double PlaskMaterialFunction::eps(const meep::vec& r) { return solver->eps(r); }

bool PlaskMaterialFunction::has_mu() { return false; }

bool PlaskMaterialFunction::has_conductivity(meep::component c) { return true; }

double PlaskMaterialFunction::mu(const meep::vec& r) { return 1.; }

double PlaskMaterialFunction::conductivity(meep::component, const meep::vec& r) { return solver->conductivity(r); }

void PlaskMaterialFunction::sigma_row(meep::component c, double sigrow[3], const meep::vec& r) {
    sigrow[0] = sigrow[1] = sigrow[2] = 0.0;
    sigrow[component_index(c)] = solver->eps(r);
}

double PlaskMaterialFunction::chi3(meep::component, const meep::vec& r) { return 0.; }

double PlaskMaterialFunction::chi2(meep::component, const meep::vec& r) { return 0.; }

}}}  // namespace plask::solvers::optical_fdtd