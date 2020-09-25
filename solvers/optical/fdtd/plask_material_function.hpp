//#pragma once // Is it a good idea?
#ifndef PLASK_MATERIAL_FUNCTION_H
#define PLASK_MATERIAL_FUNCTION_H

#include <meep.hpp>
#include "fdtd.hpp"

namespace plask { namespace solvers { namespace optical_fdtd {

class FDTDSolver;

class PlaskMaterialFunction : public meep::material_function {
    FDTDSolver* solver; 

    public:
        PlaskMaterialFunction(FDTDSolver* solver);
        double chi1p1(meep::field_type, const meep::vec &r) override;
        double eps(const meep::vec &r) override;
        bool has_mu() override;
        bool has_conductivity(meep::component c) override;
        double mu(const meep::vec &r) override;
        double conductivity(meep::component, const meep::vec &r) override;
        void sigma_row(meep::component c, double sigrow[3], const meep::vec &r) override;
        double chi3(meep::component, const meep::vec &r) override;
        double chi2(meep::component, const meep::vec &r) override;
};
}}}
#endif