#ifndef PLASK__AlGaAs_C_H
#define PLASK__AlGaAs_C_H

/** @file
This file contains C-doped AlGaAs
*/

#include "plask/material/material.hpp"
#include "AlGaAs.hpp"
#include "GaAs_C.hpp"
#include "AlAs_C.hpp"

namespace plask { namespace materials {

/**
 * Represent C-doped AlGaAs, its physical properties.
 */
struct AlGaAs_C: public AlGaAs {

    static constexpr const char* NAME = "AlGaAs:C";

    AlGaAs_C(const Material::Composition& Comp, double Val);
    std::string name() const override;
    std::string str() const override;
    double EactA(double T) const override;
    double EactD(double T) const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    double Na() const override;
    double Nd() const override;
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double NA,
           Nf_RT,
           mob_RT;

    GaAs_C mGaAs_C;
    AlAs_C mAlAs_C;
};

}} // namespace plask::materials

#endif	//PLASK__AlGaAs_C_H
