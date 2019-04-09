#ifndef PLASK__AlGaAs_Si_H
#define PLASK__AlGaAs_Si_H

/** @file
This file contains Si-doped AlGaAs
*/

#include <plask/material/material.h>
#include "AlGaAs.h"
#include "GaAs_Si.h"
#include "AlAs_Si.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped AlGaAs, its physical properties.
 */
struct AlGaAs_Si: public AlGaAs {

    static constexpr const char* NAME = "AlGaAs:Si";

    AlGaAs_Si(const Material::Composition& Comp, double Val);
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
    double ND,
           Nf_RT,
           mob_RT;

    GaAs_Si mGaAs_Si;
    AlAs_Si mAlAs_Si;
};

}} // namespace plask::materials

#endif	//PLASK__AlGaAs_Si_H
