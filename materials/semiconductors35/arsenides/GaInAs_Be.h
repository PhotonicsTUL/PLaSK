#ifndef PLASK__GaInAs_Be_H
#define PLASK__GaInAs_Be_H

/** @file
This file contains Be-doped GaInAs
*/

#include <plask/material/material.h>
#include "GaInAs.h"
//#include "GaAs_Be.h"
//#include "InAs_Be.h"

namespace plask { namespace materials {

/**
 * Represent Be-doped GaInAs, its physical properties.
 */
struct GaInAs_Be: public GaInAs {

    static constexpr const char* NAME = "InGaAs:Be";

    GaInAs_Be(const Material::Composition& Comp, double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
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

    //GaAs_Be mGaAs_Be;
    //InAs_Be mInAs_Be;
};

}} // namespace plask::materials

#endif	//PLASK__GaInAs_Be_H
