#ifndef PLASK__GaInAs_Sn_H
#define PLASK__GaInAs_Sn_H

/** @file
This file contains Sn-doped GaInAs
*/

#include <plask/material/material.h>
#include "GaInAs.h"
//#include "GaAs_Sn.h"
//#include "InAs_Sn.h"

namespace plask { namespace materials {

/**
 * Represent Sn-doped GaInAs, its physical properties.
 */
struct GaInAs_Sn: public GaInAs {

    static constexpr const char* NAME = "InGaAs:Sn";

    GaInAs_Sn(const Material::Composition& Comp, double Val);
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
    double ND,
           Nf_RT,
           mob_RT;

    //GaAs_Sn mGaAs_Sn;
    //InAs_Sn mInAs_Sn;
};

}} // namespace plask::materials

#endif	//PLASK__GaInAs_Sn_H
