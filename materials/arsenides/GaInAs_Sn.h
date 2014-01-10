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

    GaInAs_Sn(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

    //GaAs_Sn mGaAs_Sn;
    //InAs_Sn mInAs_Sn;
};

}} // namespace plask::materials

#endif	//PLASK__GaInAs_Sn_H
