#ifndef PLASK__GaInAs_Zn_H
#define PLASK__GaInAs_Zn_H

/** @file
This file contains Zn-doped GaInAs
*/

#include <plask/material/material.h>
#include "GaInAs.h"
//#include "GaAs_Zn.h"
//#include "InAs_Zn.h"

namespace plask { namespace materials {

/**
 * Represent Zn-doped GaInAs, its physical properties.
 */
struct GaInAs_Zn: public GaInAs {

    static constexpr const char* NAME = "InGaAs:Zn";

    GaInAs_Zn(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual ConductivityType condtype() const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double NA,
           Nf_RT,
           mob_RT;

    //GaAs_Zn mGaAs_Zn;
    //InAs_Zn mInAs_Zn;
};

}} // namespace plask::materials

#endif	//PLASK__GaInAs_Zn_H
