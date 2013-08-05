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

    static constexpr const char* NAME = "GaInAs:Sn";

    GaInAs_Sn(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
    virtual double Nf(double T) const; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double ND,
           Nf_RT,
           mob_RT;

    //GaAs_Sn mGaAs_Sn;
    //InAs_Sn mInAs_Sn;
};

}} // namespace plask::materials

#endif	//PLASK__GaInAs_Sn_H
