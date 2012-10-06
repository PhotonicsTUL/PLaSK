#ifndef PLASK__GaInAs_Si_H
#define PLASK__GaInAs_Si_H

/** @file
This file includes Si-doped GaInAs
*/

#include <plask/material/material.h>
#include "GaInAs.h"
//#include "GaAs_Si.h"
//#include "InAs_Si.h"

namespace plask {

/**
 * Represent Si-doped GaInAs, its physical properties.
 */
struct GaInAs_Si: public GaInAs {

    static constexpr const char* NAME = "GaInAs:Si";

    GaInAs_Si(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
    virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual double absp(double wl, double T) const;

private:
    double ND,
           Nf_RT,
           mob_RT;

    //GaAs_Si mGaAs_Si;
    //InAs_Si mInAs_Si;
};

} // namespace plask

#endif	//PLASK__GaInAs_Si_H
