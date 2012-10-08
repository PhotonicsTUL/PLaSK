#ifndef PLASK__GaInAs_C_H
#define PLASK__GaInAs_C_H

/** @file
This file includes C-doped GaInAs
*/

#include <plask/material/material.h>
#include "GaInAs.h"
//#include "GaAs_C.h"
//#include "InAs_C.h"

namespace plask {

/**
 * Represent C-doped GaInAs, its physical properties.
 */
struct GaInAs_C: public GaInAs {

    static constexpr const char* NAME = "GaInAs:C";

    GaInAs_C(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
    virtual double Nf(double T) const; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual double absp(double wl, double T) const;

private:
    double NA,
           Nf_RT,
           mob_RT;

    //GaAs_C mGaAs_C;
    //InAs_C mInAs_C;
};

} // namespace plask

#endif	//PLASK__GaInAs_C_H
