#ifndef PLASK__GaInAs_Be_H
#define PLASK__GaInAs_Be_H

/** @file
This file includes Be-doped GaInAs
*/

#include <plask/material/material.h>
#include "GaInAs.h"
//#include "GaAs_Be.h"
//#include "InAs_Be.h"

namespace plask {

/**
 * Represent Be-doped GaInAs, its physical properties.
 */
struct GaInAs_Be: public GaInAs {

    static constexpr const char* NAME = "GaInAs:Be";

    GaInAs_Be(const Material::Composition& Comp, DopingAmountType Type, double Val);
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

    //GaAs_Be mGaAs_Be;
    //InAs_Be mInAs_Be;
};

} // namespace plask

#endif	//PLASK__GaInAs_Be_H
