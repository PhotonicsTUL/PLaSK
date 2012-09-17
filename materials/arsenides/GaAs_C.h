#ifndef PLASK__GaAs_C_H
#define PLASK__GaAs_C_H

/** @file
This file includes C doped GaAs
*/

#include <plask/material/material.h>
#include "GaAs.h"

namespace plask {

/**
 * Represent C doped GaAs, its physical properties.
 */
struct GaAs_C: public GaAs {

    static constexpr const char* NAME = "GaAs:C";

    GaAs_C(DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
    //virtual double Nf(double T) const;
    //virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual double absp(double wl, double T) const;

private:
    double NA,
           Nf_RT,
           mob_RT;

};


} // namespace plask

#endif	//PLASK__GaAs_C_H
