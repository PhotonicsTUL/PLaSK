#ifndef PLASK__GaAs_Si_H
#define PLASK__GaAs_Si_H

/** @file
This file includes Si-doped GaAs
*/

#include <plask/material/material.h>
#include "GaAs.h"

namespace plask {

/**
 * Represent Si-doped GaAs, its physical properties.
 */
struct GaAs_Si: public GaAs {

    static constexpr const char* NAME = "GaAs:Si";

    GaAs_Si(DopingAmountType Type, double Val);
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

};

} // namespace plask

#endif	//PLASK__GaAs_Si_H
