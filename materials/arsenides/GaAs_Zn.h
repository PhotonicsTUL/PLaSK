#ifndef PLASK__GaAs_Zn_H
#define PLASK__GaAs_Zn_H

/** @file
This file includes Zn-doped GaAs
*/

#include <plask/material/material.h>
#include "GaAs.h"

namespace plask {

/**
 * Represent Zn-doped GaAs, its physical properties.
 */
struct GaAs_Zn: public GaAs {

    static constexpr const char* NAME = "GaAs:Zn";

    GaAs_Zn(DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
    virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual double absp(double wl, double T) const;

private:
    double NA,
           Nf_RT,
           mob_RT;

};

} // namespace plask

#endif	//PLASK__GaAs_Zn_H
