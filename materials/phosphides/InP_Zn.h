#ifndef PLASK__InP_Zn_H
#define PLASK__InP_Zn_H

/** @file
This file includes Zn-doped InP
*/

#include <plask/material/material.h>
#include "InP.h"

namespace plask {

/**
 * Represent Zn-doped InP, its physical properties.
 */
struct InP_Zn: public InP {

    static constexpr const char* NAME = "InP:Zn";

    InP_Zn(DopingAmountType Type, double Val);
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

#endif	//PLASK__InP_Zn_H
