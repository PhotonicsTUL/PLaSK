#ifndef PLASK__GaN_Si_H
#define PLASK__GaN_Si_H

/** @file
This file includes Si-doped GaN
*/

#include <plask/material/material.h>
#include "GaN.h"

namespace plask {

/**
 * Represent Si-doped GaN, its physical properties.
 */
struct GaN_Si: public GaN {

    static constexpr const char* NAME = "GaN:Si";

    GaN_Si(DopingAmountType Type, double Val);
	virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
	virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual std::pair<double,double> thermCond(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

private:
    double ND,
           Nf_RT,
           mob_RT;

};


} // namespace plask

#endif	//PLASK__GaN_Si_H
