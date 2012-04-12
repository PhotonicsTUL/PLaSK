#ifndef PLASK__GaN_Mg_H
#define PLASK__GaN_Mg_H

/** @file
This file includes Mg-doped GaN
*/

#include <plask/material/material.h>
#include "GaN.h"

namespace plask {

/**
 * Represent Mg-doped GaN, its physical properties.
 */
struct GaN_Mg: public GaN {

    static constexpr const char* NAME = "GaN:Mg";

    GaN_Mg(DopingAmountType Type, double Val);
	virtual std::string name() const;
    virtual double mob(double T) const;
	virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual double cond(double T) const;
    virtual double absp(double wl, double T) const;

private:
    double NA,
           Nf_RT,
		   mob_RT,
		   cond_RT;

};


} // namespace plask

#endif	//PLASK__GaN_Mg_H
