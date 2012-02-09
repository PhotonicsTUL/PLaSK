#ifndef PLASK__GaN_Si_H
#define PLASK__GaN_Si_H

/** @file
This file includes Si-doped GaN
*/

#include "../material.h"

namespace plask {

/**
 * Represent Si-doped GaN, its physical properties.
 */
struct GaN_Si: public Material {

	GaN_Si(DOPING_AMOUNT_TYPE Type, double Si);
	virtual std::string name() const;
    virtual double mob(double T) const;
	virtual double Nf(double T) const;
    virtual double cond(double T) const;
    virtual double condT(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
    virtual dcomplex Nr(double wl, double T) const;

protected:
	double Nf_RT,
		   mob_RT;
	
};


} // namespace plask

#endif	//PLASK__GaN_Si_H
