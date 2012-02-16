#ifndef PLASK__GaN_H
#define PLASK__GaN_H

/** @file
This file includes undoped GaN
*/

#include "../material.h"

namespace plask {

/**
 * Represent undoped GaN, its physical properties.
 */
struct GaN: public Material {

	virtual std::string name() const;
    virtual double cond(double T) const;
    virtual double condT(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
    virtual dcomplex Nr(double wl, double T) const;

protected:
	double Nf_RT,
		   mob_RT,
		   cond_RT,
		   condTmax_RT;	
};


} // namespace plask

#endif	//PLASK__GaN_H
