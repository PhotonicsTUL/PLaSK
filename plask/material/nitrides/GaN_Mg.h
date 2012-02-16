#ifndef PLASK__GaN_Mg_H
#define PLASK__GaN_Mg_H

/** @file
This file includes Mg-doped GaN
*/

#include "../material.h"
#include "GaN.h"

namespace plask {

/**
 * Represent Mg-doped GaN, its physical properties.
 */
struct GaN_Mg: public GaN {

    static constexpr const char* NAME = "GaN:Mg";

	GaN_Mg(DOPING_AMOUNT_TYPE Type, double Mg);
	virtual std::string name() const;
    virtual double mob(double T) const;
	virtual double Nf(double T) const;
    virtual double cond(double T) const;
    virtual double nr(double wl, double T) const; //NO Temperature dependence
    virtual double absp(double wl, double T) const; //NO Temperature dependence	//NO interband p-type dependence

private:
	double Nf_RT,
		   mob_RT,
		   cond_RT;
	
};


} // namespace plask

#endif	//PLASK__GaN_Mg_H
