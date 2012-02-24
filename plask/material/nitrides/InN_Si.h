#ifndef PLASK__InN_Si_H
#define PLASK__InN_Si_H

/** @file
This file includes Si-doped InN
*/

#include "../material.h"
#include "InN.h"

namespace plask {

/**
 * Represent Si-doped InN, its physical properties.
 */
struct InN_Si: public InN {

    static constexpr const char* NAME = "InN:Si";

	InN_Si(DopingAmountType Type, double Si);
	virtual std::string name() const;
    virtual double mob(double T) const;
	virtual double Nf(double T) const;
    virtual double cond(double T) const;
    //virtual double nr(double wl, double T) const;
    //virtual double absp(double wl, double T) const;
private:
	double Nf_RT,
		   mob_RT,
		   condTmax_RT;
	
};


} // namespace plask

#endif	//PLASK__InN_Si_H
