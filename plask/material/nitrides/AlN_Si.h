#ifndef PLASK__AlN_Si_H
#define PLASK__ALN_Si_H

/** @file
This file includes Si-doped AlN
*/

#include "../material.h"
#include "AlN.h"

namespace plask {

/**
 * Represent Si-doped AlN, its physical properties.
 */
struct AlN_Si: public AlN {

    static constexpr const char* NAME = "AlN:Si";

    AlN_Si(DopingAmountType Type, double Val);
	virtual std::string name() const;
    virtual double mob(double T) const;
	virtual double Nf(double T) const;
    virtual double cond(double T) const;

private:
    double ND,
           Nf_RT,
		   mob_RT;

};


} // namespace plask

#endif	//PLASK__AlN_Si_H
