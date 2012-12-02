#ifndef PLASK__InN_Si_H
#define PLASK__InN_Si_H

/** @file
This file includes Si-doped InN
*/

#include <plask/material/material.h>
#include "InN.h"

namespace plask {

/**
 * Represent Si-doped InN, its physical properties.
 */
struct InN_Si: public InN {

    static constexpr const char* NAME = "InN:Si";

    InN_Si(DopingAmountType Type, double Val);
	virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
	virtual double Nf(double T) const; //TODO change to cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;
private:
    double ND,
           Nf_RT,
           mob_RT;

};


} // namespace plask

#endif	//PLASK__InN_Si_H
