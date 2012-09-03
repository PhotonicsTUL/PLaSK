#ifndef PLASK__InN_Mg_H
#define PLASK__InN_Mg_H

/** @file
This file includes Mg-doped InN
*/

#include <plask/material/material.h>
#include "InN.h"

namespace plask {

/**
 * Represent Mg-doped InN, its physical properties.
 */
struct InN_Mg: public InN {

    static constexpr const char* NAME = "InN:Mg";

    InN_Mg(DopingAmountType Type, double Val);
	virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
	virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;
private:
    double NA,
           Nf_RT,
           mob_RT,
           cond_RT;

};


} // namespace plask

#endif	//PLASK__InN_Mg_H
