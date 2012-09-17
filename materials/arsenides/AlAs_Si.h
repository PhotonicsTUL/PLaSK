#ifndef PLASK__AlAs_Si_H
#define PLASK__AlAs_Si_H

/** @file
This file includes Si-doped AlAs
*/

#include <plask/material/material.h>
#include "AlAs.h"

namespace plask {

/**
 * Represent Si-doped AlAs, its physical properties.
 */
struct AlAs_Si: public AlAs {

    static constexpr const char* NAME = "AlAs:Si";

    AlAs_Si(DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
    virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;

private:
    double ND,
           Nf_RT,
           mob_RT;

};


} // namespace plask

#endif	//PLASK__AlAs_Si_H
