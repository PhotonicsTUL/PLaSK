#ifndef PLASK__InAs_C_H
#define PLASK__InAs_C_H

/** @file
This file includes C doped InAs
*/

#include <plask/material/material.h>
#include "InAs.h"

namespace plask {

/**
 * Represent C doped InAs, its physical properties.
 */
struct InAs_C: public InAs {

    static constexpr const char* NAME = "InAs:C";

    InAs_C(DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
    virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;

private:
    double NA,
           Nf_RT,
           mob_RT;

};


} // namespace plask

#endif	//PLASK__InAs_C_H
