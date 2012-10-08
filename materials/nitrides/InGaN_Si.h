#ifndef PLASK__InGaN_Si_H
#define PLASK__InGaN_Si_H

/** @file
This file includes Si-doped InGaN
*/

#include <plask/material/material.h>
#include "InGaN.h"
#include "GaN_Si.h"
#include "InN_Si.h"

namespace plask {

/**
 * Represent Si-doped InGaN, its physical properties.
 */
struct InGaN_Si: public InGaN {

    static constexpr const char* NAME = "InGaN:Si";

    InGaN_Si(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
    virtual double Nf(double T) const; //TODO change to cm^(-3)
    virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual std::pair<double,double> thermCond(double T, double t) const;
    virtual double absp(double wl, double T) const;

protected:
    double ND,
           Nf_RT;

    GaN_Si mGaN_Si;
    InN_Si mInN_Si;

};


} // namespace plask

#endif	//PLASK__InGaN_Si_H
