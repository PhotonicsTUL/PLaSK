#ifndef PLASK__AlGaN_Si_H
#define PLASK__AlGaN_Si_H

/** @file
This file includes Si-doped AlGaN
*/

#include <plask/material/material.h>
#include "AlGaN.h"
#include "GaN_Si.h"
#include "AlN_Si.h"

namespace plask {

/**
 * Represent Si-doped AlGaN, its physical properties.
 */
struct AlGaN_Si: public AlGaN {

    static constexpr const char* NAME = "AlGaN:Si";

    AlGaN_Si(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
    virtual double Nf(double T) const; //TODO change to cm^(-3)
    virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual std::pair<double,double> thermk(double T, double t) const;
    virtual double absp(double wl, double T) const;

private:
    double ND,
           Nf_RT;

    GaN_Si mGaN_Si;
    AlN_Si mAlN_Si;
};

} // namespace plask

#endif	//PLASK__AlGaN_Si_H
