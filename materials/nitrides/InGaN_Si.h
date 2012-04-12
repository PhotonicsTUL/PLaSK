#ifndef PLASK__InGaN_Si_H
#define PLASK__InGaN_Si_H

/** @file
This file includes undoped InGaN_Si
*/

#include <plask/material/material.h>
#include "InGaN.h"
#include "GaN_Si.h"
#include "InN_Si.h"

namespace plask {

/**
 * Represent undoped InGaN_Si, its physical properties.
 */
struct InGaN_Si: public InGaN {

    static constexpr const char* NAME = "InGaN:Si";

    InGaN_Si(const Material::Composition& Comp, DopingAmountType Type, double Val);
    ~InGaN_Si();

    virtual std::string name() const;
    virtual double mob(double T) const;
    virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual double cond(double T) const;
    virtual double condT(double T, double t) const;
    virtual double absp(double wl, double T) const;

protected:
    double ND,
           Nf_RT;

    GaN_Si *mGaN_Si;
    InN_Si *mInN_Si;

};


} // namespace plask

#endif	//PLASK__InGaN_Si_H
