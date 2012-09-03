#ifndef PLASK__InGaN_Mg_H
#define PLASK__InGaN_Mg_H

/** @file
This file includes undoped InGaN_Mg
*/

#include <plask/material/material.h>
#include "InGaN.h"
#include "GaN_Mg.h"
#include "InN_Mg.h"

namespace plask {

/**
 * Represent undoped InGaN_Mg, its physical properties.
 */
struct InGaN_Mg: public InGaN {

    static constexpr const char* NAME = "InGaN:Mg";

    InGaN_Mg(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
    virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual double absp(double wl, double T) const;

protected:
    double NA,
           Nf_RT;

    GaN_Mg mGaN_Mg;
    InN_Mg mInN_Mg;

};


} // namespace plask

#endif	//PLASK__InGaN_Mg_H
