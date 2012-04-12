#ifndef PLASK__AlGaN_Mg_H
#define PLASK__AlGaN_Mg_H

/** @file
This file includes undoped AlGaN_Mg
*/

#include <plask/material/material.h>
#include "AlGaN.h"
#include "GaN_Mg.h"
#include "AlN_Mg.h"

namespace plask {

/**
 * Represent undoped AlGaN_Mg, its physical properties.
 */
struct AlGaN_Mg: public AlGaN {

    static constexpr const char* NAME = "AlGaN:Mg";

    AlGaN_Mg(const Material::Composition& Comp, DopingAmountType Type, double Val);
    ~AlGaN_Mg();

    virtual std::string name() const;    
    virtual double mob(double T) const;
    virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual double cond(double T) const;
    virtual double absp(double wl, double T) const;

private:
    double NA,
           Nf_RT;

    GaN_Mg *mGaN_Mg;
    AlN_Mg *mAlN_Mg;
};

} // namespace plask

#endif	//PLASK__AlGaN_Mg_H
