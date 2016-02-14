#ifndef PLASK__InGaN_Mg_H
#define PLASK__InGaN_Mg_H

/** @file
This file contains Mg-doped InGaN
*/

#include <plask/material/material.h>
#include "InGaN.h"
#include "GaN_Mg.h"
#include "InN_Mg.h"

namespace plask { namespace materials {

/**
 * Represent Mg-doped InGaN, its physical properties.
 */
struct InGaN_Mg: public InGaN {

    static constexpr const char* NAME = "InGaN:Mg";

    InGaN_Mg(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
    virtual double Nf(double T) const; //TODO change to cm^(-3)
    virtual double Na() const override;
    virtual double Nd() const override;
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

protected:
    double NA,
           Nf_RT;

    GaN_Mg mGaN_Mg;
    InN_Mg mInN_Mg;

};


}} // namespace plask::materials

#endif	//PLASK__InGaN_Mg_H
