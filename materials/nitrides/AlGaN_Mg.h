#ifndef PLASK__AlGaN_Mg_H
#define PLASK__AlGaN_Mg_H

/** @file
This file contains Mg-doped AlGaN
*/

#include <plask/material/material.h>
#include "AlGaN.h"
#include "GaN_Mg.h"
#include "AlN_Mg.h"

namespace plask { namespace materials {

/**
 * Represent Mg-doped AlGaN, its physical properties.
 */
struct AlGaN_Mg: public AlGaN {

    static constexpr const char* NAME = "AlGaN:Mg";

    AlGaN_Mg(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO change to cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double NA,
           Nf_RT;

    GaN_Mg mGaN_Mg;
    AlN_Mg mAlN_Mg;
};

}} // namespace plask::materials

#endif	//PLASK__AlGaN_Mg_H
