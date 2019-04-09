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

    InGaN_Mg(const Material::Composition& Comp, double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO change to cm^(-3)
    double Na() const override;
    double Nd() const override;
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

protected:
    double NA,
           Nf_RT;

    GaN_Mg mGaN_Mg;
    InN_Mg mInN_Mg;

};


}} // namespace plask::materials

#endif	//PLASK__InGaN_Mg_H
