#ifndef PLASK__InGaN_Si_H
#define PLASK__InGaN_Si_H

/** @file
This file contains Si-doped InGaN
*/

#include "plask/material/material.hpp"
#include "InGaN.hpp"
#include "GaN_Si.hpp"
#include "InN_Si.hpp"

namespace plask { namespace materials {

/**
 * Represent Si-doped InGaN, its physical properties.
 */
struct InGaN_Si: public InGaN {

    static constexpr const char* NAME = "InGaN:Si";

    InGaN_Si(const Material::Composition& Comp, double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO change to cm^(-3)
    double Na() const override;
    double Nd() const override;
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    Tensor2<double> thermk(double T, double t) const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

protected:
    double ND,
           Nf_RT;

    GaN_Si mGaN_Si;
    InN_Si mInN_Si;

};


}} // namespace plask::materials

#endif	//PLASK__InGaN_Si_H
