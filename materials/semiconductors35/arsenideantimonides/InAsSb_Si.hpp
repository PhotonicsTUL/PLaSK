#ifndef PLASK__InAsSb_Te_H
#define PLASK__InAsSb_Te_H

/** @file
This file contains Si-doped InAsSb
*/

#include "plask/material/material.hpp"
#include "InAsSb.hpp"
//#include "InAs_Si.hpp"
//#include "InSb_Si.hpp"

namespace plask { namespace materials {

/**
 * Represent Si-doped InAsSb, its physical properties.
 */
struct InAsSb_Si: public InAsSb {

    static constexpr const char* NAME = "InAsSb:Si";

    InAsSb_Si(const Material::Composition& Comp, double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double nr(double lam, double T, double n = .0) const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

    //InAs_Si mInAs_Si;
    //InSb_Si mInSb_Si;
};

}} // namespace plask::materials

#endif	//PLASK__InAsSb_Si_H
