#ifndef PLASK__InSb_Si_H
#define PLASK__InSb_Si_H

/** @file
This file contains Si-doped InSb
*/

#include "plask/material/material.hpp"
#include "InSb.hpp"

namespace plask { namespace materials {

/**
 * Represent Si-doped InSb, its physical properties.
 */
struct InSb_Si: public InSb {

    static constexpr const char* NAME = "InSb:Si";

    InSb_Si(double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__InSb_Si_H
