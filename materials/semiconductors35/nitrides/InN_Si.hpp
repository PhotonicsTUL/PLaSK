#ifndef PLASK__InN_Si_H
#define PLASK__InN_Si_H

/** @file
This file contains Si-doped InN
*/

#include "plask/material/material.hpp"
#include "InN.hpp"

namespace plask { namespace materials {

/**
 * Represent Si-doped InN, its physical properties.
 */
struct InN_Si: public InN {

    static constexpr const char* NAME = "InN:Si";

    InN_Si(double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO change to cm^(-3)
    double Na() const override;
    double Nd() const override;
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

#endif	//PLASK__InN_Si_H
