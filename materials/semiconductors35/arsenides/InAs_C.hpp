#ifndef PLASK__InAs_C_H
#define PLASK__InAs_C_H

/** @file
This file contains C-doped InAs
*/

#include "plask/material/material.hpp"
#include "InAs.hpp"

namespace plask { namespace materials {

/**
 * Represent C-doped InAs, its physical properties.
 */
struct InAs_C: public InAs {

    static constexpr const char* NAME = "InAs:C";

    InAs_C(double Val);
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
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__InAs_C_H
