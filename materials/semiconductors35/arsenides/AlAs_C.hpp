#ifndef PLASK__AlAs_C_H
#define PLASK__AlAs_C_H

/** @file
This file contains C-doped AlAs
*/

#include "plask/material/material.hpp"
#include "AlAs.hpp"

namespace plask { namespace materials {

/**
 * Represent C-doped AlAs, its physical properties.
 */
struct AlAs_C: public AlAs {

    static constexpr const char* NAME = "AlAs:C";

    AlAs_C(double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__AlAs_C_H
