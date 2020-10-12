#ifndef PLASK__GaSb_Si_H
#define PLASK__GaSb_Si_H

/** @file
This file contains Si-doped GaSb
*/

#include "plask/material/material.hpp"
#include "GaSb.hpp"

namespace plask { namespace materials {

/**
 * Represent Si-doped GaSb, its physical properties.
 */
struct GaSb_Si: public GaSb {

    static constexpr const char* NAME = "GaSb:Si";

    GaSb_Si(double Val);
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
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__GaSb_Si_H
