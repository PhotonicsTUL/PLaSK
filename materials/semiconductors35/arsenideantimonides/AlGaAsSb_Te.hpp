#ifndef PLASK__AlGaAsSb_Te_H
#define PLASK__AlGaAsSb_Te_H

/** @file
This file contains Te-doped AlGaAsSb
*/

#include "plask/material/material.hpp"
#include "AlGaAsSb.hpp"

namespace plask { namespace materials {

/**
 * Represent Te-doped AlGaAsSb, its physical properties.
 */
struct AlGaAsSb_Te: public AlGaAsSb {

    static constexpr const char* NAME = "AlGaAsSb:Te";

    AlGaAsSb_Te(const Composition &Comp, double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    double doping() const override; //TODO Piotr: NEW method (there is no doping() in Material), maybe it should be non-virtual?
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

};

}} // namespace plask::materials

#endif	//PLASK__AlGaAsSb_Te_H
