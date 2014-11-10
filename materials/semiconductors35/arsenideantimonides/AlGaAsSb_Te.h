#ifndef PLASK__AlGaAsSb_Te_H
#define PLASK__AlGaAsSb_Te_H

/** @file
This file contains Te-doped AlGaAsSb
*/

#include <plask/material/material.h>
#include "AlGaAsSb.h"

namespace plask { namespace materials {

/**
 * Represent Te-doped AlGaAsSb, its physical properties.
 */
struct AlGaAsSb_Te: public AlGaAsSb {

    static constexpr const char* NAME = "AlGaAsSb:Te";

    AlGaAsSb_Te(const Composition &Comp, DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const; //TODO Piotr: NEW virtual method (there is no Dop() in Material), maybe it should be non-virtual?
    virtual Tensor2<double> cond(double T) const override;
    virtual double nr(double wl, double T, double n = .0) const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__AlGaAsSb_Te_H