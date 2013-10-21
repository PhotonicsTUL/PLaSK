#ifndef PLASK__GaSb_Te_H
#define PLASK__GaSb_Te_H

/** @file
This file contains Te-doped GaSb
*/

#include <plask/material/material.h>
#include "GaSb.h"

namespace plask { namespace materials {

/**
 * Represent Te-doped GaSb, its physical properties.
 */
struct GaSb_Te: public GaSb {

    static constexpr const char* NAME = "GaSb:Te";

    GaSb_Te(DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
    virtual double Nf(double T) const; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;
    virtual double nr(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double ND,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__GaAs_Si_H
