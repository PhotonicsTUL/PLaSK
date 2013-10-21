#ifndef PLASK__GaSb_Si_H
#define PLASK__GaSb_Si_H

/** @file
This file contains Si-doped GaSb
*/

#include <plask/material/material.h>
#include "GaSb.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped GaSb, its physical properties.
 */
struct GaSb_Si: public GaSb {

    static constexpr const char* NAME = "GaSb:Si";

    GaSb_Si(DopingAmountType Type, double Val);
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
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__GaSb_Si_H
