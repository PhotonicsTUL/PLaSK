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
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual ConductivityType condtype() const override;
    virtual double nr(double lam, double T, double n = .0) const override;
    virtual double absp(double lam, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__GaSb_Si_H
