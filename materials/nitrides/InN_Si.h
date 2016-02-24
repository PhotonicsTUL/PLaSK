#ifndef PLASK__InN_Si_H
#define PLASK__InN_Si_H

/** @file
This file contains Si-doped InN
*/

#include <plask/material/material.h>
#include "InN.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped InN, its physical properties.
 */
struct InN_Si: public InN {

    static constexpr const char* NAME = "InN:Si";

    InN_Si(DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO change to cm^(-3)
    virtual double Na() const override;
    virtual double Nd() const override;
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

};


}} // namespace plask::materials

#endif	//PLASK__InN_Si_H
