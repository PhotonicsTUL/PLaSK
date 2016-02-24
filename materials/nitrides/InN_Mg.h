#ifndef PLASK__InN_Mg_H
#define PLASK__InN_Mg_H

/** @file
This file contains Mg-doped InN
*/

#include <plask/material/material.h>
#include "InN.h"

namespace plask { namespace materials {

/**
 * Represent Mg-doped InN, its physical properties.
 */
struct InN_Mg: public InN {

    static constexpr const char* NAME = "InN:Mg";

    InN_Mg(DopingAmountType Type, double Val);
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
    double NA,
           Nf_RT,
           mob_RT,
           cond_RT;

};


}} // namespace plask::materials

#endif	//PLASK__InN_Mg_H
