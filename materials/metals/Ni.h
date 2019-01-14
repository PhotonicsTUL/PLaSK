#ifndef PLASK__Ni_H
#define PLASK__Ni_H

/** @file
This file contains Ni
*/

#include "metal.h"

namespace plask { namespace materials {

/**
 * Represent Ni, its physical properties.
 */
struct Ni: public LorentzDrudeMetal {

    Ni();

    static constexpr const char* NAME = "Ni";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;

protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif    //PLASK__Ni_H
