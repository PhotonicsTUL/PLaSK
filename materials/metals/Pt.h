#ifndef PLASK__Pt_H
#define PLASK__Pt_H

/** @file
This file contains Pt
*/

#include "metal.h"

namespace plask { namespace materials {

/**
 * Represent Pt, its physical properties.
 */
struct Pt: public LorentzDrudeMetal {

    Pt();

    static constexpr const char* NAME = "Pt";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;

protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif    //PLASK__Pt_H
