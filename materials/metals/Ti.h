#ifndef PLASK__Ti_H
#define PLASK__Ti_H

/** @file
This file contains Ti
*/

#include "metal.h"

namespace plask { namespace materials {

/**
 * Represent Ti, its physical properties.
 */
struct Ti: public LorentzDrudeMetal {

    Ti();

    static constexpr const char* NAME = "Ti";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;

protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif    //PLASK__Ti_H
