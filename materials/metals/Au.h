#ifndef PLASK__Au_H
#define PLASK__Au_H

/** @file
This file contains Au
*/

#include "metal.h"

namespace plask { namespace materials {

/**
 * Represent Au, its physical properties.
 */
struct Au: public LorentzDrudeMetal {

    Au();

    static constexpr const char* NAME = "Au";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;

  protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif    //PLASK__Au_H
