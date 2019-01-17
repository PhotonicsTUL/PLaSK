#ifndef PLASK__Ag_H
#define PLASK__Ag_H

/** @file
This file contains Ag
*/

#include "metal.h"

namespace plask { namespace materials {

/**
 * Represent Ag, its physical properties.
 */
struct Ag: public LorentzDrudeMetal {

    Ag();

    static constexpr const char* NAME = "Ag";

    virtual std::string name() const override;

  protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif //PLASK__Ag_H
