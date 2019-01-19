#ifndef PLASK__Al_H
#define PLASK__Al_H

/** @file
This file contains Al
*/

#include "metal.h"

namespace plask { namespace materials {

/**
 * Represent Al, its physical properties.
 */
struct Al: public LorentzDrudeMetal {

    Al();

    static constexpr const char* NAME = "Al";

    virtual std::string name() const override;

  protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif //PLASK__Al_H
