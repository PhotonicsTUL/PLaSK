#ifndef PLASK__InN_H
#define PLASK__InN_H

/** @file
This file includes undoped InN
*/

#include <plask/material/material.h>


namespace plask {

/**
 * Represent undoped InN, its physical properties.
 */
struct InN: public Semiconductor {

    static constexpr const char* NAME = "InN";

    virtual std::string name() const;
    virtual double condT(double T) const;
};


} // namespace plask

#endif	//PLASK__InN_H
