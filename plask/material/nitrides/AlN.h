#ifndef PLASK__AlN_H
#define PLASK__AlN_H

/** @file
This file includes undoped AlN
*/

#include "../material.h"

namespace plask {

/**
 * Represent undoped AlN, its physical properties.
 */
struct AlN: public Semiconductor {

    static constexpr const char* NAME = "AlN";

	AlN();
	virtual std::string name() const;
    virtual double condT(double T, double t) const;
    //virtual double nr(double wl, double T) const;
    //virtual double absp(double wl, double T) const;

private:
	double condTmax_RT;
};


} // namespace plask

#endif	//PLASK__AlN_H
