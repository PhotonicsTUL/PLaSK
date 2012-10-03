#ifndef PLASK__MODULE_ELECTRICAL_CONSTANTS_H
#define PLASK__MODULE_ELECTRICAL_CONSTANTS_H

namespace plask { namespace solvers { namespace electrical {

namespace cPhys // physical constants
{
	const double 
        //pi = 3.14159265,    // pi
        q = 1.60217733e-19, // elementary charge [C]
        c = 299792458, // the speed of light [m/s]
        h = 6.626068e-34; // Planck's constant [J*s]
}

}}} // namespace

#endif
