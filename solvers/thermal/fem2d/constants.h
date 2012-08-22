#ifndef PLASK__MODULE_THERMAL_CONSTANTS_H
#define PLASK__MODULE_THERMAL_CONSTANTS_H

namespace plask { namespace solvers { namespace thermal {
//Physical Constants (cPhys)
namespace cPhys
{
	const double 
        //RT = 300.,					// Room Temperature
		Pi = 3.14159265,			// Pi
        Q = 1.60217733e-19;			// Elementary Charge [C]
        //M0 = 9.10938215e-31,		// electron mass [kg]
        //KB = 8.617385e-5,			// Boltzmann Constant [eV/K]
        //KB_J = 1.380658e-23,		// Boltzmann Constant [J/K] (KB_J = KB * Q)
        //Eps0_J = 8.854187817e-12,	// Permittivity Of Free Space [F/m = C^2/J/m]
        //Eps0 = 1.4185978996e-30,	// Permittivity Of Free Space [C^2/eV/m] (Eps0 = Eps0_J * Q)
        //H = 4.13566733e-15,			// Planck constant [eV * s]
        //HBar = 6.582118993e-16,		// Dirac constant [eV * s] (HBar = 0.5 * H / Pi)
        //H_J = 6.62607244e-34,		// Planck constant [J * s]
        //HBar_J = 1.054572183e-34;	// Dirac constant [J * s] (HBar_J = 0.5 * H_J / Pi)
}

}}} // namespace

#endif
