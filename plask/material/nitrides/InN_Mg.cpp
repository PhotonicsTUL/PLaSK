#include "InN_Mg.h"

#include "../../utils/string.h"
#include <boost/lexical_cast.hpp>

#include <cmath>

namespace plask {

InN_Mg::InN_Mg(DOPING_AMOUNT_TYPE Type, double Mg) {
	//M. Kuc 12.02.2012	
    if (Type == CARRIER_CONCENTRATION) Nf_RT = Mg;
	else Nf_RT = 7.392E9*pow(Mg,0.439);
	//Nf_RT(Mg), Mg: 3e18 - 8e20 cm-3; based on 2 papers (2008-2009): Mg-doped InN
	//mobRT(Nf_RT), Nf_RT: 1e18 - 1e19 cm-3; based on 4 papers (2006-2010): MBE-grown Mg-doped InN
    mob_RT = 5.739E13*pow(Nf_RT,-0.663);
	cond_RT = 1.602E-17*Nf_RT*mob_RT;
}

std::string InN_Mg::name() const { return ("InN:Mg"); }

double InN_Mg::mob(double T) const { 
	//M. Kuc 12.02.2012
	//No T Dependence based on Kumakura K, J. Appl. Phys. 93 (2003) 3370
    return ( mob_RT );
}

double InN_Mg::Nf(double T) const {
	//M. Kuc 12.02.2012
	//No T Dependence based on Kumakura K, J. Appl. Phys. 93 (2003) 3370
	return ( Nf_RT );
}

double InN_Mg::cond(double T) const {
	//M. Kuc 12.02.2012
	return ( cond_RT ); 
}

//double InN_Mg::absp(double wl, double T) const { }
//double InN_Mg::nr(double wl, double T) const { }

}       // namespace plask
