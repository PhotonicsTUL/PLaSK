#include "InN_Si.h"

#include "../db.h"  //MaterialsDB::Register
#include <cmath>

namespace plask {

InN_Si::InN_Si(DopingAmountType Type, double Si) {
	//M. Kuc 12.02.2012	
	Nf_RT = Si;
	//mobRT(Nf_RT), Nf_RT: 2e18 - 7e20 cm-3; based on 6 papers (2005-2010): undoped/Si-doped InN/c-sapphire
    mob_RT = 2.753e13*pow(Nf_RT,-0.559);
}

std::string InN_Si::name() const { return NAME; }

double InN_Si::mob(double T) const { 
	//M. Kuc 12.02.2012	
	//mob(T), T: 300 - 400 K; Hwang E S, J. Korean Phys. Soc. 48 (2006) 93
	//mob(Nf_RT,T) = mobRT(Nf_RT)*fun(T)
    return ( mob_RT*(T*T*5.174E-6 -T*5.241E-3 +2.107) );
}

double InN_Si::Nf(double T) const {
	//M. Kuc 8.02.2012
	//Nf(T), T: 300 - 400 K; Hwang E S, J. Korean Phys. Soc. 48 (2006) 93
	//Nf(Si,T) = Nf_RT(Si)*fun(T)
	return ( Nf_RT*(-T*T*3.802E-6 +T*3.819E-3 +0.1965) );
}

double InN_Si::cond(double T) const {
	//M. Kuc 12.02.2012
	//100*e*Nf(T)*mob(T) [S/m]
	return ( 1.602E-17*Nf(T)*mob(T) ); 
}
 
//double InN_Si::absp(double wl, double T) const { }
//double InN_Si::nr(double wl, double T) const { }

MaterialsDB::Register<InN_Si> materialDB_register_InN_Si;

}       // namespace plask
