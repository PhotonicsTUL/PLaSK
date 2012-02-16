#include "AlN_Si.h"

#include <cmath>
#include "../db.h"  //MaterialsDB::Register

namespace plask {

AlN_Si::AlN_Si(DOPING_AMOUNT_TYPE Type, double Si) {
	//M. Kuc 12.02.2012	
    if (Type == CARRIER_CONCENTRATION) Nf_RT = Si;
	else Nf_RT = 6.197E-19*pow(Si,1.805);
	//Nf_RT(Si), based on 2 papers (2004-2008): Si-doped AlN
	//mobRT(Nf_RT), based on 4 papers (2004-2008): Si-doped AlN
    mob_RT = 29.410*exp(-1.838E-17*Nf_RT);
}

std::string AlN_Si::name() const { return NAME; }

double AlN_Si::mob(double T) const { 
	//M. Kuc 12.02.2012	
	//mob(T), T: 270 - 400 K; Kusakabe K, Physica B 376-377 (2006) 520
	//mob(Nf_RT,T) = mobRT(Nf_RT)*fun(T)
    return ( mob_RT * (1.486 -T*0.00162) );
}

double AlN_Si::Nf(double T) const {
	//M. Kuc 12.02.2012
	//Nf(T), T: 300 - 400 K; Taniyasu Y, Nature Letters 44 (2006) 325
	//Nf(Si,T) = Nf_RT(Si)*fun(T)
	return ( Nf_RT * 3.502E-27*pow(T,10.680) );
}

double AlN_Si::cond(double T) const {
	//M. Kuc 12.02.2012
	//100*e*Nf(T)*mob(T) [S/m]
	return ( 1.602E-17*Nf(T)*mob(T) ); 
}

//double AlN_Si::absp(double wl, double T) const { }
//double AlN_Si::nr(double wl, double T) const { }

static MaterialsDB::Register<AlN_Si> materialDB_register_AlN_Si;

}       // namespace plask
