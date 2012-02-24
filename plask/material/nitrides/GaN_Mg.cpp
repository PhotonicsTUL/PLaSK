#include "GaN_Mg.h"

#include "../db.h"  //MaterialsDB::Register

#include <cmath>

namespace plask {

GaN_Mg::GaN_Mg(DopingAmountType Type, double Mg) {
	//M. Kuc 12.02.2012
    if (Type == CARRIER_CONCENTRATION) Nf_RT = Mg;
	else Nf_RT = 1.676E2*pow(Mg,0.7925);
	//Nf_RT(Mg), Mg: 1e19 - 8e20 cm-3; based on 4 papers (1998-2008): MBE-grown Mg-doped GaN
	//mobRT(Nf_RT), Nf_RT: 2e17 - 6e18 cm-3; based on 9 papers (2000-2009): MBE-grown Mg-doped GaN
    mob_RT = 25.747*exp(-9.034E-19*Nf_RT);
	cond_RT = 1.602E-17*Nf_RT*mob_RT;
}

std::string GaN_Mg::name() const { return NAME; }

double GaN_Mg::mob(double T) const { 
	//M. Kuc 12.02.2012	
	//mob(T), T: 300 - 400 K; Kozodoy P, J. Appl. Phys. 87 (2000) 1832
	//mob(Nf_RT,T) = mobRT(Nf_RT)*fun(T)
    return ( mob_RT * (T*T*2.495E-5 -T*2.268E-2 +5.557) );
}

double GaN_Mg::Nf(double T) const {
	//M. Kuc 12.02.2012
	//Nf(T), T: 300 - 400 K; Kozodoy P, J. Appl. Phys. 87 (2000) 1832
	//Nf(Mg,T) = Nf_RT(Mg)*fun(T)
	return ( Nf_RT * (T*T*2.884E-4 -T*0.147 + 19.080) );
}

double GaN_Mg::cond(double T) const {
	//M. Kuc 12.02.2012
	//100*e*Nf(T)*mob(T) [S/m]
	return ( 1.602E-17*Nf(T)*mob(T) );  
}

double GaN_Mg::absp(double wl, double T) const { 
	//M. Kuc 12.02.2012
	//NO absp(T) DEPENDENCE !!!	
	//NO interband p-type DEPENDENCE !!!	
    return ( GaN::absp(wl,T) );
}

double GaN_Mg::nr(double wl, double T) const { 
	//NO nr(T) DEPENDENCE !!!
	//M. Kuc 12.02.2012	
    return ( GaN::nr(wl,T) );
}

static MaterialsDB::Register<GaN_Mg> materialDB_register_Mg;

}       // namespace plask
