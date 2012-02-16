#include "GaN_Si.h"

#include "../db.h"  //MaterialsDB::Register

#include <cmath>

namespace plask {

GaN_Si::GaN_Si(DOPING_AMOUNT_TYPE Type, double Si) {
	//M. Kuc 8.02.2012	
    if (Type == CARRIER_CONCENTRATION) Nf_RT = Si;
	else Nf_RT = 0.158*pow(Si,1.039);
	//Nf_RT(Si), Si: 6e17 - 7e18 cm-3; Oshima Y, Phys. Status Solidi C 4 (2007) 2215
	//mobRT(Nf_RT), Nf_RT: 1e16 - 2e19 cm-3; based on 7 papers (1996-2007): undoped/Si-doped GaN/c-sapphire
    mob_RT = 4.164e6*pow(Nf_RT,-0.228);
}

std::string GaN_Si::name() const { return NAME; }

double GaN_Si::mob(double T) const {
	//M. Kuc 8.02.2012
	//mob(T), T: 270 - 400 K; Kusakabe K, Physica B 376-377 (2006) 520
	//mob(Nf_RT,T) = mobRT(Nf_RT)*fun(T)
    return ( mob_RT*(1.486-T*0.00162) );
}

double GaN_Si::Nf(double T) const {
	//M. Kuc 8.02.2012
	//Nf(T), T: 270 - 400 K; Kusakabe K, Physica B 376-377 (2006) 520
	//Nf(Si,T) = Nf_RT(Si)*fun(T)
	return ( Nf_RT*(-T*T*3.405e-6 + T*3.553e-3 + 0.241) );
}

double GaN_Si::cond(double T) const {
	//M. Kuc 8.02.2012
	//100*e*Nf(T)*mob(T) [S/m]
	return ( 1.602E-17*Nf(T)*mob(T) );
}

double GaN_Si::condT(double T, double t) const {
	//M. Kuc 8.02.2012
	//condT(Nf), Nf: 1e18 - 1e19 cm-3; Oshima Y, Phys. Status Solidi C 4 (2007) 2215
	double fun_Nf = 2.18*pow(Nf_RT,-0.022);	
	//condT_GaN(t,T)*fun(Nf)
    return( GaN::condT(T,t)*fun_Nf );
 }

double GaN_Si::absp(double wl, double T) const {
	//NO absp(T) DEPENDENCE !!!
	//M. Kuc 8.02.2012
	//abs(Nf), Nf: 1e18 - 5e19 cm-3; Perlin Unipress 11.2011 no published
	//abs(wl), wl: 380 - 450 nm; Perlin Unipress 11.2011 no published
	double A = 6.,
           B =  1.01010E-5*pow(wl,3.) - 1.32662E-2*pow(wl,2.) + 5.62769*wl - 7.76981E2,
           C = -1.89394E13*pow(wl,3.) + 2.54426E16*pow(wl,2.) - 1.14497E19*wl + 1.73122E21,
           D = -4.46970E-4*pow(wl,3.) + 5.70108E-1*pow(wl,2.) - 2.43599E2*wl + 3.49746E4;
	//A*exp(B(wl)+Nf(T)/C(wl))+D(wl)
    return (A*exp(B+Nf_RT/C)+D);	
}

double GaN_Si::nr(double wl, double T) const {
	//NO nr(T) DEPENDENCE !!!
	//M. Kuc 8.02.2012
	//nr(Nf, wl = 410nm), Nf: 1e18 - 5e19 cm-3; Perlin Unipress 11.2011 no published: n-type GaN:doped
	//nr_GaN(wl,T)*f(Nf)
	return ( GaN::nr(wl,T) * (1.0001-Nf_RT/1e18*1.05003e-4 ) );
}

MaterialsDB::Register<GaN_Si> materialDB_register_GaN_Si;

}       // namespace plask
