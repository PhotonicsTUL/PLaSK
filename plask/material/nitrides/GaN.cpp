#include "GaN.h"

#include "../../utils/string.h"
#include <boost/lexical_cast.hpp>

#include <cmath>

namespace plask {

GaN::GaN() {
	//M. Kuc 12.02.2012	
	//Nf_RT, mob_RT, Koblmüller G, Appl. Phys. Lett. 91 (2007) 221905
	Nf_RT = 2e16;
    mob_RT = 800;
    cond_RT = 255;
	//Mion C, App. Phys. Lett. 89 (2006) 092123
	condTmax_RT = 230;
}

std::string GaN::name() const { return ("GaN:undoped"); }

double GaN::cond(double T) const {
	//M. Kuc 12.02.2012
	//cond_RT*fun(T)
	return ( cond_RT*pow((T/300.),-0.18) ); 
}

double GaN::condT(double T, double t) const {
	//M. Kuc 12.02.2012
	//condT(t), t: 1 - 1000 um; Mion C, App. Phys. Lett. 89 (2006) 092123
	double fun_t = pow((tanh(0.001529*pow(t,0.984))),0.12);
	//condT(T), T: 300 - 450 K; Mion C, App. Phys. Lett. 89 (2006) 092123
    return( condTmax_RT*fun_t*pow((T/300.),-1.43) );
 }
 
double GaN::absp(double wl, double T) const { 
	//NO absp(T) DEPENDENCE !!!	
	//M. Kuc 12.02.2012	
	//abs(wl), wl: 380 - 450 nm; Perlin Unipress 11.2011 no published
	double A = 6.,
           B =  1.01010E-5*pow(wl,3.) - 1.32662E-2*pow(wl,2.) + 5.62769*wl - 7.76981E2,
           C = 1.333E-03,
           D = -4.46970E-4*pow(wl,3.) + 5.70108E-1*pow(wl,2.) - 2.43599E2*wl + 3.49746E4;
	//A*exp(B(wl)+C)+D(wl)
    return (A*exp(B+C)+D);	
}

double GaN::nr(double wl, double T) const { 
	//NO nr(T) DEPENDENCE !!!
	//M. Kuc 12.02.2012
	//nr(wl), wl: 355 - 1240 nm; www.rpi.edu Educational Resources (E.F. Schubert 2004)		
	double A = 1./wl;	
	double nr_wl = 4.94507E7*pow(A,3.) - 1.56053E5*pow(A,2.) + 2.25051E2*A + 2.15670;
	return ( nr_wl );
}

}       // namespace plask
