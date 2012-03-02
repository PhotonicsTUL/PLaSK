#include "AlN.h"

#include "../db.h"  //MaterialsDB::Register
#include "../info.h"    //MaterialInfo::DB::Register
#include <cmath>

namespace plask {

AlN::AlN() {
	// M. Kuc 12.02.2012
	// Slack G A, J. Phys. Chem. Sol. 48 (1987) 641
	condTmax_RT = 285;
}

std::string AlN::name() const { NAME; }

MI_PROPERTY(AlN, condT,
			MISource("M. Kuc 12.02.2012"),
			MISource("condT(T), T: Bondokov R T, J. Crystal Growth 310 (2008) 4020"),
			MIComment("estimation based on Si-doped GaN and AlN data"))
double AlN::condT(double T, double t) const {
	//M. Kuc 12.02.2012
	//estimation based on Si-doped GaN and AlN data
	double fun_t = pow((tanh(0.001529*pow(t,0.984))),0.12);
	//condT(T), T: Bondokov R T, J. Crystal Growth 310 (2008) 4020
    return( condTmax_RT*fun_t*pow((T/300.),-1.25) );
 }

static MaterialsDB::Register<AlN> materialDB_register_AlN;

}       // namespace plask
