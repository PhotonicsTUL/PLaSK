#include "AlN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlN::name() const { return NAME; }

MI_PROPERTY(AlN, condT,
            MISource("G. A. Slack, J. Phys. Chem. Sol. 48 (1987) 641"),
            MISource("Bondokov R T, J. Crystal Growth 310 (2008) 4020"),
            MIComment("based on Si-doped GaN and AlN data to estimate thickness dependence"))
double AlN::condT(double T, double t) const {
	double fun_t = pow((tanh(0.001529*pow(t,0.984))),0.12);
    return( 285*fun_t*pow((T/300.),-1.25) );
 }

static MaterialsDB::Register<AlN> materialDB_register_AlN;

}       // namespace plask
