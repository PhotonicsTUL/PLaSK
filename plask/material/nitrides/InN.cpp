#include "InN.h"

#include "../db.h"  //MaterialsDB::Register

#include <cmath>

namespace plask {

InN::InN() {
	//M. Kuc 12.02.2012	
	//condT(max,RT); Tong H, Proc. SPIE 7602 (2010) 76020U
	condTmax_RT = 126;
}

std::string InN::name() const { return NAME; }

double InN::condT(double T) const {
	//M. Kuc 12.02.2012
    return( condTmax_RT );
 }
 
MaterialsDB::Register<InN> materialDB_register_InN;

}       // namespace plask
