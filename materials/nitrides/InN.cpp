#include "InN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

#include <cmath>

namespace plask {

std::string InN::name() const { return NAME; }

MI_PROPERTY(InN, condT,
            MISource("H. Tong et al., Proc. SPIE 7602 (2010) 76020U")
            )
double InN::condT(double T) const {
    return( 126. );
 }

MaterialsDB::Register<InN> materialDB_register_InN;

}       // namespace plask
