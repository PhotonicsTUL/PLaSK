#include "AlGaAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string AlGaAs::name() const { return NAME; }

std::string AlGaAs::str() const { return StringBuilder("Al", Al)("Ga")("As"); }

AlGaAs::AlGaAs(const Material::Composition& Comp) {

    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
}

MI_PROPERTY(AlGaAs, condT,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009")
            )
std::pair<double,double> AlGaAs::condT(double T, double t) const {
    double lCondT = 1/(Al/mAlAs.condT(T,t).first + Ga/mGaAs.condT(T,t).first + Al*Ga*0.32),
           vCondT = 1/(Al/mAlAs.condT(T,t).second + Ga/mAlAs.condT(T,t).second + Al*Ga*0.32);
    return(std::make_pair(lCondT,vCondT));
 }

MI_PROPERTY(AlGaAs, absp,
            MIComment("TODO")
            )
double AlGaAs::absp(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(AlGaAs, nr,
            MIComment("TODO")            )
double AlGaAs::nr(double wl, double T) const {
    return ( 0. );
}

MI_PROPERTY(AlGaAs, Eg,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
double AlGaAs::Eg(double T, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = 1.430 + 1.707*Al - 1.437*Al*Al + 1.310*Al*Al*Al;
    return (tEg);
}

MI_PROPERTY(AlGaAs, lattC,
            MISource("linear interpolation: GaAs, AlAs")
            )
double AlGaAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = mAlAs.lattC(T,'a')*Al + mGaAs.lattC(T,'a')*Ga;
    else if (x == 'c') tLattC = mAlAs.lattC(T,'c')*Al + mGaAs.lattC(T,'c')*Ga;
    return (tLattC);
}

static MaterialsDB::Register<AlGaAs> materialDB_register_AlGaAs;

}       // namespace plask
