#include "GaSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaSb::name() const { return NAME; }

MI_PROPERTY(GaSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 6.0959 + 4.72e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(GaSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(0.812, 0.417e-3, 140., T);
    else if (point == 'X') tEg = phys::Varshni(1.141, 0.475e-3, 94., T);
    else if (point == 'L') tEg = phys::Varshni(0.875, 0.597e-3, 140., T);
    return ( tEg );
}

MI_PROPERTY(GaSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::Dso(double T, double e) const {
    return ( 0.76 );
}

MI_PROPERTY(GaSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("only for Gamma point"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaSb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if (point == 'G') {
        tMe.c00 = 0.039;
        tMe.c11 = 0.039;
    }
    return ( tMe );
}

MI_PROPERTY(GaSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaSb::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.22, 0.22); // [001]
    return ( tMhh );
}

MI_PROPERTY(GaSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaSb::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.045, 0.045);
    return ( tMlh );
}

MI_PROPERTY(GaSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point) + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::VB(double T, double e, char point, char hole) const {
    double tVB(-0.03);
    if (!e) return ( tVB );
    else return ( tVB + 2.*av(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::ac(double T) const {
    return ( -7.5 );
}

MI_PROPERTY(GaSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::av(double T) const {
    return ( 0.8 );
}

MI_PROPERTY(GaSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::b(double T) const {
    return ( -2.0 );
}

MI_PROPERTY(GaSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::d(double T) const {
    return ( -4.7 );
}

MI_PROPERTY(GaSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::c11(double T) const {
    return ( 88.42 );
}

MI_PROPERTY(GaSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::c12(double T) const {
    return ( 40.26 );
}

MI_PROPERTY(GaSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::c44(double T) const {
    return ( 43.22 );
}

MI_PROPERTY(GaSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, Wiley 2005"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 50, 920)
           )
Tensor2<double> GaSb::thermk(double T, double t) const {
    double tCondT = 36.*pow((300./T),1.35);
    return ( Tensor2<double>(tCondT, tCondT) );
}

bool GaSb::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<GaSb> materialDB_register_GaSb;

}} // namespace plask::materials
