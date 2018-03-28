#include "GaAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

MI_PROPERTY(GaAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double GaAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.65325 + 3.88e-5 * (T-300.);
    return tLattC;
}

MI_PROPERTY(GaAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double GaAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(1.519, 0.5405e-3, 204., T);
    else if (point == 'X') tEg = phys::Varshni(1.981, 0.460e-3, 204., T);
    else if (point == 'L') tEg = phys::Varshni(1.815, 0.605e-3, 204., T);
    else if (point == '*') {
        double tEgG = phys::Varshni(1.519, 0.5405e-3, 204., T);
        double tEgX = phys::Varshni(1.981, 0.460e-3, 204., T);
        double tEgL = phys::Varshni(1.815, 0.605e-3, 204., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );

}

MI_PROPERTY(GaAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::Dso(double /*T*/, double /*e*/) const {
    return 0.341;
}

MI_PROPERTY(GaAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MIComment("no temperature dependence")
           )
Tensor2<double> GaAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.067), tMeX(0.85), tMeL(0.56);
    if (point == 'G') {
        tMe.c00 = tMeG; tMe.c11 = tMeG;
    }
    else if (point == 'X') {
        tMe.c00 = tMeX; tMe.c11 = tMeX;
    }
    else if (point == 'L') {
        tMe.c00 = tMeL; tMe.c11 = tMeL;
    }
    else if (point == '*') {
        if ( Eg(T,e,'G') == Eg(T,e,'*') ) {
            tMe.c00 = tMeG; tMe.c11 = tMeG;
        }
        else if ( Eg(T,e,'X') == Eg(T,e,'*') ) {
            tMe.c00 = tMeX; tMe.c11 = tMeX;
        }
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) {
            tMe.c00 = tMeL; tMe.c11 = tMeL;
        }
    }
    return ( tMe );
}

MI_PROPERTY(GaAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
           )
Tensor2<double> GaAs::Mhh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.33, 0.33); // [001]
}

MI_PROPERTY(GaAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
           )
Tensor2<double> GaAs::Mlh(double /*T*/, double /*e*/) const {
    return Tensor2<double>(0.090, 0.090);
}

MI_PROPERTY(GaAs, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> GaAs::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    return Tensor2<double>(tMc00, tMc11); // [001]
}

MI_PROPERTY(GaAs, y1,
	MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) chapter 7"),
	MIComment("no temperature dependence")
)
double GaAs::y1() const {
	return 7.10;
}

MI_PROPERTY(GaAs, y2,
	MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) chapter 7"),
	MIComment("no temperature dependence")
)
double GaAs::y2() const {
	return 2.02;
}

MI_PROPERTY(GaAs, y3,
	MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) chapter 7"),
	MIComment("no temperature dependence")
)
double GaAs::y3() const {
	return 2.91;
}

MI_PROPERTY(GaAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double GaAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return tCB;
    else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;
}

MI_PROPERTY(GaAs, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::VB(double T, double e, char /*point*/, char hole) const {
    double tVB(-0.80);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
    return tVB;
}

MI_PROPERTY(GaAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::ac(double /*T*/) const {
    return -7.17;
}

MI_PROPERTY(GaAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::av(double /*T*/) const {
    return 1.16;
}

MI_PROPERTY(GaAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::b(double /*T*/) const {
    return -2.0;
}

MI_PROPERTY(GaAs, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::d(double /*T*/) const {
    return -4.8;
}

MI_PROPERTY(GaAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::c11(double /*T*/) const {
    return 122.1;
}

MI_PROPERTY(GaAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::c12(double /*T*/) const {
    return 56.6;
}

MI_PROPERTY(GaAs, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double GaAs::c44(double /*T*/) const {
    return 60.0;
}

MI_PROPERTY(GaAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 150, 1500)
           )
Tensor2<double> GaAs::thermk(double T, double /*t*/) const {
    double tCondT = 45.*pow((300./T),1.28);
    return Tensor2<double>(tCondT, tCondT);
}

MI_PROPERTY(GaAs, cond,
            MISource("http://www.ioffe.ru/SVA/NSM/Semicond/GaAs/electric.html"),
            MIComment("Carrier concentration estimated")
           )
Tensor2<double> GaAs::cond(double T) const {
    double c = 1e2 * phys::qe * 8000.* pow((300./T), 2./3.) * 1e16;
    return Tensor2<double>(c, c);
}

MI_PROPERTY(GaAs, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MIComment("no temperature dependence")
            )
double GaAs::dens(double /*T*/) const { return 5.31749e3; }

MI_PROPERTY(GaAs, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52"),
            MIComment("no temperature dependence")
            )
double GaAs::cp(double /*T*/) const { return 0.327e3; }

Material::ConductivityType GaAs::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(GaAs, nr,
            //MISource("D.T.F. Marple, J. Appl. Phys. 35 (1964) 1241-1242; "),
            //MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, Wiley 2005"), // temperature dependence
            //MIComment("fit by Lukasz Piskorski")
            MISource("S. Gehrsitz, J. Appl. Phys. 87 (2000) 7825-7837; "),
            MIComment("fit by Leszek Frasunkiewicz")
           )
double GaAs::nr(double lam, double T, double /*n*/) const {
    double Al = 1e-30;
    double A = -0.4490233379*Al + 3.25759049;
    double B = 29.22871618*pow(Al,(-2.35349122*pow(Al,8.844978824)));
    double C = -304.7269552*Al*Al*Al + 335.1918592*Al*Al + 194.6726344*Al - 559.6098207;

    //dn/dT
    double Ad = 0.0003442176581*Al*Al*Al - 0.0005412098145*Al*Al + 0.00008671640556*Al + 0.0002093262406;
    double Bd = 132.1382231*exp(-8.32822628*Al*Al*Al + 14.65496754*Al*Al - 7.135900438*Al);
    double Cd = 117.24*Al -689.06;
    double dndT = Ad*exp(Bd / (lam + Cd));

    return ( A*exp(B / (lam + C)) + dndT*(T-296) );
    //old
    //double L2 = lam*lam*1e-6;
    //double nR296K = sqrt(1.+9.659*L2/(L2-0.137));
    //return ( nR296K + nR296K*4.5e-5*(T-296.) );
}

MI_PROPERTY(GaAs, absp,
            MISource(""),
            MIComment("calculated as for Si-doped GaAs but with n = 1e16")
           )
double GaAs::absp(double lam, double T) const {
    double tDWl = phys::h_eVc1e9*(Eg(300.,0.,'G')-Eg(T,0.,'G'))/(Eg(300.,0.,'G')*Eg(T,0.,'G'));
    double tWl = (lam-tDWl)*1e-3;
    double tAbsp(0.);
    double tN = 1e16; // concentration for undoped GaAs
    if (tWl <= 6.) // 0.85-6 um
        tAbsp = (tN/1e18)*(1e24*exp(-tWl/0.0169)+4.67+0.00211*pow(tWl,4.80));
    else if (tWl <= 27.) // 6-27 um
        tAbsp = (tN/1e18)*(-8.4+0.233*pow(tWl,2.6));
    return ( tAbsp );
    //return 0.;
}

MI_PROPERTY(GaAs, eps,
            MISource("http://www.ioffe.ru/SVA/NSM/Semicond/GaAs/basic.html")
           )
double GaAs::eps(double /*T*/) const {
    return 12.9;
}

std::string GaAs::name() const { return NAME; }

bool GaAs::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<GaAs> materialDB_register_GaAs;

}} // namespace plask::materials
