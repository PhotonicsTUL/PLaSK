#include "AlGaAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlGaAs::AlGaAs(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
}

std::string AlGaAs::str() const { return StringBuilder("Al", Al)("Ga")("As"); }

std::string AlGaAs::name() const { return NAME; }

MI_PROPERTY(AlGaAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Al*mAlAs.lattC(T,'a') + Ga*mGaAs.lattC(T,'a');
    else if (x == 'c') tLattC = Al*mAlAs.lattC(T,'c') + Ga*mGaAs.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(AlGaAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: AlAs, GaAs")
            )
double AlGaAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlAs.Eg(T,e,point) + Ga*mGaAs.Eg(T,e,point) - Al*Ga*(-0.127+1.310*Al);
    else if (point == 'X') tEg = Al*mAlAs.Eg(T,e,point) + Ga*mGaAs.Eg(T,e,point) - Al*Ga*(0.055);
    else if (point == 'L') tEg = Al*mAlAs.Eg(T,e,point) + Ga*mGaAs.Eg(T,e,point);
    else if (point == '*') {
        double tEgG = Al*mAlAs.Eg(T,e,'G') + Ga*mGaAs.Eg(T,e,'G') - Al*Ga*(-0.127+1.310*Al);
        double tEgX = Al*mAlAs.Eg(T,e,'X') + Ga*mGaAs.Eg(T,e,'X') - Al*Ga*(0.055);
        double tEgL = Al*mAlAs.Eg(T,e,'L') + Ga*mGaAs.Eg(T,e,'L');
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlGaAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs::Dso(double T, double e) const {
    return ( Al*mAlAs.Dso(T,e) + Ga*mGaAs.Dso(T,e) );
}

MI_PROPERTY(AlGaAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("linear interpolation: AlSb, GaSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = Al*mAlAs.Me(T,e,point).c00 + Ga*mGaAs.Me(T,e,point).c00,
        tMe.c11 = Al*mAlAs.Me(T,e,point).c11 + Ga*mGaAs.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = Al*mAlAs.Me(T,e,point).c00 + Ga*mGaAs.Me(T,e,point).c00;
        tMe.c11 = Al*mAlAs.Me(T,e,point).c11 + Ga*mGaAs.Me(T,e,point).c11;
    }
    return ( tMe );
}

MI_PROPERTY(AlGaAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaAs::Mhh(double T, double e) const {
    double lMhh = Al*mAlAs.Mhh(T,e).c00 + Ga*mGaAs.Mhh(T,e).c00,
           vMhh = Al*mAlAs.Mhh(T,e).c11 + Ga*mGaAs.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlGaAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlGaAs::Mlh(double T, double e) const {
    double lMlh = Al*mAlAs.Mlh(T,e).c00 + Ga*mGaAs.Mlh(T,e).c00,
           vMlh = Al*mAlAs.Mlh(T,e).c11 + Ga*mGaAs.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlGaAs, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlGaAs::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlGaAs, y1,
	MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) chapter 7; "),
	MISource("linear interpolation: AlAs, GaAs"),
	MIComment("no temperature dependence")
)
double AlGaAs::y1() const {
	return (Al*mAlAs.y1() + Ga * mGaAs.y1());
}

MI_PROPERTY(AlGaAs, y2,
	MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) chapter 7; "),
	MISource("linear interpolation: AlAs, GaAs"),
	MIComment("no temperature dependence")
)
double AlGaAs::y2() const {
	return (Al*mAlAs.y2() + Ga * mGaAs.y2());
}

MI_PROPERTY(AlGaAs, y3,
	MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) chapter 7; "),
	MISource("linear interpolation: AlAs, GaAs"),
	MIComment("no temperature dependence")
)
double AlGaAs::y3() const {
	return (Al*mAlAs.y3() + Ga * mGaAs.y3());
}

MI_PROPERTY(AlGaAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlGaAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlGaAs, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs::VB(double T, double e, char point, char hole) const {
    double tVB( Al*mAlAs.VB(T,0.,point,hole) + Ga*mGaAs.VB(T,0.,point,hole) );
    if (!e) return tVB;
    else
    {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
}

MI_PROPERTY(AlGaAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs::ac(double T) const {
    return ( Al*mAlAs.ac(T) + Ga*mGaAs.ac(T) );
}

MI_PROPERTY(AlGaAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs::av(double T) const {
    return ( Al*mAlAs.av(T) + Ga*mGaAs.av(T) );
}

MI_PROPERTY(AlGaAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs::b(double T) const {
    return ( Al*mAlAs.b(T) + Ga*mGaAs.b(T) );
}

MI_PROPERTY(AlGaAs, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs::d(double T) const {
    return ( Al*mAlAs.d(T) + Ga*mGaAs.d(T) );
}

MI_PROPERTY(AlGaAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs::c11(double T) const {
    return ( Al*mAlAs.c11(T) + Ga*mGaAs.c11(T) );
}

MI_PROPERTY(AlGaAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs::c12(double T) const {
    return ( Al*mAlAs.c12(T) + Ga*mGaAs.c12(T) );
}

MI_PROPERTY(AlGaAs, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs::c44(double T) const {
    return ( Al*mAlAs.c44(T) + Ga*mGaAs.c44(T) );
}

MI_PROPERTY(AlGaAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: AlAs, GaAs")
            )
Tensor2<double> AlGaAs::thermk(double T, double t) const {
    double lCondT = 1./(Al/mAlAs.thermk(T,t).c00 + Ga/mGaAs.thermk(T,t).c00 + Al*Ga*0.32),
           vCondT = 1./(Al/mAlAs.thermk(T,t).c11 + Ga/mGaAs.thermk(T,t).c11 + Al*Ga*0.32);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlGaAs, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs::dens(double T) const {
    return ( Al*mAlAs.dens(T) + Ga*mGaAs.dens(T) );
}

MI_PROPERTY(AlGaAs, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: AlAs, GaAs"),
            MIComment("no temperature dependence")
            )
double AlGaAs::cp(double T) const {
    return ( Al*mAlAs.cp(T) + Ga*mGaAs.cp(T) );
}

Material::ConductivityType AlGaAs::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(AlGaAs, nr,
            //MISource("D.T.F. Marple, J. Appl. Phys. 35 (1964) 1241-1242; "),
            MISource("S. Gehrsitz, J. Appl. Phys. 87 (2000) 7825-7837; "),
            //MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, Wiley 2005"), // temperature dependence
            MIComment("fit by Leszek Frasunkiewicz")
            )
double AlGaAs::nr(double lam, double T, double /*n*/) const {
    double A = -0.4490233379*Al + 3.25759049;
    double B = 29.22871618*pow(Al,(-2.35349122*pow(Al,8.844978824)));
    double C = -304.7269552*Al*Al*Al + 335.1918592*Al*Al + 194.6726344*Al - 559.6098207;

    //dn/dT
    double Ad = 0.0003442176581*Al*Al*Al - 0.0005412098145*Al*Al + 0.00008671640556*Al + 0.0002093262406;
    double Bd = 132.1382231*exp(-8.32822628*Al*Al*Al + 14.65496754*Al*Al - 7.135900438*Al);
    double Cd = 117.24*Al -689.06;
    double dndT = Ad*exp(Bd / (lam + Cd));

    return ( A*exp(B / (lam + C)) + dndT*(T-296) );

    // old
    //double L2 = lam*lam*1e-6;
    //double nR296K = sqrt(1.+(9.659-2.604*Al)*L2/(L2-(0.137-0.069*Al)));
    //return ( nR296K + nR296K*(Al*4.6e-5+Ga*4.5e-5)*(T-296.) );
}

MI_PROPERTY(AlGaAs, absp,
            MIComment("calculated as for Si-doped AlGaAs but with n = 1e16")
            )
double AlGaAs::absp(double lam, double T) const {
    double tEgRef300 = mGaAs.Eg(300.,0.,'G');
    double tEgT = Eg(T,0.,'G');
    if (tEgT > Eg(T,0.,'X'))
        tEgT = Eg(T,0.,'X');
    double tDWl = phys::h_eVc1e9*(tEgRef300-tEgT)/(tEgRef300*tEgT);
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

double AlGaAs::eps(double T) const {
    return Al*mAlAs.eps(T) + Ga*mGaAs.eps(T);
}

bool AlGaAs::isEqual(const Material &other) const {
    const AlGaAs& o = static_cast<const AlGaAs&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlGaAs> materialDB_register_AlGaAs;

}} // namespace plask::materials
