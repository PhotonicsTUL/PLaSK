#include "AlGaN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlGaN::name() const { return NAME; }

std::string AlGaN::str() const { return StringBuilder("Al", Al)("Ga")("N"); }

AlGaN::AlGaN(const Material::Composition& Comp) {

    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
}

MI_PROPERTY(AlGaN, thermk,
            MISource("B. C. Daly et al., Journal of Applied Physics 92 (2002) 3820"),
            MIComment("based on data for Al = 0.2, 0.45")
            )
Tensor2<double> AlGaN::thermk(double T, double t) const {
    double lCondT = 1/(Al/mAlN.thermk(T,t).c00 + Ga/mGaN.thermk(T,t).c00 + Al*Ga*0.4),
           vCondT = 1/(Al/mAlN.thermk(T,t).c11 + Ga/mGaN.thermk(T,t).c11 + Al*Ga*0.4);
    return Tensor2<double>(lCondT,vCondT);
 }

MI_PROPERTY(AlGaN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("fit to GaN:Si/GaN:Mg/GaN:undoped in region 360 - 400 nm"),
            MIComment("no temperature dependence")
            )
double AlGaN::absp(double lam, double T) const {
    double a = phys::h_eVc1e9/lam - Eg(T,0,'G');
    return 19000*exp(a/0.019) + 330*exp(a/0.07);
}

MI_PROPERTY(AlGaN, nr,
            MIComment("shift of the nR for GaN")
            )
double AlGaN::nr(double lam, double T, double /*n*/) const {
    double dEg = Eg(T,0.,'G') - mGaN.Eg(300.,0.,'G'),
           Eold = phys::h_eVc1e9 / lam,
           Enew = Eold - dEg;

    if (Enew > 1.000 && Enew < 2.138) // 580-1240 nm
        return ( 0.013914*Enew*Enew*Enew*Enew - 0.096422*Enew*Enew*Enew + 0.27318*Enew*Enew - 0.27725*Enew + 2.3535 );
    else if (Enew < 3.163) // 392-580 nm
        return ( 0.1152*Enew*Enew*Enew - 0.7955*Enew*Enew + 1.959*Enew + 0.68 );
    else if (Enew < 3.351) // 370-392 nm
        return ( 18.2292*Enew*Enew*Enew - 174.6974*Enew*Enew + 558.535*Enew - 593.164 );
    else if (Enew < 3.532) // 351-370 nm
        return ( 33.63905*Enew*Enew*Enew - 353.1446*Enew*Enew + 1235.0168*Enew - 1436.09 );
    else if (Enew < 4.100) // 336-351 nm
        return ( -0.72116*Enew*Enew*Enew + 8.8092*Enew*Enew - 35.8878*Enew + 51.335 );
    else if (Enew < 5.000) // 248-336 nm
        return ( 0.351664*Enew*Enew*Enew*Enew - 6.06337*Enew*Enew*Enew + 39.2317*Enew*Enew - 112.865*Enew + 124.358 );
    else
        return 0.;
}

MI_PROPERTY(AlGaN, Eg,
            MISource("Vurgaftman et al. in Piprek 2007 Nitride Semicondcuctor Devices")
            )
double AlGaN::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G' || point == '*') tEg = Al*mAlN.Eg(T,e,point) + Ga*mGaN.Eg(T,e,point) - Al*Ga*0.8;
    return (tEg);
}

double AlGaN::VB(double T, double e, char point, char /*hole*/) const {
    return 0.3 * (mGaN.Eg(T,e,point) - Eg(T,e,point));
}

double AlGaN::Dso(double T, double e) const {
    return Al*mAlN.Dso(T,e) + Ga*mGaN.Dso(T,e);
}

MI_PROPERTY(AlGaN, Me,
            MISource("linear interpolation: AlN, GaN")
            )
Tensor2<double> AlGaN::Me(double T, double e, char point) const {
    double lMe = Al*mAlN.Me(T,e,point).c00 + Ga*mGaN.Me(T,e,point).c00,
           vMe = Al*mAlN.Me(T,e,point).c11 + Ga*mGaN.Me(T,e,point).c11;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(AlGaN, Mhh,
            MISource("linear interpolation: AlN, GaN")
            )
Tensor2<double> AlGaN::Mhh(double T, double e) const {
    double lMhh = Al*mAlN.Mhh(T,e).c00 + Ga*mGaN.Mhh(T,e).c00,
           vMhh = Al*mAlN.Mhh(T,e).c11 + Ga*mGaN.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlGaN, Mlh,
            MISource("linear interpolation: AlN, GaN")
            )
Tensor2<double> AlGaN::Mlh(double T, double e) const {
    double lMlh = Al*mAlN.Mlh(T,e).c00 + Ga*mGaN.Mlh(T,e).c00,
           vMlh = Al*mAlN.Mlh(T,e).c11 + Ga*mGaN.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlGaN, lattC,
            MISource("linear interpolation: GaN, AlN")
            )
double AlGaN::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = mAlN.lattC(T,'a')*Al + mGaN.lattC(T,'a')*Ga;
    else if (x == 'c') tLattC = mAlN.lattC(T,'c')*Al + mGaN.lattC(T,'c')*Ga;
    return (tLattC);
}

Material::ConductivityType AlGaN::condtype() const { return Material::CONDUCTIVITY_I; }

bool AlGaN::isEqual(const Material &other) const {
    const AlGaN& o = static_cast<const AlGaN&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlGaN> materialDB_register_AlGaN;

}}       // namespace plask::materials
