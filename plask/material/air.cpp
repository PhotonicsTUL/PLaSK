#include "air.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Air::name() const { return NAME; }
Material::Kind Air::kind() const { return Material::NONE; }


double Air::A(double T) const { throwNotApplicable("A(double T)"); return 0; }

double Air::absp(double wl, double T) const { return 0.; }

double Air::B(double T) const { throwNotApplicable("B(double T)"); return 0; }

double Air::C(double T) const { throwNotApplicable("C(double T)"); return 0; }

double Air::CB(double T, double e, char point) const { throwNotApplicable("CB(double T, double e, char point)"); return 0; }

double Air::chi(double T, double e, char point) const { throwNotApplicable("chi(double T, double e, char point)"); return 0; }

MI_PROPERTY(Air, cond,
            MISource("S.D. Pawar et al., Journal of Geophysical Research, vol. 114, no. D2, id. D02205 (8 pp.), 2009"),
            MIComment("average value from (0.3-0.8)*10^-14 S/m")
           )
Tensor2<double> Air::cond(double T) const {
    double c = 0.55e-14;
    return Tensor2<double>(c, c);
}

Material::ConductivityType Air::condtype() const { return Material::CONDUCTIVITY_OTHER; }

double Air::D(double T) const { throwNotApplicable("D(double T)"); return 0; }

MI_PROPERTY(Air, dens,
            MISource("D.R. Lide, ed., CRC Handbook of Chemistry and Physics, Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005, section 6, p. 1")
           )
double Air::dens(double T) const { return 1.161; }

double Air::Dso(double T, double e) const { throwNotApplicable("Dso(double T, double e)"); return 0; }

double Air::EactA(double T) const { throwNotApplicable("EactA(double T)"); return 0; }
double Air::EactD(double T) const { throwNotApplicable("EactD(double T)"); return 0; }

double Air::Eg(double T, double e, char point) const { throwNotApplicable("Eg(double T, double e, char point)"); return 0; }

double Air::eps(double T) const { return 1.; }

double Air::lattC(double T, char x) const { throwNotApplicable("lattC(double T, char x)"); return 0; }

Tensor2<double> Air::Me(double T, double e, char point) const { throwNotApplicable("Me(double T, double e, char point)"); return 0.; }

Tensor2<double> Air::Mh(double T, double e) const { throwNotApplicable("Mh(double T, double e, char point)"); return 0.; }

Tensor2<double> Air::Mhh(double T, double e) const { throwNotApplicable("Mhh(double T, double e, char point)"); return 0.; }

Tensor2<double> Air::Mlh(double T, double e) const { throwNotApplicable("B(double T)"); return 0.; }

Tensor2<double> Air::mob(double T) const { throwNotApplicable("mob(double T)"); return 0.; }

double Air::Mso(double T, double e) const { throwNotApplicable("Mso(double T, double e)"); return 0; }

double Air::Nc(double T, double e, char point) const { throwNotApplicable("Nc(double T, double e, char point)"); return 0; }
double Air::Nv(double T, double e, char point) const { throwNotApplicable("Nv(double T, double e, char point)"); return 0; }

double Air::Nf(double T) const { throwNotApplicable("Nf(double T)"); return 0; }

double Air::Ni(double T) const { throwNotApplicable("Ni(double T)"); return 0; }

MI_PROPERTY(Air, nr,
            MISource("D.R. Lide, ed., CRC Handbook of Chemistry and Physics, Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005, section 10, p. 224"),
            MIComment("using equation in source the calculated values are 1.0002-1.0003 for 200-2000nm wavelength range"),
            MIArgumentRange(MaterialInfo::T, 200, 2000)
           )
double Air::nr(double wl, double T, double n) const { return 1.; }

MI_PROPERTY(Air, cp,
            MISource("D.R. Lide, ed., CRC Handbook of Chemistry and Physics, Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005, section 6, p. 1")
           )
double Air::cp(double T) const { return 1.007e3; }

MI_PROPERTY(Air, thermk,
            MISource("D.R. Lide, ed., CRC Handbook of Chemistry and Physics, Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005, section 6, p. 175"),
            MIComment("fit by Lukasz Piskorski"),
            MIArgumentRange(MaterialInfo::T, 100, 600)
           )
Tensor2<double> Air::thermk(double T, double h) const {
    double tCondT = 0.0258*pow(T/300.,0.84);
    return Tensor2<double>(tCondT, tCondT);
}

double Air::VB(double T, double e, char point, char hole) const { throwNotApplicable("VB(double T, double e, char point, char hole)"); return 0; }

bool Air::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<Air> materialDB_register_Air;

}} // namespace plask::materials
