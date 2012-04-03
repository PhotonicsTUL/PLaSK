#include "air.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string Air::name() const { return NAME; }
Material::Kind Air::kind() const { return Material::NONE; }


double Air::A(double T) const { throwNotApplicable("A(double T)"); assert(0); }

double Air::absp(double wl, double T) const { return 0.; }

double Air::B(double T) const { throwNotApplicable("B(double T)"); assert(0); }

double Air::C(double T) const { throwNotApplicable("C(double T)"); assert(0); }

double Air::CBO(double T, char point) const { throwNotApplicable("CBO(double T, char point)"); assert(0); }

double Air::chi(double T, char point) const { throwNotApplicable("chi(double T, char point)"); assert(0); }

double Air::cond(double T) const { return 0.; }
double Air::cond_l(double T) const { return 0.; }
double Air::cond_v(double T) const { return 0.; }

double Air::D(double T) const { throwNotApplicable("D(double T)"); assert(0); }

double Air::dens(double T) const { throwNotApplicable("dens(double T)"); assert(0); }

double Air::Dso(double T) const { throwNotApplicable("Dso(double T)"); assert(0); }

double Air::EactA(double T) const { throwNotApplicable("EactA(double T)"); assert(0); }
double Air::EactD(double T) const { throwNotApplicable("EactD(double T)"); assert(0); }

double Air::Eg(double T, char point) const { throwNotApplicable("Eg(double T, char point)"); assert(0); }

double Air::eps(double T) const { return 1.0; }

double Air::lattC(double T, char x) const { throwNotApplicable("lattC(double T, char x)"); assert(0); }

double Air::Me(double T, char point) const { throwNotApplicable("Me(double T, char point)"); assert(0); }
double Air::Me_l(double T, char point) const { throwNotApplicable("Me_l(double T, char point)"); assert(0); }
double Air::Me_v(double T, char point) const { throwNotApplicable("Me_v(double T, char point)"); assert(0); }

double Air::Mh(double T, char EqType) const { throwNotApplicable("Mh(double T, char EqType)"); assert(0); }
double Air::Mh_l(double T, char point) const { throwNotApplicable("Mh_l(double T, char point)"); assert(0); }
double Air::Mh_v(double T, char point) const { throwNotApplicable("Mh_v(double T, char point)"); assert(0); }

double Air::Mhh(double T, char point) const { throwNotApplicable("Mhh(double T, char point)"); assert(0); }
double Air::Mhh_l(double T, char point) const { throwNotApplicable("Mhh_l(double T, char point)"); assert(0); }
double Air::Mhh_v(double T, char point) const { throwNotApplicable("Mhh_v(double T, char point)"); assert(0); }

double Air::Mlh(double T, char point) const { throwNotApplicable("B(double T)"); assert(0); }
double Air::Mlh_l(double T, char point) const { throwNotApplicable("B(double T)"); assert(0); }
double Air::Mlh_v(double T, char point) const { throwNotApplicable("B(double T)"); assert(0); }

double Air::mob(double T) const { throwNotApplicable("mob(double T)"); assert(0); }

double Air::Mso(double T) const { throwNotApplicable("Mso(double T)"); assert(0); }

double Air::Nc(double T, char point) const { throwNotApplicable("Nc(double T, char point)"); assert(0); }
double Air::Nc(double T) const { throwNotApplicable("Nc(double T)"); assert(0); }

double Air::Nf(double T) const { throwNotApplicable("Nf(double T)"); assert(0); }

double Air::Ni(double T) const { throwNotApplicable("Ni(double T)"); assert(0); }

double Air::nr(double wl, double T) const { return 1.; }

double Air::res(double T) const { return INFINITY; }
double Air::res_l(double T) const { return INFINITY; }
double Air::res_v(double T) const { return INFINITY; }

double Air::specHeat(double T) const { throwNotApplicable("specHeat(double T)"); assert(0); }

double Air::condT(double T) const { return 0.; }
double Air::condT(double T, double thickness) const { return 0.; }
double Air::condT_l(double T, double thickness) const { return 0.; }
double Air::condT_v(double T, double thickness) const { return 0.; }

double Air::VBO(double T) const { throwNotApplicable("VBO(double T)"); assert(0); }


static MaterialsDB::Register<Air> materialDB_register_Air;

} // namespace plask
