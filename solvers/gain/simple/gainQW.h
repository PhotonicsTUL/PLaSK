#include "typy.h"
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_fermi_dirac.h>
#include <gsl/gsl_min.h>

#include <plask/plask.hpp>
using namespace plask;

namespace QW{

  struct ExternalLevels {
      double* el;
      double* hh;
      double* lh;
      ExternalLevels() {}
      ExternalLevels(double* el, double* hh, double* lh): el(el), hh(hh), lh(lh) {}
  };

  class nosnik{
    friend class gain;
    nosnik();
    double * poziomy;
    double masa_w_plaszcz;
    double masa_w_kier_prost;
    double masabar;
    double gleb;
    double gleb_fal;
    double Eodk(double);
    double En(double,int);
    double pozoddna(int);
    int ilepoz();
    ~nosnik();
  };
  class parametry{
    friend class gain;
    double * ldopar;
    char rdziury;
    ~parametry();
  };

  class gain{
  public:
    gain();
    double En_to_len(double);
    void Set_temperature(double); //W kelwinach
    double Get_temperature();
    void Set_refr_index(double);
    double Get_refr_index();
    void Set_well_width(double); //W angstremach
    double Get_well_width();
    void Set_barrier_width(double); //W angstremach
    double Get_barrier_width();
    void Set_waveguide_width(double);
    double Get_waveguide_width();
    void Set_bandgap(double); //W eV
    double Get_bandgap();
    void Set_split_off(double); //W eV
    double Get_split_off();
    void Set_lifetime(double); //W ps
    double Get_lifetime();
    void Set_koncentr(double); //W 1/cm^3
    double Get_koncentr();
    double Get_bar_konc_c();
    double Get_bar_konc_v();
    long Calculate_Gain_Profile();
    long Calculate_Gain_Profile2();
    long Calculate_Gain_Profile_n(const ExternalLevels&, double);
    long Calculate_Spont_Profile();
    double Get_qFlc();
    double Get_qFlv();
    double Get_last_point();
    void Set_last_point(double); //Ustawia prawy kraniec przedzialu
    double Get_first_point();
    void Set_first_point(double); //Ustawia lewy kraniec przedzialu
    double Get_step();
    void Set_step(double); //Krok obliczen
    void Set_conduction_depth(double); //Glebokosc studni w pasmie przew.
    double Get_conduction_depth();
    void Set_cond_waveguide_depth(double);
    double Get_cond_waveguide_depth();
    void Set_valence_depth(double); //Glebokosc studni w pasmie walenc.
    double Get_valence_depth();
    void Set_vale_waveguide_depth(double);
    double Get_vale_waveguide_depth();
    double Get_electron_level_depth(int);
    double Get_electron_level_from_bottom(int);
    double Get_heavy_hole_level_depth(int);
    double Get_heavy_hole_level_from_bottom(int);
    double Get_light_hole_level_depth(int);
    double Get_light_hole_level_from_bottom(int);
    int Get_number_of_electron_levels();
    int Get_number_of_heavy_hole_levels();
    int Get_number_of_light_hole_levels();
    void Set_electron_mass_in_plain(double);
    double Get_electron_mass_in_plain();
    void Set_electron_mass_transverse(double);
    double Get_electron_mass_transverse();
    void Set_heavy_hole_mass_in_plain(double);
    double Get_heavy_hole_mass_in_plain();
    void Set_heavy_hole_mass_transverse(double);
    double Get_heavy_hole_mass_transverse();
    void Set_light_hole_mass_in_plain(double);
    double Get_light_hole_mass_in_plain();
    void Set_light_hole_mass_transverse(double);
    double Get_light_hole_mass_transverse();
    void Set_electron_mass_in_barrier(double);
    double Get_electron_mass_in_barrier();
    void Set_heavy_hole_mass_in_barrier(double);
    double Get_heavy_hole_mass_in_barrier();
    void Set_light_hole_mass_in_barrier(double);
    double Get_light_hole_mass_in_barrier();
    void Set_momentum_matrix_element(double); //W eV
    double Get_momentum_matrix_element();
    double Get_gain_at(double);
    double Get_gain_at_n(double, double);
    double Get_gain_at_n(double, const ExternalLevels&, double);
    double Get_bar_gain_at(double);
    double Get_inversion(double E, int i=0);
    double Get_spont_at(double);
    double ** Get_gain_tab(); //Wskaznik do tabl [2][ile_trzeba]
    std::vector<std::vector<double> > & Get_spont_wek();
    //    void przygobl();
    double Find_max_gain();
    double Find_max_gain_n(const ExternalLevels&, double);
    ~gain();

    void przygobl();
    void przygobl2();
    void przygobl_n(double);
    void przygobl_n(const ExternalLevels&, double);
    void przygoblE(); // LUKASZ
    void przygoblHH(); // LUKASZ
	void przygoblLH(); // LUKASZ
    double * sendLev(std::vector<double> &zewpoziomy); // LUKASZ
    void przygoblHHc(std::vector<double> &iLevHH); // LUKASZ
    void przygoblLHc(std::vector<double> &iLevLH); // LUKASZ
	void przygoblQFL(double iTotalWellH); // LUKASZ
    double GetGainAt(double E,double iTotalWellH); // LUKASZ

    int Break;
    static const double kB;
    static const double przelm;
    static const double przels;
    static const double ep0;
    static const double c;
    static const double exprng;
    double bladb; //dopuszczalny błąd bezwzględny
    double T; //temperatura
    double n_r; //wsp. załamania
    double szer; // szerokość studni
    double szerb; //szerokosc bariery
    double szer_fal; // szerokość nad studnią
    double Eg; // przerwa energetyczna
    double Mt; // el. macierzowy
    double deltaSO; // split-off
    double tau; // czas życia
    double konc; // koncentracja
    double barkonc_c; //koncentracja elektronów w barierach
    double barkonc_v; //koncentracja dzur w barierach
    double Efc; // quasi-poziom Fermiego dla pasma walencyjnego
    double Efv; // quasi-poziom Fermiego dla pasma przewodnictwa
    double ** Twzmoc;
    std::vector<std::vector<double> > Tspont;
    long ilpt;
    double enpo,enko;
    double krok;
    int ilwyw;
    char ustawione;
    nosnik el;
    nosnik hh;
    nosnik lh;
    bool kasuj_poziomy;

    double qFlc(); // liczy poziom Fermiego
    double qFlc2();
    double qFlc_n(double);
    double qFlv();
    double qFlv2();
    double qFlv_n(double);
    double element(); // liczy element macierzowy
    double przel_dlug_z_angstr(double);
    double przel_dlug_na_angstr(double);
    double przel_czas_z_psek(double);
    double przel_czas_na_psek(double);
    double przel_konc_z_cm(double);
    double przel_konc_na_cm(double);
    double fc(double); // rozk?ad Fermiego dla p. przewodnictwa
    double fv(double); // rozk?ad Fermiego dla p. walencyjnego
    double L(double,double); // funkcja poszerzająca
    double Lpr(double,double); // jej pochodna
    double gdziepoziomy(double,double *);
    double gdziepoziomy2A(double,double *);
    double gdziepoziomy2B(double,double *);
    double krance(int,double,double);
    double * znajdzpoziomy(nosnik &);
    double * znajdzpoziomy2(nosnik &);
    double gdzieqflv(double,double *);
    double gdzieqflv2(double,double *);
    double gdzieqflv_n(double,double *);
    double gdzieqflc(double,double *);
    double gdzieqflc2(double,double *);
    double gdzieqflc_n(double,double *);
    double kodE(double,double,double);
    double rored(double,double,double);
    double rored2(double,double,double);
    double rored_n(double,double,double,double);
    double dosplotu(double, parametry *);
    double dosplotu2(double, parametry *);
    double dosplotu_n(double, parametry *);
    double dosplotu_spont(double, parametry *);
    double wzmoc_z_posz(double);
    double wzmoc_z_posz2(double);
    double wzmoc_z_posz_n(double,double);
    double spont_z_posz(double t);
    double wzmoc0(double);
    double wzmoc02(double);
    double wzmoc0_n(double,double);
    double spont0(double);
    double Prost(double (gain::*)(double, parametry *),double,double,double,parametry *,double);
    double metsiecz(double (gain::*)(double,double *),double,double,double * =NULL,double prec=1e-7);
  };
  double min_wzmoc(double E,void*);
}
