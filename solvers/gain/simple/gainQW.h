#include "typy.h"
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_fermi_dirac.h>
#include <gsl/gsl_min.h>

namespace QW{
  class nosnik{
    friend class gain;
    nosnik();
    ldouble * poziomy;
    ldouble masa_w_plaszcz;
    ldouble masa_w_kier_prost;
    ldouble masabar;
    ldouble gleb;
    ldouble gleb_fal;
    ldouble Eodk(ldouble);
    ldouble En(ldouble,int);
    ldouble pozoddna(int);
    int ilepoz();
    ~nosnik();
  };
  class parametry{
    friend class gain;
    ldouble * ldopar;
    char rdziury;
    ~parametry();
  };

  class gain{
  public:
    gain();
    ldouble En_to_len(ldouble);
    void Set_temperature(ldouble); //W kelwinach
    ldouble Get_temperature();
    void Set_refr_index(ldouble);
    ldouble Get_refr_index();
    void Set_well_width(ldouble); //W angstremach
    ldouble Get_well_width();
    void Set_barrier_width(ldouble); //W angstremach
    ldouble Get_barrier_width();
    void Set_waveguide_width(ldouble);
    ldouble Get_waveguide_width();
    void Set_bandgap(ldouble); //W eV
    ldouble Get_bandgap();
    void Set_split_off(ldouble); //W eV
    ldouble Get_split_off();
    void Set_lifetime(ldouble); //W ps
    ldouble Get_lifetime();
    void Set_koncentr(ldouble); //W 1/cm^3
    ldouble Get_koncentr();
    ldouble Get_bar_konc_c();
    ldouble Get_bar_konc_v();
    long Calculate_Gain_Profile();
    long Calculate_Gain_Profile2();
    long Calculate_Gain_Profile_n(std::vector<std::vector<ldouble> > &, ldouble);
    long Calculate_Spont_Profile();
    ldouble Get_qFlc();
    ldouble Get_qFlv();
    ldouble Get_last_point();
    void Set_last_point(ldouble); //Ustawia prawy kraniec przedzialu
    ldouble Get_first_point();
    void Set_first_point(ldouble); //Ustawia lewy kraniec przedzialu
    ldouble Get_step();
    void Set_step(ldouble); //Krok obliczen
    void Set_conduction_depth(ldouble); //Glebokosc studni w pasmie przew.
    ldouble Get_conduction_depth();
    void Set_cond_waveguide_depth(ldouble);
    ldouble Get_cond_waveguide_depth();
    void Set_valence_depth(ldouble); //Glebokosc studni w pasmie walenc.
    ldouble Get_valence_depth();
    void Set_vale_waveguide_depth(ldouble);
    ldouble Get_vale_waveguide_depth();
    ldouble Get_electron_level_depth(int);
    ldouble Get_electron_level_from_bottom(int);
    ldouble Get_heavy_hole_level_depth(int);
    ldouble Get_heavy_hole_level_from_bottom(int);
    ldouble Get_light_hole_level_depth(int);
    ldouble Get_light_hole_level_from_bottom(int);
    int Get_number_of_electron_levels();
    int Get_number_of_heavy_hole_levels();
    int Get_number_of_light_hole_levels();
    void Set_electron_mass_in_plain(ldouble);
    ldouble Get_electron_mass_in_plain();
    void Set_electron_mass_transverse(ldouble);
    ldouble Get_electron_mass_transverse();
    void Set_heavy_hole_mass_in_plain(ldouble);
    ldouble Get_heavy_hole_mass_in_plain();
    void Set_heavy_hole_mass_transverse(ldouble);
    ldouble Get_heavy_hole_mass_transverse();
    void Set_light_hole_mass_in_plain(ldouble);
    ldouble Get_light_hole_mass_in_plain();
    void Set_light_hole_mass_transverse(ldouble);
    ldouble Get_light_hole_mass_transverse();
    void Set_electron_mass_in_barrier(ldouble);
    ldouble Get_electron_mass_in_barrier();
    void Set_heavy_hole_mass_in_barrier(ldouble);
    ldouble Get_heavy_hole_mass_in_barrier();
    void Set_light_hole_mass_in_barrier(ldouble);
    ldouble Get_light_hole_mass_in_barrier();
    void Set_momentum_matrix_element(ldouble); //W eV
    ldouble Get_momentum_matrix_element();
    ldouble Get_gain_at(ldouble);
    ldouble Get_gain_at_n(ldouble, std::vector<std::vector<ldouble> > &, double);
    ldouble Get_bar_gain_at(ldouble);
    ldouble Get_inversion(ldouble E, int i=0);
    ldouble Get_spont_at(ldouble);
    ldouble ** Get_gain_tab(); //Wskaznik do tabl [2][ile_trzeba]
    std::vector<std::vector<ldouble> > & Get_spont_wek();
    //    void przygobl();
    ldouble Find_max_gain();
    ldouble Find_max_gain_n(std::vector<std::vector<ldouble> > &, ldouble);
    ~gain();

    // Marcin Gebski 21.02.2013
    void runPrzygobl();

  private:
    static int Break;
    static const ldouble kB;
    static const ldouble przelm;
    static const ldouble przels;
    static const ldouble ep0;
    static const ldouble c;
    static const ldouble exprng;
    ldouble bladb; //dopuszczalny b³±d bezwzglêdny
    ldouble T; //temperatura
    ldouble n_r; //wsp. za³amania
    ldouble szer; // szeroko¶æ studni
    ldouble szerb; //szerokosc bariery
    ldouble szer_fal; // szeroko¶æ nad studni±
    ldouble Eg; // przerwa energetyczna
    ldouble Mt; // el. macierzowy
    ldouble deltaSO; // split-off
    ldouble tau; // czas ¿ycia
    ldouble konc; // koncentracja
    ldouble barkonc_c; //koncentracja elektronów w barierach
    ldouble barkonc_v; //koncentracja dzur w barierach
    ldouble Efc; // quasi-poziom Fermiego dla pasma walencyjnego
    ldouble Efv; // quasi-poziom Fermiego dla pasma przewodnictwa
    ldouble ** Twzmoc;
    std::vector<std::vector<ldouble> > Tspont;
    long ilpt;
    ldouble enpo,enko;
    ldouble krok;
    int ilwyw;
    char ustawione;
    nosnik el;
    nosnik hh;
    nosnik lh;
    void przygobl();
    void przygobl2();
    void przygobl_n(std::vector<std::vector<ldouble> > &, ldouble);
    ldouble * z_vec_wsk(std::vector<std::vector<ldouble> > &, int);
    ldouble qFlc(); // liczy poziom Fermiego
    ldouble qFlc2();
    ldouble qFlc_n(ldouble);
    ldouble qFlv();
    ldouble qFlv2();
    ldouble qFlv_n(ldouble);
    ldouble element(); // liczy element macierzowy
    ldouble przel_dlug_z_angstr(ldouble);
    ldouble przel_dlug_na_angstr(ldouble);
    ldouble przel_czas_z_psek(ldouble);
    ldouble przel_czas_na_psek(ldouble);
    ldouble przel_konc_z_cm(ldouble);
    ldouble przel_konc_na_cm(ldouble);
    ldouble fc(ldouble); // rozk³ad Fermiego dla p. przewodnictwa
    ldouble fv(ldouble); // rozk³ad Fermiego dla p. walencyjnego
    ldouble L(ldouble,ldouble); // funkcja poszerzaj±ca
    ldouble Lpr(ldouble,ldouble); // jej pochodna
    ldouble gdziepoziomy(ldouble,ldouble *);
    ldouble gdziepoziomy2A(ldouble,ldouble *);
    ldouble gdziepoziomy2B(ldouble,ldouble *);
    ldouble krance(int,ldouble,ldouble);
    ldouble * znajdzpoziomy(nosnik &);
    ldouble * znajdzpoziomy2(nosnik &);
    ldouble gdzieqflv(ldouble,ldouble *);
    ldouble gdzieqflv2(ldouble,ldouble *);
    ldouble gdzieqflv_n(ldouble,ldouble *);
    ldouble gdzieqflc(ldouble,ldouble *);
    ldouble gdzieqflc2(ldouble,ldouble *);
    ldouble gdzieqflc_n(ldouble,ldouble *);
    ldouble kodE(ldouble,ldouble,ldouble);
    ldouble rored(ldouble,ldouble,ldouble);
    ldouble rored2(ldouble,ldouble,ldouble);
    ldouble rored_n(ldouble,ldouble,ldouble,ldouble);
    ldouble dosplotu(ldouble, parametry *);
    ldouble dosplotu2(ldouble, parametry *);
    ldouble dosplotu_n(ldouble, parametry *);
    ldouble dosplotu_spont(ldouble, parametry *);
    ldouble wzmoc_z_posz(ldouble);
    ldouble wzmoc_z_posz2(ldouble);
    ldouble wzmoc_z_posz_n(ldouble,ldouble);
    ldouble spont_z_posz(ldouble t);
    ldouble wzmoc0(ldouble);
    ldouble wzmoc02(ldouble);
    ldouble wzmoc0_n(ldouble,ldouble);
    ldouble spont0(ldouble);
    ldouble Prost(ldouble (gain::*)(ldouble, parametry *),ldouble,ldouble,ldouble,parametry *,ldouble);
    ldouble metsiecz(ldouble (gain::*)(ldouble,ldouble *),ldouble,ldouble,ldouble * =NULL,ldouble prec=1e-7);
  };
  double min_wzmoc(double E,void*);
}
