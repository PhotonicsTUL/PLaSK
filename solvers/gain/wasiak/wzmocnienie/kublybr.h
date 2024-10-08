#if(defined(_MSC_VER) && _MSC_VER < 1600) // RPSMES wymaga Visuala 6 (sprawdzamy < VS2010)
#define RPSMES
#elif(!defined(PLASK))
#define MICHAL
#endif

#include "gsl/gsl_sf_airy.h"
#include "gsl/gsl_sf_fermi_dirac.h"
#include "jama/jama_lu.h"
#include "jama/jama_svd.h"
#include "tnt/tnt.h"

#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <list>
#include <set>
#include <sstream>

#include <sstream>
#include <string>
#include <vector>

#ifdef MICHAL
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#endif

#if(defined(MICHAL) || defined(_DEBUG))
#define LOGUJ
#endif

namespace kubly
{

/*******************************************************************/
// MD: Nie bedziemy drukowac bledow na ekranie i przerywac programu,
// MD: ale bedziemy wyrzucac poprawny wyjatek (GCC wydrukuje komunikat)
class Error : public std::exception
{
  std::ostringstream oss;
  mutable std::string _msg;

 public:
  Error() {}
  Error(const char * msg) { oss << msg; }
  Error(const Error & orig) { oss << orig.oss.str(); }
  template <typename T> Error & operator<<(T arg)
  {
    oss << arg;
    return *this;
  }
  virtual const char * what() const throw()
  {
    _msg = oss.str();
    return _msg.c_str();
  }
  virtual ~Error() throw() {}
};

/*******************************************************************/
typedef TNT::Array2D<double> A2D;
typedef TNT::Array1D<double> A1D;

typedef std::pair<double, double> parad;

bool jaksortpar(parad a, parad b);

/*******************************************************************/
class warstwa
{
  friend class struktura;
  friend class obszar_aktywny;
  friend class wzmocnienie;
  //  friend void zrobmacierz(double, std::vector<warstwa> &, A2D & );

  double x_pocz;
  double x_kon;
  double y_pocz;
  double y_kon;
  double pole; // ladunek razy pole
  double nieparab; // alfa nieparabolicznosci
  double nieparab_2; // alfa nieparabolicznosci kwadratowa
  double m_p; // masa prostopadla
  int zera_ffal(double E, double A, double B, double sasiadl, double sasiadp) const;
  int zera_ffal(double E, double A, double B) const;
  double norma_kwadr(double E, double A, double B) const;
  double tryg_kwadr_pierwotna(double x, double E, double A, double B) const;
  double exp_kwadr_pierwotna(double x, double E, double A, double B) const;
  double airy_kwadr_pierwotna(double x, double E, double A, double B) const;
  inline double masa_p(double E) const;

 protected:
  /*
  double x_pocz;
  double x_kon;
  double y_pocz;
  double y_kon;*/

  warstwa * nast; // wskaznik na sasiadke z prawej
  double masa_r; // masa rownolegla
  double tryga(double x, double E) const;
  double trygb(double x, double E) const;
  double expa(double x, double E) const;
  double expb(double x, double E) const;
  double Ai(double x, double E) const;
  double Ai_skala(double x, double E) const;
  double Bi(double x, double E) const;
  double Bi_skala(double x, double E) const;
  double tryga_prim(double x, double E) const;
  double trygb_prim(double x, double E) const;
  double expa_prim(double x, double E) const;
  double expb_prim(double x, double E) const;
  double Ai_prim(double x, double E) const;
  double Ai_prim_skala(double x, double E) const;
  double Bi_prim(double x, double E) const;
  double Bi_prim_skala(double x, double E) const;
  double funkcjafal(double x, double E, double A, double B) const;
  double funkcjafal_prim(double x, double E, double A, double B) const;
  parad AB_z_wartp(double w, double wp, double E) const;
  double k_kwadr(double E) const;
  double Eodk(double k) const;
  void przesun_igreki(double);

 public:
  warstwa(double m_p, double m_r, double x_p, double y_p, double x_k, double y_k, double niepar = 0,
          double niepar_2 = 0);
  //  warstwa(const warstwa &);
  //  warstwa & operator=(const warstwa &);
  double ffala(double x, double E) const;
  double ffalb(double x, double E) const;
  double ffala_prim(double x, double E) const;
  double ffalb_prim(double x, double E) const;
};
/*******************************************************************/
class warstwa_skraj : public warstwa
{
  friend class struktura;
  friend class obszar_aktywny;
  friend class wzmocnienie;

 public:
  enum strona
  {
    lewa,
    prawa
  };

 private:
  strona lp;
  double masa_p;
  double masa_r;
  double iks;
  double y;

  int zera_ffal(double E, double A, double B) const;
  double norma_kwadr(double E, double A) const;
  void przesun_igreki(double dE);

 public:
  warstwa_skraj(strona lczyp, double m_p, double m_r, double x, double y);
  warstwa_skraj();
  warstwa_skraj(const warstwa_skraj &);
  double ffala(double x, double E) const;
  double ffalb(double x, double E) const;
  double ffala_prim(double x, double E) const;
  double ffalb_prim(double x, double E) const;
  double funkcjafal(double x, double E, double C) const;
  double funkcjafal_prim(double x, double E, double C) const;
  parad AB_z_wartp(double w, double wp, double E) const;
};
/*******************************************************************/
class stan
{
  friend class struktura;
  friend class obszar_aktywny;

  std::vector<double> wspolczynniki;

  stan(double E, A2D & V, int lz);

  stan(double E, std::vector<parad> & W, int lz);
  //
  void przesun_poziom(double);

 public:
  stan();

  std::vector<double> prawdopodobienstwa;
  double poziom;
  int liczba_zer;
};
/*******************************************************************/
class punkt
{
 public:
  punkt();
  punkt(double e, double w);
  punkt(const stan &);
  double en;
  double wart;
};
/*******************************************************************/
class struktura
{ // struktura poziomow itp

  friend class wzmocnienie;
  friend class obszar_aktywny;

 public:
  enum rodzaj
  {
    el,
    hh,
    lh
  };

 private:
  double dokl;
  double gora; // skrajna lewa bariera
  double dol;

  warstwa_skraj lewa, prawa;
  // std::vector<int> gwiazdki;
  std::vector<warstwa> kawalki; // Wewnetrzne warstwy
  std::vector<double> progi; // Poziome bariery dajace falszywe zera
  // std::vector<stan> rozwiazania;

  void zrobmacierz(double, A2D &);
  double sieczne(double (struktura::*f)(double), double pocz, double kon);
  // 1.5
  //   double bisekcja(double (struktura::*f)(double), double pocz, double kon);
  double bisekcja(double (struktura::*f)(double), double pocz, double kon, double dokl = 1e-9);
  //
  double norma_stanu(stan & st);
  double energia_od_k_na_ntym(double k, int nr_war, int n);
  double iloczyn_pierwotna_bezpola(double x, int nr_war, const struktura * struk1, const struktura * struk2, int i,
                                   int j);

 public:
  // MD `rozwiazania` oraz `gwiazdki` sa publiczne
  std::vector<stan> rozwiazania;
  std::vector<int> gwiazdki;

  static const double przelm;
  static const double przels;
  static const double pi;
  static const double eps0;
  static const double c;
  static const double kB;

  rodzaj typ;
  std::string nazwa;
  struktura(const std::vector<warstwa *> &, rodzaj);

  // MD: wczytywanie z pliku wylaczone w PLaSKu i RPSMESie
#ifdef MICHAL
  struktura(std::ifstream & plik, rodzaj co, bool bezlicz = false);
  struktura(std::ifstream & plik, const std::vector<double> & poziomy, rodzaj co);
#endif

  static double dlugosc_z_A(const double);
  static double dlugosc_na_A(const double);
  static double koncentracja_na_cm_3(const double);

  double czyosobliwa(double E);
  //  double funkcjafal(double x, double E, int n, double A, double B);
  int ilezer_ffal(double E);
  int ilezer_ffal(double E, A2D & V);
  std::vector<double> zageszczanie(punkt p0, punkt pk);
  double sprawdz_ciaglosc(double E, A2D & V);
  void szukanie_poziomow(double Ek, double rozdz = 1e-6, bool debug = false);
  void szukanie_poziomow_2(double Ek, double rozdz = 1e-6, bool debug = false);
  void stany_z_tablicy(const std::vector<double> & energie);
  void normowanie();
  double ilenosnikow(double qFl, double T);
  double ilenosnikow(double qFl, double T, std::set<int> ktore_warstwy);
  std::vector<double> koncentracje_w_warstwach(double qFl, double T);
  void poziomy_do_pliku_(std::ofstream & plik, char c, double iRefBand, double iX1,
                         double iX2); // LUKASZ skopiowane ze starych kublow
  void funkcje_do_pliku_(std::ofstream & plik, char c, double iRefBand, double krok,
                         double skala); // LUKASZ skopiowane ze starych kublow
  void funkcje_do_pliku(std::ofstream & plik, double krok);
  void funkcja1_do_pliku(std::ofstream & plik, stan & st, double krok);
  void struktura_do_pliku(std::ofstream & plik); // do rysowania studni
  void przesun_energie(double);
  //  double dE_po_dl(size_t nr, chrop ch); //pochodna nr-tego poziomu po szerokosci studni

  void profil(double Ek, double rozdz);
  std::vector<std::vector<double> > rysowanie_funkcji(double E, double x0, double xk, double krok);
  // dopisek 1.5
  parad sklejanie_od_lewej(double E);
  int ilezer_ffal(double E, std::vector<parad> & W);
  std::vector<parad> wsp_sklejanie_od_lewej(double E);
  double blad_sklejania(double E);
  double poprawianie_poziomu(double E, double DE);
  void szukanie_poziomow_zpoprawka(double Ek, double rozdz);
  void szukanie_poziomow_zpoprawka2(double Ek, double rozdz);
};
/*******************************************************************/
class obszar_aktywny
{
  friend class wzmocnienie;

  // MD: 4 kontruktory dziela wiekszosc kodu - nalezy to wyciagnac do jednej osobnej metody
  void _obszar_aktywny(struktura * elektron, const std::vector<struktura *> & dziury, double Eg,
                       const std::vector<double> * DSO, double chropo, double matelem = 0., double Temp = 300.);
  void _obszar_aktywny(struktura * elektron, const std::vector<struktura *> & dziury, struktura * elektron_m,
                       const std::vector<struktura *> & dziury_m, double Eg, const std::vector<double> * DSO, double br,
                       double matelem = 0., double Temp = 300.);

 public: // MD
  double przekr_max; // maksymalna calka przekrycia
  double chrop; // chropowatosc interfejsow, wzgledna (nalezy rozumiec jako wzgledna chropowatosc najwazniejszej studni)
  double broad; // mnoznik do nierownomiernosci
  std::vector<struktura *> pasmo_przew;
  std::vector<struktura *> pasmo_wal;
  std::vector<struktura *> pasmo_przew_mod;
  std::vector<struktura *> pasmo_wal_mod;
  std::vector<std::vector<A2D *> > calki_przekrycia;
  std::vector<std::vector<TNT::Array2D<std::vector<double> > *> > calki_przekrycia_kawalki;
  std::vector<double> Egcc; // Przerwy energetyczne (dodatkowe, bo moga byc juz wpisane w igrekach struktur) lewych
                            // elektronowych warstw skrajnych wzgledem zerowego pasma przewodnictwa (na ogol jedno 0)
  std::vector<double> Egcv; // Przerwy energetyczne miedzy zerami elektronowymi a dziurowymi (chyba najlepiej, zeby zera
                            // byly w skrajnych warstwach)
  std::vector<double> DeltaSO; // DeltySO w warstwach wzgledem zerowego pasma walencyjnego
  std::vector<double> el_mac; // Elementy macierzowe w warstwach
  double T_ref; // Temperatura odniesienia, dla ktorej ustawione sa przerwy energetyczne
  // double element(int nr_war); // przenioslem do publicznych LUKASZ 23.05.2023

 public:
  obszar_aktywny(struktura * elektron, const std::vector<struktura *> & dziury, double Eg,
                 const std::vector<double> & DSO, double chropo, double matelem = 0., double Temp = 300.);
  obszar_aktywny(struktura * elektron, const std::vector<struktura *> & dziury, struktura * elektron_m,
                 const std::vector<struktura *> dziury_m, double Eg, const std::vector<double> & DSO, double br,
                 double matelem = 0., double Temp = 300.);
  obszar_aktywny(struktura * elektron, const std::vector<struktura *> & dziury, double Eg, double DSO, double chropo,
                 double matelem = 0., double Temp = 300);
#ifdef MICHAL
  obszar_aktywny(struktura * elektron, const std::vector<struktura *> & dziury, struktura * elektron_m,
                 const std::vector<struktura *> dziury_m, double Eg, double DSO, double broad, double Temp = 300,
                 double matelem = 0.);
#else
  obszar_aktywny(struktura * elektron, const std::vector<struktura *> & dziury, struktura * elektron_m,
                 const std::vector<struktura *> dziury_m, double Eg, double DSO, double broad, double matelem = 0.,
                 double Temp = 300);
#endif
  void zapisz_poziomy(std::string nazwa);
  double min_przerwa_energetyczna() const;
  //  void policz_calki(const struktura * elektron, const struktura * dziura, A2D & macierz);
  void policz_calki(const struktura * elektron, const struktura * dziura, A2D & macierz,
                    TNT::Array2D<std::vector<double> > & wekt_calk_kaw);
  void policz_calki_kawalki(const struktura * elektron, const struktura * dziura,
                            TNT::Array2D<vector<double> > & macierz); // dopisane na szybko, bo kompilator nie widzial

  double calka_ij(const struktura * elektron, const struktura * dziura, int i, int j, vector<double> & wektor_calk_kaw);
  double iloczyn_pierwotna_bezpola(double x, int nr_war, const struktura * struk1, const struktura * struk2, int i,
                                   int j);
  double calka_iloczyn_zpolem(int nr_war, const struktura * struk1, const struktura * struk2, int i,
                              int j); // numeryczne calkowanie
  //  void macierze_przejsc();
  void zrob_macierze_przejsc(); // dopisane 2013
  void paryiprzekrycia_dopliku(ofstream & plik, int nr_c, int nr_v);
  double przekrycia_schodkowe(double E, int nr_c, int nr_v);
  void przekrycia_dopliku(ofstream & plik, int nr_c, int nr_v);

  double element(int nr_war); // przenioslem z prywatnych LUKASZ 23.05.2023
  void ustaw_element(double iM); // dodalem metode LUKASZ 23.05.2023
};
/*******************************************************************/
class wzmocnienie
{
 public: // LUKASZ dodalem public
  const obszar_aktywny * pasma;
  double nosniki_c, nosniki_v; // gestosc powierzchniowa
  std::set<int> warstwy_do_nosnikow;
  double T;
  std::string ch_br; // LUKASZ skopiowane ze starych kublow
  std::vector<double> Egcv_T;
  double n_r;
  double qFlc; // quasi-poziom Fermiego dla elektronow wzgledem 0 struktur pasma c
  double qFlv; // quasi-poziom Fermiego dla elektronow wzgledem 0 struktur pasma v, w geometrii elektronowej, czyli
               // studnie to gorki
  // double posz_en; // Poszerzenie energetyczne (sigma w RN) wynikajace z chropowatosci. Uproszczone, wspolne dla
  // wszystkich par stanow

  void ustaw_przerwy(double poprawka = 0.); // ustawia przerwy energetyczne dla podanej temperatury
  double sieczne(double (wzmocnienie::*f)(double), double pocz, double kon);
  double przel_gest_z_cm2(double gest_w_cm2); // gestosc powierzchniowa
  double przel_gest_na_cm2(double gest_w_wew);
  double gdzie_qFlc(double E);
  double gdzie_qFlv(double E);
  double kodE(double E, double mc, double mv);
  double rored(double, double mc, double mv);
  double erf_dorored(double E, double E0, double sigma);
  double rored_posz(double E, double E0, double mc, double mv, double sigma);
  double fc(double E);
  double fv(double E);
  double posz_z_chrop(size_t nr_c, int poz_c, size_t nr_v, int poz_v);
  double posz_z_br(size_t nr_c, int poz_c, size_t nr_v, int poz_v);
  double nosniki_w_c(double Fl);
  void policz_qFlc();
  double nosniki_w_v(double Fl);
  void policz_qFlv();

  // MD: jak juz wprowadzamy mozliwosc wyboru wersji, to opiszmy ja porzadnie
  enum Wersja
  {
    Z_CHROPOWATOSCIA,
    Z_POSZERZENIEM
  };

 private:
  Wersja wersja; // MD

 public:
  double szer_do_wzmoc; // szerokosc obszaru czynnego, ktora bedzie model optyczny rozpatrywal
  wzmocnienie(obszar_aktywny * obsz, double konc_pow, double T, double wsp_zal, double poprawkaEg = 0.,
              double szdowzm = -1., Wersja wersja = Z_POSZERZENIEM);

  std::vector<double> koncentracje_elektronow_w_warstwach();
  std::vector<double> koncentracje_dziur_w_warstwach();
  double pozFerm_przew();
  double pozFerm_wal();
  double rozn_poz_Ferm();
  double szerdowzmoc();
  double wzmocnienie_od_pary_poziomow(double E, size_t nr_c, int poz_c, size_t nr_v, int poz_v,
                                      double polar); // MD - dodalem polar 2024-09-02
  double wzmocnienie_od_pary_pasm(double E, size_t nr_c, size_t nr_v, double polar); // MD - dodalem polar 2024-09-02
  double spont_od_pary_poziomow(double E, size_t nr_c, int poz_c, size_t nr_v, int poz_v, double polar);
  double spont_od_pary_pasm(double E, size_t nr_c, size_t nr_v, double polar);
  double wzmocnienie_calk_ze_splotem(double E, double b, double polar, // MD - dodalem polar 2024-09-02
                                     double blad = 0.02); // podzial na kawalek o promieniu Rb wokol 0 i reszte
  double wzmocnienie_calk_bez_splotu(double E, double polar); // MD - dodalem polar 2024-09-02
  double
  wzmocnienie_calk_bez_splotu_L(double lambda,
                                double polar); // dodalem metode LUKASZ LUKI 23.05.2023, MD - dodalem polar 2024-09-02
  double lumin(double E, double polar = 2.); // polar = 0. -> TE,   polar = 1. -> TM,   polar = 2. -> TE + TM,
  void profil_wzmocnienia_ze_splotem_dopliku(std::ofstream & plik, double pocz, double kon, double krok, double b);
  void profil_wzmocnienia_bez_splotu_dopliku(std::ofstream & plik, double pocz, double kon, double krok);
  void profil_wzmocnienia_bez_splotu_dopliku_L(std::ofstream & plik, double pocz, double kon,
                                               double krok); // dodalem metode LUKASZ LUKI 23.05.2023
  void profil_lumin_dopliku(std::ofstream & plik, double pocz, double kon, double krok);
  void profil_lumin_dopliku_L(std::ofstream & plik, double pocz, double kon,
                              double krok); // dodalem metode LUKASZ LUKI 5.09.2023

  double moc_lumin();
  static double L(double x, double b);
};

} // namespace kubly

#ifndef PLASK
using namespace kubly;
#endif
