#include "tnt/tnt.h"
#include "jama/jama_svd.h"
#include "jama/jama_lu.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>
#include <list>
#include <complex>
#include <string>
#include <sstream>
#include <boost/math/special_functions/airy.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/lexical_cast.hpp>

#include "fd.h"

#include <plask/plask.hpp>

typedef TNT::Array2D<double> A2D;
typedef TNT::Array1D<double> A1D;

namespace QW{

/*******************************************************************/
class Warstwa {
    friend class Struktura;
    friend class ObszarAktywny;
    friend class Gain;
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
//   Warstwa * nast; // wskaznik na sasiadke z prawej
  double masa_r; // masa rownolegla
  double tryga(double x, double E) const;
  double trygb(double x, double E) const;
  double expa(double x, double E) const;
  double expb(double x, double E) const;
  double Ai(double x, double E) const;
  //double Ai_skala(double x, double E) const;
  double Bi(double x, double E) const;
  double tryga_prim(double x, double E) const;
  double trygb_prim(double x, double E) const;
  double expa_prim(double x, double E) const;
  double expb_prim(double x, double E) const;
  double Ai_prim(double x, double E) const;
  double Bi_prim(double x, double E) const;
  double funkcjafal(double x, double E, double A, double B) const;
  double funkcjafal_prim(double x, double E, double A, double B) const;
  double k_kwadr(double E) const;
  double Eodk(double k) const;
  void przesun_igreki(double);

public:
    Warstwa(double m_p, double m_r, double x_p, double y_p, double x_k, double y_k, double niepar = 0, double niepar_2 = 0);
    //  warstwa(const warstwa &);
    //  warstwa & operator=(const warstwa &);
    double ffala(double x, double E) const;
    double ffalb(double x, double E) const;
    double ffala_prim(double x, double E) const;
    double ffalb_prim(double x, double E) const;

private:
    static const int mInfo = 0; // LUKASZ
};
/*******************************************************************/
class WarstwaSkraj : public Warstwa
{
    friend class Struktura;
    friend class ObszarAktywny;
    friend class Gain;

public:

    enum strona
    {
        lewa, prawa
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

    WarstwaSkraj(strona lczyp, double m_p, double m_r, double x, double y);
    WarstwaSkraj();
    WarstwaSkraj(const WarstwaSkraj &);
    double ffala(double x, double E) const;
    double ffalb(double x, double E) const;
    double ffala_prim(double x, double E) const;
    double ffalb_prim(double x, double E) const;
    double funkcjafal(double x, double E, double C) const;
    double funkcjafal_prim(double x, double E, double C) const;
};
/*******************************************************************/
class stan{
    friend class Struktura;
    friend class ObszarAktywny;

    std::vector<double> wspolczynniki;

    stan(double E, A2D & V, int lz);

    void przesun_poziom(double);

public:

    stan();

    std::vector<double> prawdopodobienstwa;
    double poziom;
    int liczba_zer;
};
/*******************************************************************/
class Punkt {
public:
    Punkt();
    Punkt(double e, double w);
    Punkt(const stan &);
    double en;
    double wart;
};
/*******************************************************************/
class Struktura { // struktura poziomow itp

    friend class Gain;
    friend class ObszarAktywny;

public:
    enum rodzaj
    {
        el, hh, lh
    };

    static const int mInfo = 0; // LUKASZ

    rodzaj typ;
    double dokl;
    double gora; // skrajna lewa bariera
    double dol;

    WarstwaSkraj lewa, prawa;
    std::vector<Warstwa> kawalki; // Wewnetrzne warstwy
    std::vector<double> progi; // Poziome bariery dajace falszywe zera
    std::vector<stan> rozwiazania;

    void zrobmacierz(double, A2D & );
    double sieczne(double (Struktura::*f)(double), double pocz, double kon);
    double norma_stanu(stan & st);
    double energia_od_k_na_ntym(double k, int nr_war, int n);
    double iloczyn_pierwotna_bezpola(double x, int nr_war, const Struktura * struk1, const Struktura * struk2, int i, int j);

public:

    static const double przelm;
    static const double przels;
    static const double pi;
    static const double eps0;
    static const double c;
    static const double kB;

    Struktura(const std::vector<std::unique_ptr<Warstwa>>&, rodzaj);
    //struktura(std::ifstream & plik, rodzaj co); // won't be used LUKASZ

    static double dlugosc_z_A(const double);
    static double dlugosc_na_A(const double);
    static double koncentracja_na_cm_3(const double);

    double czyosobliwa(double E);
    //  double funkcjafal(double x, double E, int n, double A, double B);
    int ilezer_ffal(double E);
    int ilezer_ffal(double E, A2D & V);
    std::vector<double> zageszczanie(Punkt p0, Punkt pk);
    void szukanie_poziomow(double Ek, double rozdz = 1e-6);
    void normowanie();
    double ilenosnikow(double qFl, double T);
    std::vector<double> koncentracje_w_warstwach(double qFl, double T);
    void funkcje_do_pliku(std::ofstream & plik, double krok);
    void showEnergyLevels(std::string iStr, double iNoOfQWs);
    void struktura_do_pliku(std::ofstream & plik); // do rysowania studni
    void przesun_energie(double);
    //  double dE_po_dl(size_t nr, chrop ch); //pochodna nr-tego poziomu po szerokosci studni

//     void profil(double Ek, double rozdz);
    std::vector<std::vector<double> > rysowanie_funkcji(double E, double x0, double xk, double krok);
};
/*******************************************************************/
class ObszarAktywny
{
    friend class Gain;

    double przekr_max; // maksymalna calka przekrycia
    double chrop; // chropowatosc interfejsow, wzgledna (nalezy rozumiec jako wzgledna chropowatosc najwazniejszej studni)
    std::vector<Struktura *> pasmo_przew;
    std::vector<Struktura *> pasmo_wal;
    std::vector<std::vector<A2D *> > calki_przekrycia;
    std::vector<std::vector<TNT::Array2D<std::vector<double> > * > > calki_przekrycia_kawalki;
    std::vector<double> Egcc; // Przerwy energetyczne (dodatkowe, bo moga byc juz wpisane w igrekach struktur) lewych elektronowych warstw skrajnych wzgledem zerowego pasma przewodnictwa (na ogol jedno 0)
    std::vector<double> Egcv; // Przerwy energetyczne miedzy zerami elektronowymi a dziurowymi (chyba najlepiej, zeby zera byly w skrajnych warstwach)
    std::vector<double> DeltaSO; // DeltySO w warstwach wzgledem zerowego pasma walencyjnego
    std::vector<double> el_mac; // Elementy macierzowe w warstwach
    double T_ref; // Temperatura odniesienia, dla której ustawione sa przerwy energetyczne
    double element(int nr_war);

public:

    ObszarAktywny(Struktura * elektron, const std::vector<Struktura *> dziury, double Eg, std::vector<double> DeltaSO, double chropo, double Temp, double iMatrixElemScFact, bool iShowM); // najprostszy konstruktor: jeden elektron i wspolna przerwa

    double min_przerwa_energetyczna() const;
    //  void policz_calki(const struktura * elektron, const struktura * dziura, A2D & macierz);
    void policz_calki(const Struktura * elektron, const Struktura * dziura, A2D & macierz, TNT::Array2D<std::vector<double> > & wekt_calk_kaw);
    void policz_calki_kawalki(const Struktura * elektron, const Struktura * dziura, TNT::Array2D<vector<double> > & macierz); //dopisane na szybko, bo kompilator nie widzial

    double calka_ij(const Struktura * elektron, const Struktura * dziura, int i, int j, vector<double> & wektor_calk_kaw);
    double iloczyn_pierwotna_bezpola(double x, int nr_war, const Struktura * struk1, const Struktura * struk2, int i, int j);
double calka_iloczyn_zpolem(int nr_war, const Struktura * struk1, const Struktura * struk2, int i, int j); // numeryczne calkowanie
    //  void macierze_przejsc();
    void zrob_macierze_przejsc(); // dopisane 2013
    void paryiprzekrycia_dopliku(ofstream & plik, int nr_c, int nr_v);

private:
    static const int mInfo = 0; // LUKASZ
};
/*******************************************************************/
class PLASK_SOLVER_API Gain
{
    plask::shared_ptr<const ObszarAktywny> pasma;
    double nosniki_c, nosniki_v; // gestosc powierzchniowa
    double T;
    std::vector<double> Egcv_T;
    double n_r;
    double mEgClad;
    double qFlc; // quasi-poziom Fermiego dla elektronow wzgledem 0 struktur pasma c
    double qFlv; // quasi-poziom Fermiego dla elektronow wzgledem 0 struktur pasma v, w geometrii elektronowej, czyli studnie to gorki
    double szer_do_wzmoc; // szerokosc obszaru czynnego, ktora bedzie model optyczny rozpatrywal
    //  double posz_en; // Poszerzenie energetyczne (sigma w RN) wynikajace z chropowatosci. Uproszczone, wspolne dla wszystkich par stanow
    void ustaw_przerwy(); // ustawia przerwy energetyczne dla podanej temperatury
    double sieczne(double (Gain::*f)(double), double pocz, double kon);
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
public:
    Gain(); // LUKASZ remember to delete this
    void setGain(plask::shared_ptr<const ObszarAktywny> obsz, double konc_pow, double T, double wsp_zal, double EgClad);
    void setEgClad(double iEgClad);
    void setNsurf(double iNsurf);
    double nosniki_w_c(double Fl);
    double policz_qFlc();

    double nosniki_w_v(double Fl);
    double policz_qFlv();
    double getT();
    double Get_gain_at_n(double E, double hQW, double iL, double iTau); // LUKASZ
    double Get_luminescence_at_n(double E, double hQW, double iL); // LUKASZ

    double wzmocnienie_od_pary_poziomow(double E, size_t nr_c, int poz_c, size_t nr_v, int poz_v);
    double wzmocnienie_od_pary_pasm(double E, size_t nr_c, size_t nr_v);
    double spont_od_pary_poziomow(double E, size_t nr_c, int poz_c, size_t nr_v, int poz_v);
    double spont_od_pary_pasm(double E, size_t nr_c, size_t nr_v);
    double wzmocnienie_calk_ze_splotem(double E, double b, double blad = 0.02); // podzial na kawalek o promieniu Rb wokol 0 i reszte
    double wzmocnienie_calk_bez_splotu(double E);
    double luminescencja_calk(double E); // LUKASZ
    void profil_wzmocnienia_ze_splotem_dopliku(std::ofstream & plik, double pocz, double kon, double krok, double b);
    void profil_wzmocnienia_bez_splotu_dopliku(std::ofstream & plik, double pocz, double kon, double krok);
    void profil_lumin_dopliku(std::ofstream & plik, double pocz, double kon, double krok);
    double moc_lumin();
    static double L(double x, double b);

private:
    static const int mInfo = 0;
};

}
