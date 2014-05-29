#include "tnt/tnt.h"
#include "jama/jama_svd.h"
#include "jama/jama_lu.h"
#include <gsl/gsl_sf_fermi_dirac.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>
#include <list>
#include <complex>
#include <string>
#include <sstream>
#include <boost/math/special_functions/erf.hpp>
#include <boost/regex.hpp> // tylko do wczytywania z pliku
#include <boost/lexical_cast.hpp>

#include <plask/plask.hpp>
using namespace plask;

typedef TNT::Array2D<double> A2D;
typedef TNT::Array1D<double> A1D;

namespace QW{

/*******************************************************************/
class warstwa{
    friend class struktura;
    friend class obszar_aktywny;
    friend class gain;
    //  friend void zrobmacierz(double, std::vector<warstwa> &, A2D & );

    double x_pocz;
    double x_kon;
    double y_pocz;
    double y_kon;
    double pole;
    double nieparab; // alfa nieparabolicznosci
    double nieparab_2; // alfa nieparabolicznosci kwadratowa
    double m_p; // masa prostopadla
    int zera_ffal(double E, double A, double B) const;
    double norma_kwadr(double E, double A, double B) const;
    double tryg_kwadr_pierwotna(double x, double E, double A, double B) const;
    double exp_kwadr_pierwotna(double x, double E, double A, double B) const;
    inline double masa_p(double E) const;

protected:
    double masa_r; // masa rownolegla
    double tryga(double x, double E) const;
    double trygb(double x, double E) const;
    double expa(double x, double E) const;
    double expb(double x, double E) const;
    double Ai(double x, double E) const;
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
    warstwa(double m_p, double m_r, double x_p, double y_p, double x_k, double y_k, double niepar = 0, double niepar_2 = 0);
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
class warstwa_skraj : public warstwa
{
    friend class struktura;
    friend class obszar_aktywny;
    friend class gain;

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

    warstwa_skraj(strona lczyp, double m_p, double m_r, double x, double y);
    warstwa_skraj();
    warstwa_skraj(const warstwa_skraj &);
    double ffala(double x, double E) const;
    double ffalb(double x, double E) const;
    double ffala_prim(double x, double E) const;
    double ffalb_prim(double x, double E) const;
    double funkcjafal(double x, double E, double C) const;
    double funkcjafal_prim(double x, double E, double C) const;
};
/*******************************************************************/
class stan{
    friend class struktura;
    friend class obszar_aktywny;

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
class punkt{
public:
    punkt();
    punkt(double e, double w);
    punkt(const stan &);
    double en;
    double wart;
};
/*******************************************************************/
class struktura{ // struktura poziomow itp

    friend class gain;
    friend class obszar_aktywny;

public:
    enum rodzaj
    {
        el, hh, lh
    };

private:

    static const int mInfo = 0; // LUKASZ

    rodzaj typ;
    double dokl;
    double gora; // skrajna lewa bariera
    double dol;

    warstwa_skraj lewa, prawa;
    std::vector<warstwa> kawalki; // Wewnetrzne warstwy
    std::vector<double> progi; // Poziome bariery dajace falszywe zera
    std::vector<stan> rozwiazania;

    void zrobmacierz(double, A2D & );
    double sieczne(double (struktura::*f)(double), double pocz, double kon);
    double norma_stanu(stan & st);
    double energia_od_k_na_ntym(double k, int nr_war, int n);
    double iloczyn_pierwotna_bezpola(double x, int nr_war, const struktura * struk1, const struktura * struk2, int i, int j);

public:

    static const double przelm;
    static const double przels;
    static const double pi;
    static const double eps0;
    static const double c;
    static const double kB;

    struktura(const std::vector<warstwa*> &, rodzaj); // won't be used LUKASZ
    //struktura(std::ifstream & plik, rodzaj co);

    static double dlugosc_z_A(const double);
    static double dlugosc_na_A(const double);
    static double koncentracja_na_cm_3(const double);

    double czyosobliwa(double E);
    //  double funkcjafal(double x, double E, int n, double A, double B);
    int ilezer_ffal(double E);
    int ilezer_ffal(double E, A2D & V);
    std::vector<double> zageszczanie(punkt p0, punkt pk);
    void szukanie_poziomow(double Ek, double rozdz = 1e-6);
    void normowanie();
    double ilenosnikow(double qFl, double T);
    std::vector<double> koncentracje_w_warstwach(double qFl, double T);
    void funkcje_do_pliku(std::ofstream & plik, double krok);
    void struktura_do_pliku(std::ofstream & plik); // do rysowania studni
    void przesun_energie(double);
    //  double dE_po_dl(size_t nr, chrop ch); //pochodna nr-tego poziomu po szerokosci studni

    void profil(double Ek, double rozdz);
    std::vector<std::vector<double> > rysowanie_funkcji(double E, double x0, double xk, double krok);
};
/*******************************************************************/
class obszar_aktywny
{
    friend class gain;

    double przekr_max; // maksymalna calka przekrycia
    double chrop; // chropowatosc interfejsow, wzgledna (nalezy rozumiec jako wzgledna chropowatosc najwazniejszej studni)
    std::vector<struktura *> pasmo_przew;
    std::vector<struktura *> pasmo_wal;
    std::vector<std::vector<A2D *> > calki_przekrycia;
    std::vector<std::vector<TNT::Array2D<std::vector<double> > * > > calki_przekrycia_kawalki;
    std::vector<double> Egcc; // Przerwy energetyczne (dodatkowe, bo moga byc juz wpisane w igrekach struktur) lewych elektronowych warstw skrajnych wzgledem zerowego pasma przewodnictwa (na ogol jedno 0)
    std::vector<double> Egcv; // Przerwy energetyczne miedzy zerami elektronowymi a dziurowymi (chyba najlepiej, zeby zera byly w skrajnych warstwach)
    std::vector<double> DeltaSO; // DeltySO w warstwach wzgledem zerowego pasma walencyjnego
    std::vector<double> el_mac; // Elementy macierzowe w warstwach

    double element(int nr_war);

public:

    obszar_aktywny(struktura * elektron, const std::vector<struktura *> dziury, double Eg, double DeltaSO, double chropo); // najprostszy konstruktor: jeden elektron i wspolna przerwa

    double min_przerwa_energetyczna();
    //  void policz_calki(const struktura * elektron, const struktura * dziura, A2D & macierz);
    void policz_calki(const struktura * elektron, const struktura * dziura, A2D & macierz, TNT::Array2D<std::vector<double> > & wekt_calk_kaw);
    void policz_calki_kawalki(const struktura * elektron, const struktura * dziura, TNT::Array2D<vector<double> > & macierz); //dopisane na szybko, bo kompilator nie widzial

    double calka_ij(const struktura * elektron, const struktura * dziura, int i, int j, vector<double> & wektor_calk_kaw);
    double iloczyn_pierwotna_bezpola(double x, int nr_war, const struktura * struk1, const struktura * struk2, int i, int j);
    //  void macierze_przejsc();
    void zrob_macierze_przejsc(); // dopisane 2013
    void paryiprzekrycia_dopliku(ofstream & plik, int nr_c, int nr_v);

private:
    static const int mInfo = 0; // LUKASZ
};
/*******************************************************************/
class gain
{
    obszar_aktywny * pasma;
    double nosniki_c, nosniki_v; // gestosc powierzchniowa
    double T;
    double n_r;
    double qFlc; // quasi-poziom Fermiego dla elektronow wzgledem 0 struktur pasma c
    double qFlv; // quasi-poziom Fermiego dla elektronow wzgledem 0 struktur pasma v, w geometrii elektronowej, czyli studnie to gorki
    double szer_do_wzmoc; // szerokosc obszaru czynnego, ktora bedzie model optyczny rozpatrywal
    //  double posz_en; // Poszerzenie energetyczne (sigma w RN) wynikajace z chropowatosci. Uproszczone, wspolne dla wszystkich par stanow

    double sieczne(double (gain::*f)(double), double pocz, double kon);
    double przel_gest_z_cm2(double gest_w_cm2); // gestosc powierzchniowa
    double przel_gest_na_cm2(double gest_w_wew);
    double gdzie_qFlc(double E);
    double gdzie_qFlv(double E);
    double kodE(double E, double mc, double mv);
    double rored(double, double mc, double mv);
    double rored_posz(double E, double E0, double mc, double mv, double sigma);
    double fc(double E);
    double fv(double E);
public:
    gain(); // LUKASZ remember to delete this
    gain(obszar_aktywny * obsz, double konc_pow, double T, double wsp_zal);

    double nosniki_w_c(double Fl);
    double policz_qFlc();

    double nosniki_w_v(double Fl);
    double policz_qFlv();
    double Get_gain_at_n(double, double); // LUKASZ remember to delete this
    double wzmocnienie_od_pary_poziomow(double E, size_t nr_c, int poz_c, size_t nr_v, int poz_v);
    double wzmocnienie_od_pary_pasm(double E, size_t nr_c, size_t nr_v);
    double spont_od_pary_poziomow(double E, size_t nr_c, int poz_c, size_t nr_v, int poz_v);
    double spont_od_pary_pasm(double E, size_t nr_c, size_t nr_v);
    void profil_wzmocnienia_dopliku(std::ofstream & plik, double pocz, double kon, double krok);
    void profil_lumin_dopliku(std::ofstream & plik, double pocz, double kon, double krok);

private:
    static const int mInfo = 0;
};

}
