#include "kubly.h"

using QW::warstwa;
using QW::warstwa_skraj;
using QW::stan;
using QW::punkt;
using QW::struktura;
using QW::obszar_aktywny;
using QW::gain;

const double struktura::przelm = 10*1.05459/(sqrt(1.60219*9.10956));
const double struktura::przels = 1.05459/1.60219*1e-3;
const double struktura::kB = 1.38062/1.60219*1e-4;
const double struktura::eps0 = 8.8542*1.05459/(100*1.60219*sqrt(1.60219*9.10956));
const double struktura::c = 300*sqrtl(9.10956/1.60219);
const double struktura::pi=4*atan(1.);

/*****************************************************************************
warstwa::warstwa(const warstwa & war) : x_pocz(war.x_pocz), x_kon(war.x_kon), y_pocz(war.y_pocz), y_kon(war.y_kon), masa(war.masa) {}
*****************************************************************************
warstwa & warstwa::operator=(const warstwa & war)
{
  return *this;
}
*****************************************************************************/
warstwa::warstwa(double m_p, double m_r, double x_p, double y_p, double x_k, double y_k, double niepar, double niepar2) : x_pocz(x_p/struktura::przelm), x_kon(x_k/struktura::przelm), y_pocz(y_p), y_kon(y_k), nieparab(niepar), nieparab_2(niepar2), m_p(m_p), nast(NULL), masa_r(m_r) // Położenia w A
{
  if(x_k <= x_p)
    {
      std::cerr<<"Złe dane!\n";
      std::cerr<<"pocz = "<<x_p<<"\tkoniec = "<<x_k<<"\tmasa_p = "<<m_p<<"\n";
      abort();
    }
  //  std::clog<<"x_pocz = "<<x_pocz<<"\tx_kon = "<<x_kon<<"\n";
  pole = (y_kon - y_pocz)/(x_kon - x_pocz);
}
/*****************************************************************************/
warstwa_skraj::warstwa_skraj(strona lczyp, double m_p, double m_r, double x0, double y0) : warstwa(m_p, m_r, (lczyp == lewa)?x0 - 1:x0, y0, (lczyp == lewa)?x0:x0 + 1, y0), lp(lczyp), masa_p(m_p), masa_r(m_r), iks(x0/struktura::przelm), y(y0) {} // x0 w A
/*****************************************************************************/
warstwa_skraj::warstwa_skraj() : warstwa(0., 0., 0., 0., 1., 0.)
{
}
/*****************************************************************************/
warstwa_skraj::warstwa_skraj(const warstwa_skraj & war) : warstwa(war.masa_p, war.masa_r, (war.lp == lewa)?war.iks - 1:war.iks, war.y, (war.lp == lewa)?war.iks:war.iks + 1, war.y), lp(war.lp), iks(war.iks), y(war.y) {}
/*****************************************************************************/
inline double warstwa::masa_p(double E) const
{
  double wynik;
  double Ek = E - (y_pocz + y_kon)/2;
  if( (nieparab == 0 && nieparab_2 == 0) || Ek < 0 )
    {
      wynik = m_p;
    }
  else
    {
      if( (nieparab_2 < 0) && (Ek > -nieparab/(2*nieparab_2)) ) // czy nie przesz<B3>o na opadaj<B1>c<B1> cz<EA><B6><E6>?
	wynik = m_p*(1. -nieparab*nieparab/(4*nieparab_2));
      else
	wynik = (1. + nieparab*Ek + nieparab_2*Ek*Ek)*m_p;
    }
  return wynik;
}
/*****************************************************************************/
void warstwa::przesun_igreki(double dE)
{
  y_pocz += dE;
  y_kon += dE;
}
/*****************************************************************************/
double warstwa::ffala(double x, double E) const
{
  double wartosc;
  //  std::clog<<" E = "<<E<<"\n";
  if(pole !=0)
    {
      //      std::clog<<"\n początek warstwy w "<<x_pocz<<" Airy";
      wartosc = Ai(x, E);
    }
  else
    {
      if(E >= y_pocz)
	{
	  //	  std::clog<<"\n początek warstwy w "<<x_pocz<<" tryg";
	  wartosc = tryga(x, E);
	}
      else
	{
	  //	  std::clog<<"\n początek warstwy w "<<x_pocz<<" exp";
	  wartosc = expa(x, E);
	}
    }
  return wartosc;
}
/*****************************************************************************/
double warstwa::ffala_prim(double x, double E) const
{
  double wartosc;
  if(pole !=0)
    {
      wartosc = Ai_prim(x, E);
    }
  else
    {
      if(E >= y_pocz)
	{
	  wartosc = tryga_prim(x, E);
	}
      else
	{
	  wartosc = expa_prim(x, E);
	}
    }
  return wartosc;
}
/*****************************************************************************/
double warstwa::ffalb(double x, double E) const
{
  double wartosc;
  if(pole !=0)
    {
      wartosc = Bi(x, E);
    }
  else
    {
      if(E >= y_pocz)
	{
	  wartosc = trygb(x, E);
	}
      else
	{
	  wartosc = expb(x, E);
	}
    }
  return wartosc;
}
/*****************************************************************************/
double warstwa::ffalb_prim(double x, double E) const
{
  double wartosc;
  if(pole !=0)
    {
      wartosc = Bi_prim(x, E);
    }
  else
    {
      if(E >= y_pocz)
	{
	  wartosc = trygb_prim(x, E);
	}
      else
	{
	  wartosc = expb_prim(x, E);
	}
    }
  return wartosc;
}
/*****************************************************************************/
double warstwa::tryga(double x, double E) const
{
  if((y_kon != y_pocz) || (E < y_pocz))
    {
      throw "tryga: Bad function";
    }
  double k = sqrt(2*masa_p(E)*(E-y_pocz));
  return sin(k*x);
}
/*****************************************************************************/
double warstwa::tryga_prim(double x, double E) const
{
  if((y_kon != y_pocz) || (E < y_pocz))
    {
      throw "tryga_prim: Bad function";
    }
  double k = sqrt(2*masa_p(E)*(E-y_pocz));
  return k*cos(k*x);
}
/*****************************************************************************/
double warstwa::trygb(double x, double E) const
{
  if((y_kon != y_pocz) || (E < y_pocz))
    {
      throw "trygb: Bad function";
    }
  double k = sqrt(2*masa_p(E)*(E-y_pocz));
  return cos(k*x);
}
/*****************************************************************************/
double warstwa::trygb_prim(double x, double E) const
{
  if((y_kon != y_pocz) || (E < y_pocz))
    {
      throw "trygb_prim: Bad function";
    }
  double k = sqrt(2*masa_p(E)*(E-y_pocz));
  return -k*sin(k*x);
}
/*****************************************************************************/
double warstwa::tryg_kwadr_pierwotna(double x, double E, double A, double B) const
{
  if((y_kon != y_pocz) || (E <= y_pocz))
    {
      throw "tryg_kwadr_pierwotna: Bad function";
    }
  double k = sqrt(2*masa_p(E)*(E-y_pocz));
  double si2 = sin(2*k*x);
  double co = cos(k*x);
  return (A*A + B*B)*x/2  + (si2*(B*B - A*A)/4 - A*B*co*co)/k;
}
/*****************************************************************************/
double warstwa::expa(double x, double E) const
{
  if((y_kon != y_pocz) || (E > y_pocz))
    {
      throw "expa: Bad function";
    }
  double kp = sqrt(2*masa_p(E)*(y_pocz-E));
  return exp(-kp*(x - x_pocz));
}
/*****************************************************************************/
double warstwa::expa_prim(double x, double E) const
{
  if((y_kon != y_pocz) || (E > y_pocz))
    {
      throw "expa_prim: Bad function";
    }
  double kp = sqrt(2*masa_p(E)*(y_pocz-E));
  return -kp*exp(-kp*(x - x_pocz));
}
/*****************************************************************************/
double warstwa::expb(double x, double E) const
{
  if((y_kon != y_pocz) || (E > y_pocz))
    {
      throw "expb: Bad function";
    }
  double kp = sqrt(2*masa_p(E)*(y_pocz - E));
  return exp(kp*(x - x_kon));
}
/*****************************************************************************/
double warstwa::expb_prim(double x, double E) const
{
  if((y_kon != y_pocz) || (E > y_pocz))
    {
      throw "expb_prim: Bad function";
    }
  double kp = sqrt(2*masa_p(E)*(y_pocz-E));
  return kp*exp(kp*(x - x_kon));
}
/*****************************************************************************/
double warstwa::exp_kwadr_pierwotna(double x, double E, double A, double B) const
{
  if((y_kon != y_pocz) || (E > y_pocz))
    {
      throw "exp_kwadr_pierwotna: Bad function";
    }
  double kp = sqrt(2*masa_p(E)*(y_pocz-E));
  double b = expb(x, E);
  double a = expa(x, E);
  return (B*B*b*b - A*A*a*a)/(2*kp) + 2*A*B*x*exp(kp*(x_pocz - x_kon));
}
/*****************************************************************************/
double warstwa::Ai(double x, double E) const
{
  if(y_kon == y_pocz)
      throw "Ai: Bad funtion";
  // równanie: -f''(x) + (b + ax)f(x) = 0
  // a = 2m*pole/h^2
  // b = 2m(U - E)/h^2
  // rozw f(x) = Ai( (ax+b)/a^{2/3} ) = Ai( a^{1/3} (x + b/a^{2/3}) )
  double U = y_pocz - pole*x_pocz;
  double a13 = (pole > 0)?pow(2*masa_p(E)*pole,1./3):-pow(-2*masa_p(E)*pole,1./3); // a^{1/3} 
  double b_a23 = (U - E)/pole;
  double arg = a13*(x + b_a23);
  //  std::cerr<<"\narg_Ai = "<<arg;
  return boost::math::airy_ai(arg);
}
///*****************************************************************************/
//double warstwa::Ai_skala(double x, double E) const
//{
//  if(y_kon == y_pocz)
//      throw "Ai_skala: Bad funtion";
//  // równanie: -f''(x) + (b + ax)f(x) = 0
//  // a = 2m*pole/h^2
//  // b = 2m(U - E)/h^2
//  // rozw f(x) = Ai( (ax+b)/a^{2/3} ) = Ai( a^{1/3} (x + b/a^{2/3}) )
//  double U = y_pocz - pole*x_pocz;
//  double a13 = (pole > 0)?pow(2*masa_p(E)*pole,1./3):-pow(-2*masa_p(E)*pole,1./3); // a^{1/3}
//  double b_a23 = (U - E)/pole;
//  double arg = a13*(x + b_a23);
//  return gsl_sf_airy_Ai_scaled(arg, GSL_PREC_DOUBLE); // Na razie nie ma
//}
/*****************************************************************************/
double warstwa::Ai_prim(double x, double E) const
{
  if(y_kon == y_pocz)
      throw "Ai_prim: Bad funtion";
  double U = y_pocz - pole*x_pocz;
  double a13 = (pole > 0)?pow(2*masa_p(E)*pole,1./3):-pow(-2*masa_p(E)*pole,1./3); // a^{1/3} 
  double b_a23 = (U - E)/pole;
  double arg = a13*(x + b_a23);
  //  std::cerr<<"\nx = "<<x<<" x_pocz = "<<x_pocz<<" pole = "<<pole<<" a13 = "<<a13<<" b_a23 = "<<b_a23<<" arg_Ai' = "<<arg<<" inaczej (w x_pocz)"<<(2*masa_p(E)*(y_pocz - E)/(a13*a13))<<" inaczej (w x_kon)"<<(2*masa_p(E)*(y_kon - E)/(a13*a13));
  return a13 * boost::math::airy_ai_prime(arg);
}
/*****************************************************************************/
double warstwa::Bi(double x, double E) const
{
  if(y_kon == y_pocz)
      throw "Bi: Bad funtion";
  // równanie: -f''(x) + (b + ax)f(x) = 0
  // a = 2m*pole/h^2
  // b = 2m(U - E)/h^2
  // rozw f(x) = Ai( (ax+b)/a^{2/3} ) = Ai( a^{1/3} (x + b/a^{2/3}) )
  double U = y_pocz - pole*x_pocz;
  double a13 = (pole > 0)?pow(2*masa_p(E)*pole,1./3):-pow(-2*masa_p(E)*pole,1./3); // a^{1/3} 
  double b_a23 = (U - E)/pole;
  double arg = a13*(x + b_a23);
  //  std::cerr<<"\narg_Bi = "<<arg;
  return boost::math::airy_bi(arg);
}
/*****************************************************************************/
double warstwa::Bi_prim(double x, double E) const
{
  if(y_kon == y_pocz)
      throw "Bi_prim: Bad funtion";
  double U = y_pocz - pole*x_pocz;
  double a13 = (pole > 0)?pow(2*masa_p(E)*pole,1./3):-pow(-2*masa_p(E)*pole,1./3); // a^{1/3} 
  double b_a23 = (U - E)/pole;
  double arg = a13*(x + b_a23);
  //  std::cerr<<"\narg_Bi' = "<<arg;
  return a13 * boost::math::airy_bi_prime(arg);
}
/*****************************************************************************/
double warstwa::airy_kwadr_pierwotna(double x, double E, double A, double B) const
{
  if(y_kon == y_pocz)
      throw "airy_kwadr_pierwotna: Bad funtion";
  double U = y_pocz - pole*x_pocz;
  double b_a23 = (U - E)/pole;
  double a = 2*masa_p(E)*pole;
  double f = funkcjafal(x, E, A, B);
  double fp = funkcjafal_prim(x, E, A, B);
  return (x + b_a23)*f*f - fp*fp/a;
}
/*****************************************************************************/
double warstwa::k_kwadr(double E) const // Zwraca k^2, ujemne dla energii spod bariery (- kp^2)
{
  double wartosc;
  if(pole !=0)
    {
      throw "Airy functions not computed";
    }
  else
    {
	  wartosc = 2*masa_p(E)*(E-y_pocz);
    }
  return wartosc;
}
/*****************************************************************************/
int warstwa::zera_ffal(double E, double A, double B, double sasiad_z_lewej, double sasiad_z_prawej) const // wartości sąsiadów po to, żeby uniknąć kłopotów, kiedy zero wypada na łączeniu
{
  int tylezer = 0;
  double wart_kon = (funkcjafal(x_kon, E, A, B) + sasiad_z_prawej)/2; // Uśrednienie dla uniknięcia kłopotów z zerami na łączeniu, gdzie malutka nieciągłość może generować zmiany znaków
  double wart_pocz = (funkcjafal(x_pocz, E, A, B) + sasiad_z_lewej)/2;
  double iloczyn = wart_pocz*wart_kon;
  //std::cerr<<"\nwart na koncach: "<<funkcjafal(x_pocz, E, A, B)<<", "<<funkcjafal(x_kon, E, A, B);
  //  std::cerr<<"\npo usrednieniu: "<<wart_pocz<<", "<<wart_kon;
  if(pole !=0)
    {
      double a13 = (pole > 0)?pow(2*masa_p(E)*pole,1./3):-pow(-2*masa_p(E)*pole,1./3); // a^{1/3} 
      double U = y_pocz - pole*x_pocz;
      double b_a23 = (U - E)/pole;
      double arg1, arg2, argl, argp, x1, x2, xlew, xpra;
      int nrza, nrprzed; // nrza do argp, nrprzed do argl
      arg1 = a13*(x_pocz + b_a23);
      arg2 = a13*(x_kon + b_a23);
      argl = std::min(arg1, arg2);
      argp = std::max(arg1, arg2);
      nrza=1;
      double z1 = -1.174; // oszacowanie pierwszego zera B1
      double dz = -2.098; // oszacowanie odstępu między perwszymi dwoma zerami
      //  nrprzed=1;
      //      nrprzed = floor((argl-z1)/dz + 1); // oszacowanie z dołu numeru miejsca zerowego
      nrprzed = floor((argp-z1)/dz + 1);
      nrprzed = (nrprzed >= 1)?nrprzed:1;
      int tymcz=0;
      double ntezero = boost::math::airy_bi_zero<double>(nrprzed);
      std::cerr<<"\nU = "<<U<<" a13 = "<<a13<<" b_a23 = "<<b_a23<<" argl = "<<argl<<" argp = "<<argp<<" ntezero = "<<ntezero<<" nrprzed = "<<nrprzed;
      double brak; // oszacowanie z dołu braku
      long licznik = 0;
      //      while(ntezero>=argl)
      while(ntezero>=argp)
	{
	  if(nrprzed>2)
	    {
          dz = ntezero - boost::math::airy_bi_zero<double>(nrprzed-1);
	      brak = (argp-ntezero)/dz;
	      if(brak > 2.) //jeśli jeszcze daleko
		{
		  nrprzed = nrprzed + floor(brak);
		}
	      else nrprzed++;
	    }
	  else
	    nrprzed++;
      ntezero = boost::math::airy_bi_zero<double>(nrprzed);
	  licznik++;
	  std::cerr<<"\nnrprzed = "<<nrprzed<<" ntezero = "<<ntezero;
	}
      //  std::cerr<<"\tnrprzed kon "<<nrprzed<<" po "<<licznik<<" dodawaniach\n";
      nrza=nrprzed;
      nrprzed--;
      //      while(boost::math::airy_bi_zero<double>(nrza)>=argp)
      while(boost::math::airy_bi_zero<double>(nrza)>=argl)
	{
	  nrza++;
      std::cerr<<"\nnrza = "<<nrza<<" ntezero = "<<boost::math::airy_bi_zero<double>(nrza);
	}
      std::cerr<<"\nnrprzed = "<<nrprzed<<" nrza = "<<nrza;
      /*
      std::cerr<<"\nnrprzed = "<<nrprzed<<" nrza = "<<nrza;
      x1 = b_a23 - boost::math::airy_bi_zero<double>(nrprzed+1)/a13; // polozenia skrajnych zer Ai w studni
      x2 = b_a23 - boost::math::airy_bi_zero<double>(nrza-1)/a13;
      xlew = std::min(x1, x2);
      xpra = std::max(x1, x2);
      std::cerr<<"\txlew="<<struktura::dlugosc_na_A(xlew)<<" xpra="<<struktura::dlugosc_na_A(xpra);
      tylko do testów tutaj  */

      if(nrza-nrprzed>=2)
	{
	  tymcz=nrza-nrprzed-2;
      x1 = -b_a23 + boost::math::airy_bi_zero<double>(nrprzed+1)/a13; // polozenia skrajnych zer Ai w studni
      x2 = -b_a23 + boost::math::airy_bi_zero<double>(nrza-1)/a13;
	  xlew = std::min(x1, x2);
	  xpra = std::max(x1, x2);
	  std::cerr<<"\n xlew="<<struktura::dlugosc_na_A(xlew)<<" xpra="<<struktura::dlugosc_na_A(xpra);
	  //      std::cerr<<"\n A "<<funkcja_z_polem_do_oo(struk.punkty[i],E,funkcja,struk)<<" "<<funkcja_z_polem_do_oo(xl,E,funkcja,struk);
	  if(wart_pocz*funkcjafal(xlew, E, A, B) < 0)
	    //	  if(funkcja_z_polem_do_oo(struk.punkty[i],E,funkcja,struk)*funkcja_z_polem_do_oo(xl,E,funkcja,struk)<0)
	    tymcz++;
	  if(wart_kon*funkcjafal(xpra, E, A, B) < 0)
	    tymcz++;
	}
      else
	{
	  //      std::cerr<<"\n C "<<funkcja_z_polem_do_oo(struk.punkty[i+1],E,funkcja,struk)<<" "<<funkcja_z_polem_do_oo(struk.punkty[i],E,funkcja,struk);
	  if(iloczyn < 0)
	    tymcz=1;
	}
      tylezer = tymcz;
      //      std::cerr<<"Jeszcze nie ma zer Airy'ego!\n";
      //      abort();
    }
  else
    {
      if(E >= y_pocz)
	{
	  double k = sqrt(2*masa_p(E)*(E-y_pocz));
	  tylezer = int( k*(x_kon - x_pocz)/M_PI );
	  if(tylezer % 2 == 0)
	    {
	      if(iloczyn < 0)
		{
		  tylezer++;
		}
	    }
	  else
	    {
	      if(iloczyn > 0)
		{
		  tylezer++;
		}
	    }
	}
      else
	{
	  if(iloczyn < 0)
	    {
	      tylezer++;
	    }
	}
    }
  //std::cerr<<"\nE = "<<E<<"\tiloczyn = "<<iloczyn<<"\t zer jest "<<tylezer;
  return tylezer;
}
/*****************************************************************************/
int warstwa::zera_ffal(double E, double A, double B) const
{
  int tylezer = 0;
  double wart_kon = funkcjafal(x_kon, E, A, B);
  double iloczyn = funkcjafal(x_pocz, E, A, B)*wart_kon;
  //std::cerr<<"\n wart na końcach: "<<funkcjafal(x_pocz, E, A, B)<<", "<<funkcjafal(x_kon, E, A, B);
  if(pole !=0)
    {
      double a13 = (pole > 0)?pow(2*masa_p(E)*pole,1./3):-pow(-2*masa_p(E)*pole,1./3); // a^{1/3} 
      double U = y_pocz - pole*x_pocz;
      double b_a23 = (U - E)/pole;
      double arg1, arg2, argl, argp, x1, x2, xlew, xpra;
      int nrza, nrprzed; // nrza do argp, nrprzed do argl
      arg1 = a13*(x_pocz + b_a23);
      arg2 = a13*(x_kon + b_a23);
      argl = std::min(arg1, arg2);
      argp = std::max(arg1, arg2);
      nrza=1;
      double z1 = -1.174; // oszacowanie pierwszego zera B1
      double dz = -2.098; // oszacowanie odstępu między perwszymi dwoma zerami
      //  nrprzed=1;
      //      nrprzed = floor((argl-z1)/dz + 1); // oszacowanie z dołu numeru miejsca zerowego
      nrprzed = floor((argp-z1)/dz + 1);
      nrprzed = (nrprzed >= 1)?nrprzed:1;
      int tymcz=0;
      double ntezero = boost::math::airy_bi_zero<double>(nrprzed);
      std::cerr<<"\nU = "<<U<<" a13 = "<<a13<<" b_a23 = "<<b_a23<<" argl = "<<argl<<" argp = "<<argp<<" ntezero = "<<ntezero<<" nrprzed = "<<nrprzed;
      double brak; // oszacowanie z dołu braku
      long licznik = 0;
      //      while(ntezero>=argl)
      while(ntezero>=argp)
	{
	  if(nrprzed>2)
	    {
          dz = ntezero - boost::math::airy_bi_zero<double>(nrprzed-1);
	      brak = (argp-ntezero)/dz;
	      if(brak > 2.) //jeśli jeszcze daleko
		{
		  nrprzed = nrprzed + floor(brak);
		}
	      else nrprzed++;
	    }
	  else
	    nrprzed++;
      ntezero = boost::math::airy_bi_zero<double>(nrprzed);
	  licznik++;
	  std::cerr<<"\nnrprzed = "<<nrprzed<<" ntezero = "<<ntezero;
	}
      //  std::cerr<<"\tnrprzed kon "<<nrprzed<<" po "<<licznik<<" dodawaniach\n";
      nrza=nrprzed;
      nrprzed--;
      //      while(boost::math::airy_bi_zero<double>(nrza)>=argp)
      while(boost::math::airy_bi_zero<double>(nrza)>=argl)
	{
	  nrza++;
      std::cerr<<"\nnrza = "<<nrza<<" ntezero = "<<boost::math::airy_bi_zero<double>(nrza);
	}
      std::cerr<<"\nnrprzed = "<<nrprzed<<" nrza = "<<nrza;
      /*
      std::cerr<<"\nnrprzed = "<<nrprzed<<" nrza = "<<nrza;
      x1 = b_a23 - boost::math::airy_bi_zero<double>(nrprzed+1)/a13; // polozenia skrajnych zer Ai w studni
      x2 = b_a23 - boost::math::airy_bi_zero<double>(nrza-1)/a13;
      xlew = std::min(x1, x2);
      xpra = std::max(x1, x2);
      std::cerr<<"\txlew="<<struktura::dlugosc_na_A(xlew)<<" xpra="<<struktura::dlugosc_na_A(xpra);
      tylko do testów tutaj  */

      if(nrza-nrprzed>=2)
	{
	  tymcz=nrza-nrprzed-2;
      x1 = -b_a23 + boost::math::airy_bi_zero<double>(nrprzed+1)/a13; // polozenia skrajnych zer Ai w studni
      x2 = -b_a23 + boost::math::airy_bi_zero<double>(nrza-1)/a13;
	  xlew = std::min(x1, x2);
	  xpra = std::max(x1, x2);
	  std::cerr<<"\n xlew="<<struktura::dlugosc_na_A(xlew)<<" xpra="<<struktura::dlugosc_na_A(xpra);
	  //      std::cerr<<"\n A "<<funkcja_z_polem_do_oo(struk.punkty[i],E,funkcja,struk)<<" "<<funkcja_z_polem_do_oo(xl,E,funkcja,struk);
	  if(funkcjafal(x_pocz, E, A, B)*funkcjafal(xlew, E, A, B) < 0)
	    //	  if(funkcja_z_polem_do_oo(struk.punkty[i],E,funkcja,struk)*funkcja_z_polem_do_oo(xl,E,funkcja,struk)<0)
	    tymcz++;
	  if(wart_kon*funkcjafal(xpra, E, A, B) < 0)
	    tymcz++;
	}
      else
	{
	  //      std::cerr<<"\n C "<<funkcja_z_polem_do_oo(struk.punkty[i+1],E,funkcja,struk)<<" "<<funkcja_z_polem_do_oo(struk.punkty[i],E,funkcja,struk);
	  if(funkcjafal(x_pocz, E, A, B)*wart_kon < 0)
	    tymcz=1;
	}
      tylezer = tymcz;
      //      std::cerr<<"Jeszcze nie ma zer Airy'ego!\n";
      //      abort();
    }
  else
    {
      if(E >= y_pocz)
	{
	  double k = sqrt(2*masa_p(E)*(E-y_pocz));
	  tylezer = int( k*(x_kon - x_pocz)/3.14159265359 ); // M_PI
	  if(tylezer % 2 == 0)
	    {
	      if(iloczyn < 0)
		{
		  tylezer++;
		}
	    }
	  else
	    {
	      if(iloczyn > 0)
		{
		  tylezer++;
		}
	    }
	}
      else
	{
	  if(iloczyn < 0)
	    {
	      tylezer++;
	    }
	}
    }
  if (mInfo) std::cerr<<"\nE = "<<E<<"\tiloczyn = "<<iloczyn<<"\t zer jest "<<tylezer; // LUKASZ
  return tylezer;
}
/*****************************************************************************/
double warstwa::funkcjafal(double x, double E, double A, double B) const
{
  return A*ffala(x, E) + B*ffalb(x, E);
}
/*****************************************************************************/
double warstwa::funkcjafal_prim(double x, double E, double A, double B) const
{
  return A*ffala_prim(x, E) + B*ffalb_prim(x, E);
}
/*****************************************************************************/
double warstwa::norma_kwadr(double E, double A, double B) const
{
  double wartosc;
  if(pole !=0)
    {
      wartosc = 1.; // chwilowo
      //      std::cerr<<"Jeszcze nie ma normay Airy'ego\n";
      //      abort();
      wartosc = airy_kwadr_pierwotna(x_kon, E, A, B) - airy_kwadr_pierwotna(x_pocz, E, A, B);
    }
  else
    {
      if(E >= y_pocz)
	{
	  wartosc = tryg_kwadr_pierwotna(x_kon, E, A, B) - tryg_kwadr_pierwotna(x_pocz, E, A, B);
	}
      else
	{
	  wartosc = exp_kwadr_pierwotna(x_kon, E, A, B) - exp_kwadr_pierwotna(x_pocz, E, A, B);
	}
    }
  return wartosc;
}
/*****************************************************************************/
double warstwa::Eodk(double k) const
{
  return k*k/(2*masa_r);
}
/*****************************************************************************/
void warstwa_skraj::przesun_igreki(double dE)
{
  y += dE;
  warstwa::przesun_igreki(dE);
}
/*****************************************************************************/
double warstwa_skraj::ffala(double x, double E) const
{
  double wartosc;
  if(lp == lewa)
    {
      wartosc = 0; // Po lewej nie ma funkcji a
    }
  else
    {
      if(E > y)
	{
	  std::cerr<<"Energia nad skrajną barierą!\nE = "<<E<<" y = "<<y<<"\n";
	  abort();
	}
      else
	{
	  //	  std::clog<<"Lewa, expb z x = "<<x<<" a x_warstwy = "<<(this->iks)<<"\n";
	  wartosc = expa(x, E);
	}
    }
  return wartosc;
}
/*****************************************************************************/
double warstwa_skraj::ffalb(double x, double E) const
{
  double wartosc;
  if(lp == prawa)
    {
      wartosc = 0; // Po prawej nie ma funkcji b
    }
  else
    {
      if(E > y)
	{
	  std::cerr<<"Energia nad skrajną barierą!\nE = "<<E<<" y = "<<y<<"\n";
	  abort();
	}
      else
	{
	  wartosc = expb(x, E);
	}
    }
  return wartosc;
}
/*****************************************************************************/
double warstwa_skraj::ffala_prim(double x, double E) const
{
  double wartosc;
  if(lp == lewa)
    {
      wartosc = 0; // Po lewej nie ma funkcji a
    }
  else
    {
      if(E > y)
	{
	  std::cerr<<"Energia nad skrajną barierą!\nE = "<<E<<" y = "<<y<<"\n";
	  abort();
	}
      else
	{
	  wartosc = expa_prim(x, E);
	}
    }
  return wartosc;
}
/*****************************************************************************/
double warstwa_skraj::ffalb_prim(double x, double E) const
{
  double wartosc;
  if(lp == prawa)
    {
      wartosc = 0; // Po lewej nie ma funkcji b
    }
  else
    {
      if(E > y)
	{
	  std::cerr<<"Energia nad skrajną barierą!\n";
	  abort();
	}
      else
	{
	  wartosc = expb_prim(x, E);
	}
    }
  return wartosc;
}
/*****************************************************************************/
int warstwa_skraj::zera_ffal(double, double, double) const
{
  return 0;
}
/*****************************************************************************/
double warstwa_skraj::norma_kwadr(double E, double C) const
{
  if(E > y)
    {
      std::cerr<<"Zla energia!\n";
      abort();
    }
  double kp = sqrt(2*masa_p*(y - E));
  return C*C/(2*kp);
}
/*****************************************************************************/
double warstwa_skraj::funkcjafal(double x, double E, double C) const
{
  double wynik;
  if(lp == lewa)
    {
      wynik = C*ffalb(x, E);
    }
  else
    {
      wynik = C*ffala(x, E);
    }
  return wynik;
}
/*****************************************************************************/
double warstwa_skraj::funkcjafal_prim(double x, double E, double C) const
{
  double wynik;
  if(lp == lewa)
    {
      wynik = C*ffalb_prim(x, E);
    }
  else
    {
      wynik = C*ffala_prim(x, E);
    }
  return wynik;
}
/*****************************************************************************/

/*****************************************************************************/
punkt::punkt()
{
  
}
/*****************************************************************************/
punkt::punkt(double e, double w)
{
  en=e;
  wart = w;
}
/*****************************************************************************/
punkt::punkt(const stan & st)
{
  en=st.poziom;
  wart = 0;
}
/*****************************************************************************/

/*****************************************************************************/
stan::stan()
{
  liczba_zer = -1;
}
/*****************************************************************************/
stan::stan(double E, A2D & V, int lz)
{
  poziom = E;
  int M = V.dim1();
  wspolczynniki.resize(M);
  for(int i = 0; i <= M - 1; i++)
    {
       wspolczynniki[i] = V[i][M - 1];
    }
  liczba_zer = lz;
  prawdopodobienstwa.reserve(M/2 +1);
}
/*****************************************************************************/
void stan::przesun_poziom(double dE)
{
  poziom += dE;
}
/*****************************************************************************/

/***************************************************************************** Stara wersja
struktura::struktura(const std::vector<warstwa> & tablica)
{
  gora = tablica[0].y_kon;
  dol = gora;
  double czydol;
  if( tablica[0].pole != 0 || tablica[tablica.size() - 1].pole != 0 || tablica[0].y_kon != tablica[tablica.size() - 1].y_pocz)
    {
      std::cerr<<"Zle energie skajnych warstw!\n";
      abort();
    }
  kawalki.push_back(tablica.front());
  for(int i = 1; i <= (int) tablica.size() - 2; i++)
    {
      kawalki.push_back(tablica[i]);
      czydol = (tablica[i].y_pocz > tablica[i].y_kon)?tablica[i].y_kon:tablica[i].y_pocz;
      if(czydol < dol)
	{
	  dol = czydol;
	}
      if(tablica[i].pole == 0)
	{
	  progi.push_back(tablica[i].y_pocz);
	}
    }
  kawalki.push_back(tablica.back());
  if(dol >= gora)
    {
      std::cerr<<"Brak jakiejkolwiek studni!\n";
      abort();
    }
  std::vector<double>::iterator it = progi.begin();
  while(it != progi.end())
    {
      if( *it == dol)
	{
	  progi.erase(it);
	}
      it++;
    }
  dokl = 1e-6;
}
*****************************************************************************/
struktura::struktura(const std::vector<warstwa*> & tablica, rodzaj co)
{
  lewa = *((const warstwa_skraj *)  tablica[0]);
  if(lewa.lp == warstwa_skraj::prawa)
    {
      std::cerr<<"Pierwsza warstwa nie jest lewa!\n";
      abort();
    }
  gora = lewa.y; // Zero to warstwa skrajna
  dol = gora;
  prawa = *((const warstwa_skraj *) tablica.back());
  if(prawa.lp == warstwa_skraj::lewa)
    {
      std::cerr<<"Ostatnia warstwa nie jest prawa!\n";
      abort();
    }

  double czydol;
  if( lewa.y != prawa.y)
    {
      std::cerr<<"Zle energie skajnych warstw!\n";
      abort();
    }
  int i;
  for(i = 1; i <= (int) tablica.size() - 2; i++)
    {
      //if(tablica[i - 1]->x_kon != tablica[i]->x_pocz) // LUKASZ
    //writelog(LOG_DETAIL, "Layers ends: %1%, %2%", tablica[i - 1]->x_kon, tablica[i]->x_pocz); // LUKASZ
    if (std::abs(tablica[i - 1]->x_kon - tablica[i]->x_pocz) > 1e-5) // LUKASZ
	{
      //std::cerr<<"Rozne krance warstw "<<(i - 1)<<" i "<<i<<" w "<<co<<": "<<(tablica[i - 1]->x_kon)<<" =/= "<<(tablica[i]->x_pocz)<<"\n";
      writelog(LOG_DETAIL, "Rozne krance warstw %1% i %2%", (i-1), i); // LUKASZ
      abort();
	}
      kawalki.push_back(*tablica[i]);
      tablica[i-1]->nast = tablica[i]; // ustawianie wskaznika na sasiadke
      czydol = (tablica[i]->y_pocz > tablica[i]->y_kon)?tablica[i]->y_kon:tablica[i]->y_pocz;
      if(czydol < dol)
	{
	  dol = czydol;
	}
      if(tablica[i]->pole == 0)
	{
	  progi.push_back(tablica[i]->y_pocz);
	}
    }
  //  std::clog<<"i = "<<i<<"\tx_pocz("<<(i-1)"<<(tablica[i - 1]->x_pocz)<<"\n";

  //if(tablica[i - 1]->x_kon != tablica[i]->x_pocz) // LUKASZ
  //writelog(LOG_DETAIL, "Layers ends: %1%, %2%", tablica[i - 1]->x_kon, tablica[i]->x_pocz); // LUKASZ
  if (std::abs(tablica[i - 1]->x_kon - tablica[i]->x_pocz) > 1e-5) // LUKASZ
    {
      std::cerr<<"Rozne krance warstw "<<(i - 1)<<" i "<<i<<"\n";
      writelog(LOG_DETAIL, "Rozne krance warstw %1% i %2%", (i-1), i); // LUKASZ
      writelog(LOG_DETAIL, "Rozne krance warstw"); // LUKASZ
      abort();
    }
  if(dol >= gora)
    {
      std::cerr<<"Brak jakiejkolwiek studni!\n";
      abort();
    }
  std::vector<double>::iterator it = progi.begin();
  while(it != progi.end())
    {
      //      std::clog<<"prog = "<<(*it)<<"\n";
      if( *it == dol)
	{
	  it = progi.erase(it);
	}
      else
	{
	  it++;
	}
    }
  typ = co;
  dokl = 1e-6;
  writelog(LOG_DETAIL, "Computing levels"); // LUKASZ
  szukanie_poziomow(gora);
  writelog(LOG_DETAIL, "Normalisation"); // LUKASZ
  normowanie();
  writelog(LOG_DETAIL, "Structure built"); // LUKASZ
  // profil(0., 1e-5);
}
/*****************************************************************************/
/*struktura::struktura(std::ifstream & plik, rodzaj co)
{
  std::string wiersz, bezkoment;
  boost::regex wykoment("#.*");
  std::string nic("");
  boost::regex pust("\\s+");
  boost::regex pole("pole");
  std::vector<double> parametry;
  double liczba;
  double x_pocz = 0, x_kon, y_pocz, y_kon, npar1, npar2, masa_p, masa_r;
  bool bylalewa = false, bylawew = false, bylaprawa = false;
  std::vector<warstwa*> tablica;
  warstwa * wskazwar;
  std::getline(plik, wiersz);
  bool jestpole = regex_search(wiersz, pole);
  std::clog<<"\njestpole = "<<jestpole<<"\n";
  int max_par = (jestpole)?7:6;
  int min_par = (jestpole)?5:4; // maksymalna i minimalna liczba parametrów w wierszu
  while (!plik.eof()) 
    {
      bezkoment = regex_replace(wiersz, wykoment, nic);
      boost::sregex_token_iterator it(bezkoment.begin(), bezkoment.end(), pust, -1);
      boost::sregex_token_iterator kon;
      if(it != kon) // niepusta zawartość
	{
	  parametry.clear();
	  while (it != kon) 
	    {
	      std::clog << *it << " * ";
	      try {
		liczba = boost::lexical_cast<double>(*it);
		it++;
	      } catch(boost::bad_lexical_cast&) {
		std::cerr<<"\n napis "<< *it<<" nie jest liczbą\n";
		abort();
	      }
	      parametry.push_back(liczba);
	      std::clog<<"\n";
	    }
	  if(bylalewa && ( ((int)parametry.size() < min_par && parametry.size() != 1) || (int)parametry.size() > max_par) )
	    {
	      if(jestpole)
		std::cerr<<"\nwarstwa wymaga 1 lub od 5 do 7 parametrów, a są "<<parametry.size()<<"\n";
	      else
		std::cerr<<"\nwarstwa wymaga 1 lub od 4 do 6 parametrów, a są "<<parametry.size()<<"\n";
	      abort();
	    }
	  if(bylalewa)
	    {
	      if(parametry.size() == 1) // prawa
		{
		  bylaprawa = true;
		  wskazwar = new warstwa_skraj(warstwa_skraj::prawa, parametry[0], parametry[0], x_pocz, 0.);
		  std::clog<<"\nrobi się prawa: masa = "<<parametry[0]<<" x_pocz = "<<x_pocz<<"\n";
		  tablica.push_back(wskazwar);
		  break;
		}
	      else // wewnętrzna
		{	     
		  x_kon = x_pocz + parametry[0];
		  y_pocz = -parametry[1];
		  if (jestpole)
		    {
		      y_kon = -parametry[2];;
		      masa_p = parametry[3];
		      masa_r = parametry[4];
		    }
		  else
		    {
		      y_kon = y_pocz;
		      masa_p = parametry[2];
		      masa_r = parametry[3];
		    }
		  npar1 = 0.;
		  npar2 = 0.;
		  if( (parametry[0] <= 0.) || (masa_p <= 0.) || (masa_r <= 0.) )
		    {
		      std::cerr<<"\nAaaaaa!\n";
		      abort();
		    }
		  bylalewa = true;
		  if((int)parametry.size() == min_par + 1)
		    {
		      npar1 = parametry[min_par];
		    }
		  if((int)parametry.size() == min_par + 2)
		    {
		      npar1 = parametry[min_par];
		      npar2 = parametry[min_par + 1];
		    }
		  wskazwar = new warstwa(masa_p, masa_r, x_pocz, y_pocz, x_kon, y_kon, npar1, npar2);
		  std::clog<<"masa_p = "<<masa_p<<", masa_r = "<<masa_r<<", x_pocz = "<<x_pocz<<", y_pocz = "<<y_pocz<<", x_kon = "<<x_kon<<", y_kon = "<<y_kon<<", npar1 = "<<npar1<<", npar2 = "<<npar2<<"\n";;
		  tablica.push_back(wskazwar);
		  x_pocz = x_kon;
		}
	    }
	  else
	    {
	      bylalewa = true;
	      wskazwar = new warstwa_skraj(warstwa_skraj::lewa, parametry[0], parametry[0], x_pocz, 0.);
	      tablica.push_back(wskazwar);
	      std::clog<<"\nrobi się lewa: masa = "<<parametry[0]<<" x_pocz = "<<x_pocz<<"\n";
	    }
	}
      if(bylalewa && bylawew && bylaprawa)
	{
	  std::clog<<"\nWsystko było\n";
	}
      std::getline(plik, wiersz);
    }

  // poniżej zawartość konstruktora od tablicy

  lewa = *((const warstwa_skraj *)  tablica[0]);
  if(lewa.lp == warstwa_skraj::prawa)
    {
      std::cerr<<"Pierwsza warstwa nie jest lewa!\n";
      abort();
    }
  gora = lewa.y; // Zero to warstwa skrajna
  dol = gora;
  prawa = *((const warstwa_skraj *) tablica.back());
  if(prawa.lp == warstwa_skraj::lewa)
    {
      std::cerr<<"Ostatnia warstwa nie jest prawa!\n";
      abort();
    }

  double czydol;
  if( lewa.y != prawa.y)
    {
      std::cerr<<"Zle energie skajnych warstw!\n";
      abort();
    }
  int i;
  for(i = 1; i <= (int) tablica.size() - 2; i++)
    {
      if(tablica[i - 1]->x_kon != tablica[i]->x_pocz)
	{
	  std::cerr<<"Rozne krance warstw "<<(i - 1)<<" i "<<i<<" w "<<co<<": "<<(tablica[i - 1]->x_kon)<<" =/= "<<(tablica[i]->x_pocz)<<"\n";
	  abort();
	}
      kawalki.push_back(*tablica[i]);
      tablica[i-1]->nast = tablica[i]; // ustawianie wskaźnika na sąsiadkę
      czydol = (tablica[i]->y_pocz > tablica[i]->y_kon)?tablica[i]->y_kon:tablica[i]->y_pocz;
      if(czydol < dol)
	{
	  dol = czydol;
	}
      if(tablica[i]->pole == 0)
	{
	  progi.push_back(tablica[i]->y_pocz);
	}
    }
  //  std::clog<<"i = "<<i<<"\tx_pocz("<<(i-1)"<<(tablica[i - 1]->x_pocz)<<"\n";
  if(tablica[i - 1]->x_kon != tablica[i]->x_pocz)
    {
      std::cerr<<"Rozne krance warstw "<<(i - 1)<<" i "<<i<<"\n";
      abort();
    }
  if(dol >= gora)
    {
      std::cerr<<"Brak jakiejkolwiek studni!\n";
      abort();
    }
  std::vector<double>::iterator it = progi.begin();
  while(it != progi.end())
    {
      //      std::clog<<"prog = "<<(*it)<<"\n";
      if( *it == dol)
	{
	  it = progi.erase(it);
	}
      else
	{
	  it++;
	}
    }
  typ = co;
  dokl = 1e-6;
  szukanie_poziomow(gora);
  normowanie();
}*/
/*****************************************************************************/
void struktura::przesun_energie(double dE)
{
  gora += dE;
  dol += dE;
  lewa.przesun_igreki(dE);
  prawa.przesun_igreki(dE);
  for(int i = 0; i <= (int) kawalki.size() - 1; i++)
    {
      kawalki[i].przesun_igreki(dE);
    }
  for(int i = 0; i <= (int) progi.size() - 1; i++)
    {
      progi[i] += dE;
    }
  for(int i = 0; i <= (int) rozwiazania.size() - 1; i++)
    {
      rozwiazania[i].przesun_poziom(dE);
    }
}
/*****************************************************************************/
double struktura::dlugosc_z_A(const double dlugA)
{
  return dlugA/przelm;
}
/*****************************************************************************/
double struktura::dlugosc_na_A(const double dlug)
{
  return dlug*przelm;
}
/*****************************************************************************/
double struktura::koncentracja_na_cm_3(const double k_w_wew)
{
  return k_w_wew/(przelm*przelm*przelm)*1e24;
}
/*****************************************************************************/
double struktura::czyosobliwa(double E)
{
  int N = kawalki.size() + 2; //liczba warstw
  // Bylo bez '+ 2'
  if(N < 3)
    {
      std::cerr<<"Za mało warstw, bo "<<N<<"\n";
      abort();
    }
  int M = 2*N - 2; // liczba rownan
  A2D macierz(M, M, 0.0);
  zrobmacierz(E, macierz);
  //  std::clog<<E<<"\n"<<macierz<<"\n";
  A1D S(macierz.dim1());
  JAMA::SVD<double> rozklad(macierz);
  rozklad.getSingularValues(S);

  A2D V(M, M);
  A2D U(M, M);
  rozklad.getV(V);
  rozklad.getU(U);
  A2D UV = matmult(U, V);
  //  std::clog<<E<<"\nV = "<<V<<"\n";
  JAMA::LU<double> rozkladUV(UV);
  double detUV = rozkladUV.det();
  //  return((S[S.dim() - 1]));

  //  return((S[S.dim() - 1])/S[S.dim() - 2]); // Dzielone, żeby nie było zer podwójnych
  double dzielnik = 1; // Zeby wyeleminowac zera na plaskich kawalkach
  for(int i = 0; i<= (int) progi.size() - 1; i++)
    {
      dzielnik *= (E - progi[i]);
    }
  return(detUV*S[S.dim() - 1])/dzielnik;
  //  return(S[S.dim() - 1]);
}
/***************************************************************************** Stare
void struktura::zrobmacierz(double E, A2D & macierz)
{
  int N = kawalki.size(); // liczba warstw
  double x = kawalki[1].x_pocz;
  macierz[0][0] = kawalki[0].ffalb(x, E);
  macierz[0][1] = -kawalki[1].ffala(x, E);
  macierz[0][2] = -kawalki[1].ffalb(x, E);
  macierz[1][0] = kawalki[0].ffalb_prim(x, E)/kawalki[0].masa;
  macierz[1][1] = -kawalki[1].ffala_prim(x, E)/kawalki[1].masa;
  macierz[1][2] = -kawalki[1].ffalb_prim(x, E)/kawalki[1].masa;
  int n = 1;
  for(n = 1; n <= N - 3; n++)
    {
      x = kawalki[n + 1].x_pocz;
      macierz[2*n][2*n - 1] = kawalki[n].ffala(x, E);
      macierz[2*n][2*n] = kawalki[n].ffalb(x, E);
      macierz[2*n][2*n + 1] = -kawalki[n+1].ffala(x, E);
      macierz[2*n][2*n + 2] = -kawalki[n+1].ffalb(x, E);

      macierz[2*n + 1][2*n - 1] = kawalki[n].ffala_prim(x, E)/kawalki[n].masa;
      macierz[2*n + 1][2*n] = kawalki[n].ffalb_prim(x, E)/kawalki[n].masa;
      macierz[2*n + 1][2*n + 1] = -kawalki[n+1].ffala_prim(x, E)/kawalki[n + 1].masa;
      macierz[2*n + 1][2*n + 2] = -kawalki[n+1].ffalb_prim(x, E)/kawalki[n + 1].masa;      
    }
  x = kawalki[N - 1].x_pocz;
  // std::clog<<"ostatni x = "<<(x*warstwa::przelm)<<"\n";
  macierz[2*n][2*n - 1] = kawalki[n].ffala(x, E);
  macierz[2*n][2*n] = kawalki[n].ffalb(x, E);
  macierz[2*n][2*n + 1] = -kawalki[n+1].ffala(x, E);
   
  macierz[2*n + 1][2*n - 1] = kawalki[n].ffala_prim(x, E)/kawalki[n].masa;
  macierz[2*n + 1][2*n] = kawalki[n].ffalb_prim(x, E)/kawalki[n].masa;
  macierz[2*n + 1][2*n + 1] = -kawalki[n+1].ffala_prim(x, E)/kawalki[n + 1].masa;
 }
*****************************************************************************/
void struktura::zrobmacierz(double E, A2D & macierz)
{
  int N = kawalki.size() + 2; // liczba warstw
  double x = lewa.iks;
  macierz[0][0] = lewa.ffalb(x, E);
  macierz[0][1] = -kawalki[0].ffala(x, E);
  macierz[0][2] = -kawalki[0].ffalb(x, E);
  macierz[1][0] = lewa.ffalb_prim(x, E)/lewa.masa_p;
  macierz[1][1] = -kawalki[0].ffala_prim(x, E)/kawalki[0].masa_p(E);
  macierz[1][2] = -kawalki[0].ffalb_prim(x, E)/kawalki[0].masa_p(E);
  int n = 1;
  for(n = 1; n <= N - 3; n++)
    {
      x = kawalki[n].x_pocz;
      macierz[2*n][2*n - 1] = kawalki[n - 1].ffala(x, E);
      macierz[2*n][2*n] = kawalki[n - 1].ffalb(x, E);
      macierz[2*n][2*n + 1] = -kawalki[n].ffala(x, E);
      macierz[2*n][2*n + 2] = -kawalki[n].ffalb(x, E);

      macierz[2*n + 1][2*n - 1] = kawalki[n - 1].ffala_prim(x, E)/kawalki[n - 1].masa_p(E);
      macierz[2*n + 1][2*n] = kawalki[n - 1].ffalb_prim(x, E)/kawalki[n - 1].masa_p(E);
      macierz[2*n + 1][2*n + 1] = -kawalki[n].ffala_prim(x, E)/kawalki[n].masa_p(E);
      macierz[2*n + 1][2*n + 2] = -kawalki[n].ffalb_prim(x, E)/kawalki[n].masa_p(E);      
    }
  x = prawa.iks;
  // std::clog<<"ostatni x = "<<(x*warstwa::przelm)<<"\n";
  macierz[2*n][2*n - 1] = kawalki[n - 1].ffala(x, E);
  macierz[2*n][2*n] = kawalki[n - 1].ffalb(x, E);
  macierz[2*n][2*n + 1] = -prawa.ffala(x, E);
   
  macierz[2*n + 1][2*n - 1] = kawalki[n - 1].ffala_prim(x, E)/kawalki[n - 1].masa_p(E);
  macierz[2*n + 1][2*n] = kawalki[n - 1].ffalb_prim(x, E)/kawalki[n - 1].masa_p(E);
  macierz[2*n + 1][2*n + 1] = -prawa.ffala_prim(x, E)/prawa.masa_p;
 }
/*****************************************************************************/
std::vector<std::vector<double> > struktura::rysowanie_funkcji(double E, double x0A, double xkA, double krokA)
{
  double x0 = x0A/przelm;
  double xk = xkA/przelm;
  const double krok = krokA/przelm;
  int N = kawalki.size() + 2; //liczba warstw
  int M = 2*N - 2; // liczba rownan
  A2D macierz(M, M, 0.0);
  zrobmacierz(E, macierz);
  A2D V(M, M);
  JAMA::SVD<double> rozklad(macierz);
  rozklad.getV(V);
  //  std::clog<<"To jest V:\n"<<V;
  double x = x0;
  int wsk = 0;
  std::vector<std::vector<double> > funkcja;
  funkcja.resize(2);
  funkcja[0].reserve(int((xk - x0)/krok));
  funkcja[1].reserve(int((xk - x0)/krok));
  double A, B;
  if( x < lewa.iks)
    {
      wsk = -1;
    }
  else
    {
      while( (wsk <= (int) kawalki.size() - 1) && (x > kawalki[wsk].x_kon) ) // Szukanie pierwszej kawalki
	{
	  wsk++;
	}
    }
  while(x <= xk)
    {
      funkcja[0].push_back(x);
      if(wsk == -1.)
	{
	  B = V[0][V.dim2() - 1];
	  funkcja[1].push_back(lewa.funkcjafal(x, E, B));
	  if(x > lewa.iks)
	    {
	      wsk++;
	    }
	}
      else
	if(wsk <= (int) kawalki.size() - 1)
	  {
	    A = V[2*wsk + 1][V.dim2() - 1];
	    B = V[2*wsk + 2][V.dim2() - 1];
	    funkcja[1].push_back(kawalki[wsk].funkcjafal(x, E, A, B));
	  }
	else
	  {
	    A = V[2*wsk + 1][V.dim2() - 1];
	    funkcja[1].push_back(prawa.funkcjafal(x, E, A));
	  }
      x += krok;
      if( (wsk >= 0) && (wsk <= (int) kawalki.size() - 1) && (x > kawalki[wsk].x_kon) )
	{
	  wsk++;
	}
    }
  return funkcja;
}
/*****************************************************************************
int struktura::ilezer_ffal(double E)
{
  int N = kawalki.size() + 2; //liczba warstw
  // Bylo bez '+ 2'
  int M = 2*N - 2; // liczba rownan
  A2D macierz(M, M, 0.0);
  zrobmacierz(E, macierz);
  A2D V(M, M);
  JAMA::SVD<double> rozklad(macierz);
  rozklad.getV(V);
  int sumazer = 0;
  double A, B, As, Bs;
  //  bool juz = false, jeszcze = true; // czy już lub jeszcze warto sprawdzać (nie będzie zer w warstwie, w której poziom jest pod przerwą i tak samo jest we wszystkich od lub do skrajnej)
  int pierwsza = -1;
  do
    {
      pierwsza++;
    }while(pierwsza <= N-3 && (kawalki[pierwsza].y_pocz > E && kawalki[pierwsza].y_kon > E) );
  int ostatnia = N-2;
  do
    {
      ostatnia--;
    }while(ostatnia >= 0 && (kawalki[ostatnia].y_pocz > E && kawalki[ostatnia].y_kon > E) );
  std::clog<<"\npierwsza sprawdzana = "<<pierwsza<<" ostatnia = "<<ostatnia;

  double sasiad; // wartosc ffal na lewym brzegu sasiada z prawej (ta, ktora powinna byc wspolna do obu)
  for(int j = pierwsza; j <= ostatnia - 1; j++)
    {
      A = V[2*j+1][V.dim2() - 1];
      B = V[2*j+2][V.dim2() - 1];
      As = V[2*(j+1)+1][V.dim2() - 1];
      Bs = V[2*(j+1)+2][V.dim2() - 1];
      sasiad = kawalki[j+1].funkcjafal(kawalki[j+1].x_pocz, E, As, Bs); 
      sumazer += kawalki[j].zera_ffal(E, A, B, sasiad);
    }
  A = V[2*ostatnia+1][V.dim2() - 1];
  B = V[2*ostatnia+2][V.dim2() - 1];
  sumazer += kawalki[ostatnia].zera_ffal(E, A, B); // W ostatniej warstwie nie może byc zera na laczeniu, wiec nie ma problemu
  return sumazer;
}
*****************************************************************************/
int struktura::ilezer_ffal(double E, A2D & V)
{
  int N = kawalki.size() + 2; //liczba warstw
  // Bylo bez '+ 2'
  int M = 2*N - 2; // liczba rownan
  A2D macierz(M, M, 0.0);
  zrobmacierz(E, macierz);
  JAMA::SVD<double> rozklad(macierz);
  rozklad.getV(V);
  int sumazer = 0;
  double A, B, Al, Bl, Ap, Bp;
  /*
  for(int i = 1; i <= N-2; i++)
    {
      A = V[2*i-1][V.dim2() - 1];
      B = V[2*i][V.dim2() - 1];
      sumazer += kawalki[i - 1].zera_ffal(E, A, B);
    }
  */ // tak bylo bez szukania pierwszej i ostatniej
  int pierwsza = -1;
  do
    {
      pierwsza++;
    }while(pierwsza <= N-3 && (kawalki[pierwsza].y_pocz > E && kawalki[pierwsza].y_kon > E) );
  int ostatnia = N-2;
  do
    {
      ostatnia--;
    }while(ostatnia >= 0 && (kawalki[ostatnia].y_pocz > E && kawalki[ostatnia].y_kon > E) );
  //std::clog<<"\npierwsza sprawdzana = "<<pierwsza<<" ostatnia = "<<ostatnia;
  double sasiadl, sasiadp; // wartosc ffal na lewym brzegu sasiada z prawej (ta, która powinna byc wspolna do obu)
  if(ostatnia == pierwsza) // tylko jedna podejrzana warstwa, nie trzeba sie sasiadami przejmowac
    {
      A = V[2*pierwsza+1][V.dim2() - 1];
      B = V[2*pierwsza+2][V.dim2() - 1];
      sumazer += kawalki[pierwsza].zera_ffal(E, A, B);
    }
  else
    {
      int j = pierwsza;
      A = V[2*j+1][V.dim2() - 1];
      B = V[2*j+2][V.dim2() - 1];
      Ap = V[2*(j+1)+1][V.dim2() - 1];
      Bp = V[2*(j+1)+2][V.dim2() - 1];
      sasiadp = kawalki[j+1].funkcjafal(kawalki[j+1].x_pocz, E, Ap, Bp);
      sasiadl = kawalki[j].funkcjafal(kawalki[j].x_pocz, E, A, B); // po lewej nie ma problemu, więc mozna podstawic wartosc z wlasnej warstwy
      sumazer += kawalki[j].zera_ffal(E, A, B, sasiadl, sasiadp);
      for(int j = pierwsza + 1; j <= ostatnia - 1; j++)
	{
	  Al = V[2*(j-1)+1][V.dim2() - 1];
	  Bl = V[2*(j-1)+2][V.dim2() - 1];
	  A = V[2*j+1][V.dim2() - 1];
	  B = V[2*j+2][V.dim2() - 1];
	  Ap = V[2*(j+1)+1][V.dim2() - 1];
	  Bp = V[2*(j+1)+2][V.dim2() - 1];
	  sasiadl = kawalki[j-1].funkcjafal(kawalki[j-1].x_kon, E, Al, Bl); 
	  sasiadp = kawalki[j+1].funkcjafal(kawalki[j+1].x_pocz, E, Ap, Bp); 
	  sumazer += kawalki[j].zera_ffal(E, A, B, sasiadl, sasiadp);
	}
      j = ostatnia;
      A = V[2*j+1][V.dim2() - 1];
      B = V[2*j+2][V.dim2() - 1];
      Al = V[2*(j-1)+1][V.dim2() - 1];
      Bl = V[2*(j-1)+2][V.dim2() - 1];
      sasiadp = kawalki[j].funkcjafal(kawalki[j].x_kon, E, A, B); // po prawej nie ma problemu, więc można podstawić wartość z własnej warstwy
      sasiadl = kawalki[j-1].funkcjafal(kawalki[j-1].x_kon, E, Al, Bl); 
      sumazer += kawalki[j].zera_ffal(E, A, B, sasiadl, sasiadp); // W ostatniej warswie nie może być zera na łączeniu, więc nie ma problemu
    }
  return sumazer;
  //  return 0; //do testow tylko!
}
/*****************************************************************************/
std::vector<double> struktura::zageszczanie(punkt p0, punkt pk) // Zagęszcza aż znajdzie inny znak, zakłada, że początkowe znaki są takie same
{
  std::list<punkt> lista;
  std::vector<double> wynik;
  lista.push_front(p0);
  lista.push_back(pk);
  double E;
  double znak = (p0.wart > 0)?1.:-1.;
  if(znak * pk.wart <=0 )
    {
      std::cerr<<"W zageszczaniu znaki sie roznia!\n";
      abort();
    }
  std::list<punkt>::iterator iter, iterl, iterp;
  iter=lista.begin();
  /*
  if(minipasma)
    {
      E=p0.en + (p1.en - p0.en)/256;
    }
  else
    {
      E=p0.en + (p1.en - p0.en)/2;
    }
  iter=lista.insert(++iter,punkt(E,czyosobliwa(E)));
  */
  while(wynik.empty())
    {
      iterp = lista.end();
      iterp--;
      while(iterp != lista.begin() )
	{
	  iterl = iterp;
	  iterl--;
	  E=(iterp->en + iterl->en)/2;
	  if (mInfo) std::clog<<"El = "<<iterl->en<<" Eit = "<<E<<" Ep = "<<iterp->en<<"\n"; // LUKASZ
	  iter = lista.insert(iterp, punkt(E,czyosobliwa(E)));
	  if(znak * iter->wart < 0)
	    {
	      wynik.push_back(iterl->en);
	      wynik.push_back(iter->en);
	      wynik.push_back(iterp->en);
	      break;
	    }
	  iterp = iterl;
	}
    }
  return wynik;
}
/*****************************************************************************/
void struktura::profil(double Ek, double rozdz) 
{
  double E0 = dol;
  if(Ek <= E0)
    {
      std::cerr<<"Zły zakres energii!\n";
      abort();
    }
  for( double E = E0; E <= Ek; E += rozdz)
    {
      std::cout<<E<<"\t"<<czyosobliwa(E)<<"\n";
    }
  std::cout<<std::flush;
}
/*****************************************************************************/
void struktura::szukanie_poziomow(double Ek, double rozdz) // Trzeba dorobic obsluge niezbiegania sie metody siecznych
{
  double E0 = dol;
  if(Ek <= E0)
    {
      std::cerr<<"Zły zakres energii!\n";
      abort();
    }
  int M = 2*(kawalki.size() + 2) - 2;
  // Bylo 'int M = 2*kawalki.size() - 2;'
  double wartakt;
  double (struktura::*fun)(double) = & struktura::czyosobliwa;
  std::vector<double> trojka;
  if (mInfo) std::clog<<"W szukaniu\n"; // LUKASZ
  double wartpop = czyosobliwa(Ek);
  if (mInfo) std::clog<<"Pierwsza wartosc = "<<wartpop<<"\n"; // LUKASZ
  A2D V(M, M);
  int ilepoziomow;
  int liczbazer;
  int ostatnie_dobre , nast_dobre;
  double E;
  if(wartpop == 0)
    {
      Ek -= rozdz;
      wartpop = czyosobliwa(Ek);
    }
  E = Ek - rozdz;
  wartakt = czyosobliwa(E);
  while( (E > E0) && (wartpop*wartakt > 0) )
    {
      wartpop = wartakt;
      E -= rozdz;
      //      std::clog<<"Ek = "<<E<<"\n";
      wartakt = czyosobliwa(E);
    }
  if (mInfo) std::clog<<"Sieczne z krancami "<<E<<" i "<<(E + rozdz)<<"\n"; // LUKASZ
  double Eost = sieczne(fun, E, E + rozdz); // koniec szukania najwyższego stanu
  V = A2D(M, M);
  ilepoziomow = ilezer_ffal(Eost, V) + 1;
  ostatnie_dobre = ilepoziomow - 1;
  rozwiazania.resize(ilepoziomow);
  stan nowy(Eost, V, ostatnie_dobre);
  rozwiazania[ilepoziomow - 1] = nowy;
  if (mInfo) std::clog<<"Eost = "<<Eost<<" ilepoziomow = "<<ilepoziomow<<"\n"; // LUKASZ
  punkt ost, ostdob, pierw;
  nast_dobre = -1;
  while(ostatnie_dobre >= 1)
    {
      ostdob = punkt(rozwiazania[ostatnie_dobre]);
      E = ostdob.en - rozdz;
      ost = punkt(E, czyosobliwa(E));
      E = (nast_dobre >= 0)?(rozwiazania[nast_dobre].poziom + rozdz):(E0 + rozdz);
      pierw = punkt(E, czyosobliwa(E));
      //     std::clog<<"Eost = "<<ost.en<<" wartost = "<<ost.wart<<"\n";
      //      std::clog<<"Epierw = "<<pierw.en<<" wartpierw = "<<pierw.wart<<"\n";
      if(ost.wart*pierw.wart > 0)
	{
	  if (mInfo) std::clog<<"Zagęszczanie z pierw = "<<pierw.en<<" i ost = "<<ost.en<<"\n"; // LUKASZ
	  trojka = zageszczanie(pierw, ost);
	  for(int i = 1; i >= 0; i--)
	    {
	      E = sieczne(fun, trojka[i], trojka[i+1]);
	      liczbazer = ilezer_ffal(E, V);
	      if (mInfo) std::clog<<"E = "<<E<<"\tzer "<<liczbazer<<"\n"; // LUKASZ
	      nowy = stan(E, V, liczbazer);
	      if(liczbazer > ostatnie_dobre)
		{
		  std::cerr<<"Za dużo zer!\n";
		  abort();
		}
	      rozwiazania[liczbazer] = nowy;
	    }
	}
      else
	{
	  E = sieczne(fun, pierw.en, ost.en);
	  liczbazer = ilezer_ffal(E, V);
	  if (mInfo) std::clog<<"W else E = "<<E<<"\tzer "<<liczbazer<<"\n"; // LUKASZ
	  nowy = stan(E, V, liczbazer);
	  if(liczbazer > ostatnie_dobre)
	    {
	      std::cerr<<"Za dużo zer!\n";
	      abort();
	    }
	  rozwiazania[liczbazer] = nowy;
	}
      if (mInfo) std::clog<<"ostatnie_dobre = "<<ostatnie_dobre<<"\n"; // LUKASZ
      while(ostatnie_dobre >= 1 && rozwiazania[ostatnie_dobre - 1].liczba_zer >= 0 )
	{
	  ostatnie_dobre--;
	}
      nast_dobre = ostatnie_dobre - 1;
      while(nast_dobre >= 0 && rozwiazania[nast_dobre].liczba_zer < 0 )
	{
	  nast_dobre--;
	}
      if (mInfo) std::clog<<"ostatnie_dobre = "<<ostatnie_dobre<<"nastepne_dobre = "<<nast_dobre<<"\n"; // LUKASZ
    }
  if (mInfo) std::clog<<"Liczba rozwiazan = "<<rozwiazania.size()<<"\n"; // LUKASZ
}
/*****************************************************************************/
double struktura::sieczne(double (struktura::*f)(double), double pocz, double kon)
{
  std::clog.precision(12);
  //const double eps = 1e-14; // limit zmian x
  double dokl = 1e-7;
  double xp = kon;
  double xl = pocz;
  /*
  if((this->*f)(pocz)*(this->*f)(kon) > 0)
    {
      std::cerr<<"Złe krańce przedziału!\n";
      abort();
    }
  */
  double x, fc, fp, fl, xlp, xpp; // -p -- poprzednie krance 
  fl = (this->*f)(xl);
  fp = (this->*f)(xp);
  xlp = (xl + xp)/2;
  xpp = xlp; // żeby na pewno było na początku różne od prawdzwych końców
  do
    {
      x = xp - fp*(xp - xl)/(fp - fl);
      fc = (this->*f)(x);
      if(fc == 0)
	{
	  break;
	}
      if(fc*fl < 0) // c bedzie nowym prawym koncem
	{
	  //std::clog<<"xlp - xl = "<<(xlp - xl)<<"\n";
	  if(xlp == xl) // trzeba poprawic ten kraniec (metoda Illinois)
	    {
	      if (mInfo) std::clog<<"Lewy Illinois\n"; // LUKASZ
	      fl = fl/2;
	    }
	  xpp = xp;
	  xp = x;
	  xlp = xl;
	  fp = fc;
	}
      else // c bedzie nowym lewym koncem
	{
	  //  std::clog<<"xpp - xp = "<<(xpp - xp)<<"\n";
	  if(xpp == xp) // trzeba poprawic ten kraniec (metoda Illinois)
	    {
	      if (mInfo) std::clog<<"Prawy Illinois\n"; // LUKASZ
	      fp = fp/2;
	    }
	  xlp = xl;
	  xl = x;
	  xpp = xp;
	  fl = fc;
	  //	  fp = (this->*f)(kon);
	}
      //      std::clog<<"x = "<<x<<"\tf(x) = "<<fc<<"\txl = "<<xl<<" xp = "<<xp<<"\n";//<<"\txp - x = "<<(xp - x)<<"\n";
    }
  while(xp - xl >= dokl);
  return x;
}
/*****************************************************************************/
double struktura::norma_stanu(stan & st) // liczy norme i wypelnia prawdopodobienstwa
{
  double porcja = lewa.norma_kwadr(st.poziom, st.wspolczynniki.front());
  st.prawdopodobienstwa.push_back(porcja);
  double norma2 = porcja;
  for(int i = 0; i <= (int) kawalki.size() - 1; i++)
    {
      porcja = kawalki[i].norma_kwadr(st.poziom, st.wspolczynniki[2*i + 1], st.wspolczynniki[2*i + 2]);
      st.prawdopodobienstwa.push_back(porcja);
      norma2 += porcja;
    }
  porcja = prawa.norma_kwadr(st.poziom, st.wspolczynniki.back());
  st.prawdopodobienstwa.push_back(porcja);
  norma2 += porcja;
  for(int i = 0; i <= (int) st.prawdopodobienstwa.size() - 1; i++)
    {
      st.prawdopodobienstwa[i] /= norma2;
    }
  return sqrt(norma2);
}
/*****************************************************************************/
void struktura::normowanie()
{
  std::vector<stan>::iterator it = rozwiazania.begin();
  //  std::clog<<"Liczba stanow = "<<rozwiazania.size()<<"\n";
   double norma;
  while(it != rozwiazania.end())
    {
      norma = norma_stanu(*it);
      if (mInfo) std::clog<<"Norma dla E = "<<(it->poziom)<<" wynosi "<<norma<<"\n"; // LUKASZ
      for( int i = 0; i <= (int) it->wspolczynniki.size() - 1; i++)
	{
	  it->wspolczynniki[i] /= norma;
	}
      it++;
    }
}
/*****************************************************************************/
double struktura::ilenosnikow(double qFl, double T)
{
  double tylenosnikow = 0;
  double niepomnozone;
  double calkazFD;
  double szer;
  double gam32 = sqrt(pi)/2; // Gamma(3/2)
  double kT = kB*T;
  std::vector<stan>::iterator it = rozwiazania.end();
  //  std::clog<<"Liczba stanow = "<<rozwiazania.size()<<"\n";
  while(it != rozwiazania.begin()) // suma po poziomach
    {
      --it;
      calkazFD = kT*log(1+exp((qFl - it->poziom)/kT)); 
      niepomnozone = lewa.norma_kwadr(it->poziom, it->wspolczynniki.front()) * lewa.masa_r;
      niepomnozone += prawa.norma_kwadr(it->poziom, it->wspolczynniki.back()) * prawa.masa_r;
      for(int i = 0; i<= (int) kawalki.size() - 1; i++) // Suma po warstwach
	{
	  niepomnozone += kawalki[i].norma_kwadr(it->poziom, it->wspolczynniki[2*i + 1], it->wspolczynniki[2*i + 2]) * kawalki[i].masa_r;
	}
      tylenosnikow += niepomnozone*calkazFD/pi; // spiny juz sa
    }
  niepomnozone = 0;
  double Egorna = lewa.y;
  for(int i = 0; i<= (int) kawalki.size() - 1; i++) // Po stanach 3D
    {
      szer = kawalki[i].x_kon - kawalki[i].x_pocz;
      niepomnozone += szer*sqrt(2*kawalki[i].masa_p(Egorna))*kawalki[i].masa_r; // Spin juz jest
    }
  tylenosnikow += niepomnozone * kT*gam32*sqrt(kT)*2*fermiDiracHalf((qFl-Egorna)/(kB*T))/(2*pi*pi); // Powinno byc rozpoznawanie, ktore warstwy wystaja ponad poziom bariery, ale w GSL nie ma niezupelnych calek F-D. Mozna by je przyblizyc niezupelnymi funkcjami Gamma.
  return tylenosnikow;
}
/*****************************************************************************/
std::vector<double> struktura::koncentracje_w_warstwach(double qFl, double T)
{
  double tylenosnikow = 0;
  double niepomnozone;
  double calkazFD, calkaFD12;
  double szer;
  double gam32 = sqrt(pi)/2; // Gamma(3/2)
  double kT = kB*T;
  double Egorna = lewa.y;
  calkaFD12 = fermiDiracHalf((qFl-Egorna)/(kB*T));
  std::vector<double> koncentr(kawalki.size() + 2);
  std::vector<stan>::iterator it;
  if (mInfo) std::clog<<"Liczba stanow = "<<rozwiazania.size()<<"\n"; // LUKASZ
  koncentr[0] = sqrt(2*lewa.masa_p)*lewa.masa_r * kT*gam32*sqrt(kT)*2*calkaFD12/(2*pi*pi);
  for(int i = 0; i<= (int) kawalki.size() - 1; i++) // Suma po warstwach
    {
      it = rozwiazania.end();
      if (mInfo) std::clog<<"i = "<<i<<"\n"; // LUKASZ
      tylenosnikow = 0;
      while(it != rozwiazania.begin()) // suma po poziomach
	{
	  --it;
	  if(i == 1)
	    {
	      if (mInfo) std::clog<<"Zawartosc dla poziomu "<<(it->poziom)<<" wynosi "<<(kawalki[i].norma_kwadr(it->poziom, it->wspolczynniki[2*i + 1], it->wspolczynniki[2*i + 2]))<<"\n"; // LUKASZ
		}
	  calkazFD = kT*log(1+exp((qFl - it->poziom)/kT)); 
	  niepomnozone = kawalki[i].norma_kwadr(it->poziom, it->wspolczynniki[2*i + 1], it->wspolczynniki[2*i + 2]) * kawalki[i].masa_r;
	  tylenosnikow += niepomnozone*calkazFD/pi; // spiny juz sa
	}
      szer = kawalki[i].x_kon - kawalki[i].x_pocz;
      koncentr[i + 1] = tylenosnikow/szer + sqrt(2*kawalki[i].masa_p(Egorna))*kawalki[i].masa_r* kT*gam32*sqrt(kT)*2*calkaFD12/(2*pi*pi);
    }
  koncentr.back() = koncentr.front();
  return koncentr;
}
/*****************************************************************************/
void struktura::struktura_do_pliku(std::ofstream & plik)
{
  std::vector<warstwa>::iterator it_war = kawalki.begin();
  plik<<dlugosc_na_A(lewa.iks)<<" "<<(lewa.y)<<"\n";
  while(it_war != kawalki.end())
    {
      plik<<dlugosc_na_A(it_war->x_pocz)<<" "<<(it_war->y_pocz)<<"\n";
      plik<<dlugosc_na_A(it_war->x_kon)<<" "<<(it_war->y_kon)<<"\n";
      it_war++;
    }
  plik<<dlugosc_na_A(prawa.iks)<<" "<<(prawa.y);
}
/*****************************************************************************/
void struktura::funkcje_do_pliku(std::ofstream & plik, double krok)
{
  if (mInfo) std::clog<<"W f_do_p"<<std::endl; // LUKASZ
  plik<<"#\t";
  std::vector<stan>::iterator it_stan = rozwiazania.begin();
  while(it_stan != rozwiazania.end())
    {
      plik<<" E="<<(it_stan->poziom);
      it_stan++;
    }
  plik<<"\n";
  double szer = prawa.iks - lewa.iks;
  double bok = szer/4;
  double x = lewa.iks - bok;
  while(x <= lewa.iks)
    {
      plik<<dlugosc_na_A(x)<<"\t";
      it_stan = rozwiazania.begin();
      while(it_stan != rozwiazania.end())
	{
	  plik<<lewa.funkcjafal(x, it_stan->poziom, it_stan->wspolczynniki[0])<<" ";
	  it_stan++;
	}
      plik<<"\n";
      x += krok;
    }
  for(int i = 0; i <= (int) kawalki.size() - 1; i++)
    {
      x = kawalki[i].x_pocz;
      while(x <= kawalki[i].x_kon)
	{
	  plik<<dlugosc_na_A(x)<<"\t";
	  it_stan = rozwiazania.begin();
	  while(it_stan != rozwiazania.end())
	    {
	      plik<<kawalki[i].funkcjafal(x, it_stan->poziom, it_stan->wspolczynniki[2*i + 1], it_stan->wspolczynniki[2*i + 2])<<" ";
	      it_stan++;
	    }
	  plik<<"\n";
	  x += krok;
	}
    }
  x = prawa.iks ;
  while(x <= prawa.iks + bok)
    {
      plik<<dlugosc_na_A(x)<<"\t";
      it_stan = rozwiazania.begin();
      while(it_stan != rozwiazania.end())
	{
	  plik<<prawa.funkcjafal(x, it_stan->poziom, it_stan->wspolczynniki.back())<<" ";
	  it_stan++;
	}
      plik<<"\n";
      x += krok;
    }
}
/*****************************************************************************/
double struktura::energia_od_k_na_ntym(double k, int nr_war, int n)
{
  warstwa * war;
  if(nr_war == 0)
    {
      war = &lewa;
    }
  else
    {
      if(nr_war == (int) kawalki.size() + 1)
	{
	  war = &kawalki[nr_war - 1];
	}
      else
	{
	  war = &prawa;
	}
    }
  return war->Eodk(k)+rozwiazania[n].poziom;
}
/*****************************************************************************
double dE_po_dl(size_t nr, chrop ch)
{
  double k = sqrt(2*)
  double licznik = 
}
*****************************************************************************/
obszar_aktywny::obszar_aktywny(struktura * elektron, const std::vector<struktura *> dziury, double Eg, std::vector<double> DSO, double chropo)
{
  przekr_max = 0.;
  pasmo_przew.push_back(elektron);
  pasmo_wal = dziury;
  chrop = chropo;
  double dE;
  for(int i = 0; i <= (int) pasmo_przew.size() - 1; i++) // przesuwa struktury, zeby 0 bylo w lewej
    {
      dE = - pasmo_przew[i]->lewa.y;
      pasmo_przew[i]->przesun_energie(dE);
    }
  //  std::clog<<"\nkonstr aktyw. Po pierwszym for";
  for(int i = 0; i <= (int) pasmo_wal.size() - 1; i++) // przesuwa struktury, zeby 0 bylo w lewej
    {
      dE = - pasmo_wal[i]->lewa.y;
      pasmo_wal[i]->przesun_energie(dE);
    }
  //  std::clog<<"\nkonstr aktyw. Po drugim for";
  Egcc.push_back(0);
  Egcv = std::vector<double>(dziury.size(), Eg);
  int liczba_war = dziury[0]->kawalki.size() + 2;
  DeltaSO.clear();
  for(int i = 0; i < liczba_war; ++i) // LUKASZ
      DeltaSO.push_back(DSO[i]);
  //DeltaSO.assign(liczba_war, DSO); // LUKASZ
  el_mac.reserve(liczba_war);
  for(int i = 0; i <= liczba_war - 1; i++)
    {
      el_mac.push_back(element(i));
      writelog(LOG_DETAIL, "M for layer %1%: %2%", i+1, el_mac[i]); // LUKASZ
    }
  zrob_macierze_przejsc();
}
/*****************************************************************************/
void obszar_aktywny::paryiprzekrycia_dopliku(ofstream & plik, int nr_c, int nr_v)
{
  struktura * el = pasmo_przew[nr_c];
  struktura * dziu = pasmo_wal[nr_v];
  A2D * m_prz = calki_przekrycia[nr_c][nr_v];
  double E0;
  for(int nrpoz_el = 0; nrpoz_el <= int(el->rozwiazania.size()) - 1; nrpoz_el++)
    for(int nrpoz_dziu = 0; nrpoz_dziu <= int(dziu->rozwiazania.size()) - 1; nrpoz_dziu++)
      {
	E0 = Egcv[nr_v] - Egcc[nr_c] + el->rozwiazania[nrpoz_el].poziom + dziu->rozwiazania[nrpoz_dziu].poziom;
	plik<<E0<<" "<<((*m_prz)[nrpoz_el][nrpoz_dziu])<<"\n";
      }
}
/*****************************************************************************/
double obszar_aktywny::min_przerwa_energetyczna()
{
  double przerwa = pasmo_przew[0]->dol + pasmo_wal[0]->dol + Egcv[0];
  for(int i = 0; i <= (int) pasmo_przew.size() - 1; i++)
    for(int j = 0; j <= (int) pasmo_wal.size() - 1; j++)
      {
	przerwa = (przerwa > pasmo_przew[i]->dol + pasmo_wal[j]->dol + Egcc[i] + Egcv[j])?(pasmo_przew[i]->dol + pasmo_wal[j]->dol + Egcc[i] + Egcv[j]):przerwa;
      }
  return przerwa;
}
/*****************************************************************************/
void obszar_aktywny::policz_calki(const struktura * elektron, const struktura * dziura, A2D & macierz, TNT::Array2D<std::vector<double> > & wekt_calk_kaw)
{
  double tymcz;
  if (mInfo) std::cerr<<"W funkcji policz_calki\n"; // LUKASZ
  for(int i = 0; i <= (int) elektron->rozwiazania.size() - 1; i++)
    for(int j = 0; j <= (int) dziura->rozwiazania.size() - 1; j++)
      {	
	tymcz = calka_ij(elektron, dziura, i, j, wekt_calk_kaw[i][j]);
	macierz[i][j] = tymcz*tymcz;
	//	std::cerr<<"\n w indeksie "<<i<<", "<<j<<" jest "<<macierz[i][j]<<"\n";
	if(macierz[i][j] > przekr_max)
	  {
	    przekr_max = macierz[i][j]; 
	  }
      }
}
/*****************************************************************************
void obszar_aktywny::policz_calki_kawalki(const struktura * elektron, const struktura * dziura, TNT::Array2D<vector<double> > & macierz)
{
  double tymcz;
  for(int i = 0; i <= (int) elektron->rozwiazania.size() - 1; i++)
    for(int j = 0; j <= (int) dziura->rozwiazania.size() - 1; j++)
      {	
	tymcz = calka_ij(elektron, dziura, i, j);
	macierz[i][j] = (tymcz < 0)? -tymcz:tymcz;
	if(macierz[i][j] > przekr_max)
	  {
	    przekr_max = macierz[i][j]; 
	  }
      }
}
*****************************************************************************/
double obszar_aktywny::iloczyn_pierwotna_bezpola(double x, int nr_war, const struktura * struk1, const struktura * struk2, int i, int j)
{
  double Ec = struk1->rozwiazania[i].poziom;
  double Ev = struk2->rozwiazania[j].poziom;
  double Ac, Bc, Av, Bv;
  double wynik;
  if(nr_war == 0) //lewa
    {
      Bc = struk1->rozwiazania[i].wspolczynniki[0];
      Bv = struk2->rozwiazania[j].wspolczynniki[0];
      wynik = (struk1->lewa.funkcjafal(x, Ec, Bc) * struk2->lewa.funkcjafal_prim(x, Ev, Bv) - struk1->lewa.funkcjafal_prim(x, Ec, Bc) * struk2->lewa.funkcjafal(x, Ev, Bv))/(struk1->lewa.k_kwadr(Ec) - struk2->lewa.k_kwadr(Ev));
      return wynik;
    }
  if(nr_war == (int) struk1->kawalki.size() + 1) //prawa
    {
      Ac = struk1->rozwiazania[i].wspolczynniki.back();
      Av = struk2->rozwiazania[j].wspolczynniki.back();
      wynik = (struk1->prawa.funkcjafal(x, Ec, Ac) * struk2->prawa.funkcjafal_prim(x, Ev, Av) - struk1->prawa.funkcjafal_prim(x, Ec, Ac) * struk2->prawa.funkcjafal(x, Ev, Av))/(struk1->prawa.k_kwadr(Ec) - struk2->prawa.k_kwadr(Ev));
      return wynik;
    }

  Ac = struk1->rozwiazania[i].wspolczynniki[2*nr_war + 1];
  Av = struk2->rozwiazania[j].wspolczynniki[2*nr_war + 1];
  Bc = struk1->rozwiazania[i].wspolczynniki[2*nr_war + 2];
  Bv = struk2->rozwiazania[j].wspolczynniki[2*nr_war + 2];

  wynik = (struk1->kawalki[nr_war].funkcjafal(x, Ec, Ac, Bc) * struk2->kawalki[nr_war].funkcjafal_prim(x, Ev, Av, Bv) - struk1->kawalki[nr_war].funkcjafal_prim(x, Ec, Ac, Bc) * struk2->kawalki[nr_war].funkcjafal(x, Ev, Av, Bv))/(struk1->kawalki[nr_war].k_kwadr(Ec) - struk2->kawalki[nr_war].k_kwadr(Ev));
  return wynik;
}
/*****************************************************************************/
double obszar_aktywny::calka_iloczyn_zpolem(int nr_war, const struktura * struk1, const struktura * struk2, int i, int j) // numeryczne całkowanie
{
  std::clog<<"\nW całk numer. Warstwa "<<nr_war<<" poziom el "<<i<<" poziom j "<<j<<"\n";;
  double krok = 1.; // na razie krok na sztywno przelm (ok 2.6) A
  double Ec = struk1->rozwiazania[i].poziom;
  double Ev = struk2->rozwiazania[j].poziom;
  double Ac, Bc, Av, Bv;
  double wynik = 0;
  double x_pocz = struk1->kawalki[nr_war].x_pocz;
  double x_kon = struk1->kawalki[nr_war].x_kon;
  double szer = x_kon - x_pocz;
  int podzial = ceill(szer/krok);
  krok = szer/podzial; // wyrównanie kroku
  Ac = struk1->rozwiazania[i].wspolczynniki[2*nr_war + 1];
  Av = struk2->rozwiazania[j].wspolczynniki[2*nr_war + 1];
  Bc = struk1->rozwiazania[i].wspolczynniki[2*nr_war + 2];
  Bv = struk2->rozwiazania[j].wspolczynniki[2*nr_war + 2];
  double x = x_pocz + krok/2;
  for(int i = 0; i<= podzial - 1; i++)
    {
      std::clog<<"\nwynik = "<<wynik<<" ";
      wynik += struk1->kawalki[nr_war].funkcjafal(x, Ec, Ac, Bc) * struk2->kawalki[nr_war].funkcjafal(x, Ev, Av, Bv);
      x += krok;
    }
  wynik *= krok;
  return wynik;
}
/*****************************************************************************/
double obszar_aktywny::calka_ij(const struktura * elektron, const struktura * dziura, int i, int j, vector<double> & wektor_calk_kaw)
{
  double Ec = elektron->rozwiazania[i].poziom;
  double Ev = dziura->rozwiazania[j].poziom;
  double xk = elektron->lewa.iks;
  double Ac, Bc, Av, Bv;
  double calk_kaw = 0;
  Bc = elektron->rozwiazania[i].wspolczynniki[0];
  Bv = dziura->rozwiazania[j].wspolczynniki[0];
  double calka = elektron->lewa.funkcjafal(xk, Ec, Bc) * dziura->lewa.funkcjafal_prim(xk, Ev, Bv) - elektron->lewa.funkcjafal_prim(xk, Ec, Bc) * dziura->lewa.funkcjafal(xk, Ev, Bv);
  calka = calka/(elektron->lewa.k_kwadr(Ec) - dziura->lewa.k_kwadr(Ev)); // Taki sprytny wzor na calke mozna dostac
  wektor_calk_kaw.push_back(calka);
  //  std::clog<<" calka w lewej = "<<calka<<"\n";
  double pierwk, pierwp, xp;
  for(int war = 0; war <= (int) elektron->kawalki.size() - 1; war++)
    {
      if( (elektron->kawalki[war].pole == 0) && (dziura->kawalki[war].pole == 0) ) //trzeba posprzatac, i wywolywać funkcje tutaj
	{
	  xp = elektron->kawalki[war].x_pocz;
	  xk = elektron->kawalki[war].x_kon;
	  
	  Ac = elektron->rozwiazania[i].wspolczynniki[2*war + 1];
	  Av = dziura->rozwiazania[j].wspolczynniki[2*war + 1];
	  Bc = elektron->rozwiazania[i].wspolczynniki[2*war + 2];
	  Bv = dziura->rozwiazania[j].wspolczynniki[2*war + 2];
	  
	  pierwk = elektron->kawalki[war].funkcjafal(xk, Ec, Ac, Bc) * dziura->kawalki[war].funkcjafal_prim(xk, Ev, Av, Bv) - elektron->kawalki[war].funkcjafal_prim(xk, Ec, Ac, Bc) * dziura->kawalki[war].funkcjafal(xk, Ev, Av, Bv);
	  pierwp = elektron->kawalki[war].funkcjafal(xp, Ec, Ac, Bc) * dziura->kawalki[war].funkcjafal_prim(xp, Ev, Av, Bv) - elektron->kawalki[war].funkcjafal_prim(xp, Ec, Ac, Bc) * dziura->kawalki[war].funkcjafal(xp, Ev, Av, Bv);
	  calk_kaw = (pierwk - pierwp)/(elektron->kawalki[war].k_kwadr(Ec) - dziura->kawalki[war].k_kwadr(Ev));
	  wektor_calk_kaw.push_back(calk_kaw);
	  calka += calk_kaw;
	}
      else // numerycznie na razie
	{
	  calk_kaw = calka_iloczyn_zpolem(war, elektron, dziura, i, j);
	  wektor_calk_kaw.push_back(calk_kaw);
	  calka += calk_kaw;
	}
      //std::clog<<"\ncalka kawalek = "<<calk_kaw<<"\n";
    }
  xp = elektron->prawa.iks;
  
  Ac = elektron->rozwiazania[i].wspolczynniki.back();
  Av = dziura->rozwiazania[j].wspolczynniki.back();
  calk_kaw = -(elektron->prawa.funkcjafal(xp, Ec, Ac) * dziura->prawa.funkcjafal_prim(xp, Ev, Av) - elektron->prawa.funkcjafal_prim(xp, Ec, Ac) * dziura->prawa.funkcjafal(xp, Ev, Av))/(elektron->prawa.k_kwadr(Ec) - dziura->prawa.k_kwadr(Ev));  // -= bo calka jest od xp do +oo
  wektor_calk_kaw.push_back(calk_kaw);
  calka += calk_kaw;
  return calka;
}
/*****************************************************************************/
void obszar_aktywny::zrob_macierze_przejsc()
{
  if (mInfo) std::cerr<<"W funkcji zrob_macierze_przejsc\n"; // LUKASZ
  A2D * macierz_calek;
  TNT::Array2D<std::vector<double> > * macierz_kawalkow;
  calki_przekrycia.resize(pasmo_przew.size());
  calki_przekrycia_kawalki.resize(pasmo_przew.size()); 
  for(int i = 0; i <= (int) calki_przekrycia.size() - 1; i++)
    {
      calki_przekrycia[i].resize(pasmo_wal.size());
      calki_przekrycia_kawalki[i].resize(pasmo_wal.size());
    }
  for(int c = 0; c <= (int) pasmo_przew.size() - 1; c++)
    {
      for(int v = 0; v <= (int) pasmo_wal.size() - 1; v++)
	{
	  macierz_calek = new A2D(pasmo_przew[c]->rozwiazania.size(), pasmo_wal[v]->rozwiazania.size());
	  macierz_kawalkow = new TNT::Array2D<std::vector<double> >(pasmo_przew[c]->rozwiazania.size(), pasmo_wal[v]->rozwiazania.size());
	  policz_calki(pasmo_przew[c], pasmo_wal[v], *macierz_calek, *macierz_kawalkow);
	  calki_przekrycia[c][v] = macierz_calek;
	  calki_przekrycia_kawalki[c][v] = macierz_kawalkow;
	  if (mInfo) std::clog<<"Macierz przejsc:"<<(*macierz_calek)<<"\n"; // LUKASZ
	}
    }
}
/*****************************************************************************
void obszar_aktywny::zrob_macierze_kawalkow()
{
  TNT::Array2D<vector<double> > * macierz_kaw;
  calki_przekrycia_kawalki.resize(pasmo_przew.size());
  for(int i = 0; i <= (int) calki_przekrycia_kawalki.size() - 1; i++)
    {
      calki_przekrycia_kawalki[i].resize(pasmo_wal.size());
    }
  for(int c = 0; c <= (int) pasmo_przew.size() - 1; c++)
    {
      for(int v = 0; v <= (int) pasmo_wal.size() - 1; v++)
	{
	  macierz = new TNT::Array2D<vector<double> >(pasmo_przew[c]->rozwiazania.size(), pasmo_wal[v]->rozwiazania.size());
	  policz_calki_kawalki(pasmo_przew[c], pasmo_wal[v], *macierz);
	  calki_przekrycia_kawalki[c][v] = macierz;
	  std::clog<<"Macierz przejsc:"<<(*macierz)<<"\n";
	}
    }
}
*****************************************************************************/
double obszar_aktywny::element(int nr_war) // Do przerobienia
{
  warstwa * warc, * warv;
  if(nr_war == 0)
    {
      warc = & pasmo_przew[0]->lewa;
      warv = & pasmo_wal[0]->lewa;
    }
  else
    {
      if(nr_war < (int) pasmo_przew[0]->kawalki.size() + 1)
	{
	  warc = & pasmo_przew[0]->kawalki[nr_war - 1];
	  warv = & pasmo_wal[0]->kawalki[nr_war - 1];
	}
      else
	{
	  warc = & pasmo_przew[0]->prawa;
	  warv = & pasmo_wal[0]->prawa;
	}
    }
  double Eg = Egcv[0] + warc->y_pocz + warv->y_pocz; 
  if (mInfo) std::cerr<<"\nW elemencie: Eg = "<<Eg<<"\n"; // LUKASZ
  return (1/warc->masa_p(0.) - 1)*(Eg+DeltaSO[nr_war])*Eg/(Eg+2*DeltaSO[nr_war]/3)/2;
}
/*****************************************************************************/
double gain::kodE(double E, double mc, double mv)
{
  double m=(1/mc+1/mv);
  return sqrt(2*E/m);
}
/*****************************************************************************/
double gain::rored(double, double mc, double mv)
{
  double m=(1/mc+1/mv);
  return 1/(m*2*struktura::pi*szer_do_wzmoc);
}
/*****************************************************************************/
double gain::erf_dorored(double E, double E0, double sigma)
{
  return 0.5*(1 + erf((E - E0)/(sqrt(2)*sigma)));
}
/*****************************************************************************/
double gain::rored_posz(double E, double E0, double mc, double mv, double sigma) // gestosc do chopowatej studni o nominalnej roznicy energii poziomow E0. Wersja najprostsza -- jedno poszerzenie na wszystko
{
  double m=(1/mc+1/mv);
  //  double sigma = posz_en
  return erf_dorored(E, E0, sigma)/(m*2*struktura::pi*szer_do_wzmoc);
}
/*****************************************************************************/
gain::gain() // LUKASZ
{

}
/*****************************************************************************/
gain::gain(plask::shared_ptr<obszar_aktywny> obsz, double konc_pow, double temp, double wsp_zal)
    : pasma(obsz)
{
  pasma = obsz;
  nosniki_c = przel_gest_z_cm2(konc_pow);
  nosniki_v = nosniki_c;
  T = temp;
  n_r = wsp_zal;
  szer_do_wzmoc = pasma->pasmo_przew[0]->kawalki.back().x_kon - pasma->pasmo_przew[0]->kawalki.front().x_pocz;
  policz_qFlc();
  policz_qFlv();
}
/*****************************************************************************/
double gain::policz_qFlc()
{
  double Fp, Fk, krok;
  double np, nk;
  Fp = -pasma->Egcv[0]/2; // polowa przerwy
  Fk = pasma->pasmo_przew[0]->gora; // skrajna bariera
  krok = pasma->pasmo_przew[0]->gora - pasma->pasmo_przew[0]->dol;
  np = nosniki_w_c(Fp);
  nk = nosniki_w_c(Fk);
  if( nosniki_c < np )
    {
      std::cerr<<"Za malo nosnikow!\n";
      abort();
    }
  while( nk < nosniki_c )
    {
      Fp = Fk;
      Fk += krok;
      nk = nosniki_w_c(Fk);
    }
  double (gain::*fun)(double) = &gain::gdzie_qFlc;
  //  std::clog<<"Sieczne Fermi\n";
  qFlc = sieczne(fun, Fp, Fk);
  return sieczne(fun, Fp, Fk);
}
/*****************************************************************************/
double gain::policz_qFlv()
{
  double Fp, Fk, krok;
  double np, nk;
  Fp = -pasma->Egcv[0]/2; // polowa przerwy
  Fk = pasma->pasmo_wal[0]->gora; // skrajna bariera
  krok = pasma->pasmo_wal[0]->gora - pasma->pasmo_wal[0]->dol;
  np = nosniki_w_v(Fp);
  nk = nosniki_w_v(Fk);
  if( nosniki_v < np )
    {
      std::cerr<<"Za malo nosnikow!\n";
      abort();
    }
  while( nk < nosniki_v )
    {
      Fp = Fk;
      Fk += krok;
      nk = nosniki_w_v(Fk);
    }
  double (gain::*fun)(double) = &gain::gdzie_qFlv;
  //  std::clog<<"Sieczne Fermi\n";
  qFlv = - sieczne(fun, Fp, Fk);
  return - sieczne(fun, Fp, Fk); // minus, bo energie sa odwrocone, bo F-D opisuje obsadzenia elektronowe
}
/*****************************************************************************/
double gain::getT()
{
  return T;
}
/*****************************************************************************/
double gain::Get_gain_at_n(double E, double hQW, double iL, double iTau)
{
    /*double tGehh, tGelh;
    tGehh = tGelh = 0.;

    tGehh = wzmocnienie_od_pary_pasm(E, 0, 0) / iL;
    tGelh = wzmocnienie_od_pary_pasm(E, 0, 1) / iL;
    return (tGehh+tGelh);*/
    //WRITELOG(LOG_DETAIL, "Tau in kubly: %1% ps", iTau);
    if (!iTau) return ( wzmocnienie_calk_bez_splotu(E) / iL ); //20.10.2014 adding lifetime
    else return ( wzmocnienie_calk_ze_splotem(E,phys::hb_eV*1e12/iTau) / iL ); //20.10.2014 adding lifetime
}
/*****************************************************************************/
double gain::gdzie_qFlc(double E)
{
  return nosniki_w_c(E) - nosniki_c;
}
/*****************************************************************************/
double gain::gdzie_qFlv(double E)
{
  return nosniki_w_v(E) - nosniki_v;
}
/*****************************************************************************/
double gain::nosniki_w_c(double Fl)
{
  double przes;
  double n = pasma->pasmo_przew[0]->ilenosnikow(Fl, T);
  for(int i = 1; i <=  (int) pasma->pasmo_przew.size() - 1; i++)
    {
      //      std::clog<<"Drugie pasmo c\n";
      przes = -pasma->Egcc[i]; // bo dodatnie Egcc oznacza przesuniecie w dol 
      n += pasma->pasmo_przew[i]->ilenosnikow(Fl - przes, T);
    }
  //  std::clog<<"Fl = "<<Fl<<"\tgest pow w 1/cm^2 = "<<przel_gest_na_cm2(n)<<"\n";
  return n;
}
/*****************************************************************************/
double gain::nosniki_w_v(double Fl)
{
  double przes;
  double n = pasma->pasmo_wal[0]->ilenosnikow(Fl, T);
  for(int i = 1; i <=  (int) pasma->pasmo_wal.size() - 1; i++)
    {
      //      std::clog<<"Drugie pasmo v\n";
      przes = pasma->Egcv[0] - pasma->Egcv[i]; // przerwa Ev_i - Ev_0. dodatnie oznacza ze, v_1 jest blizej c, czyli poziom Fermiego trzeba podniesc dla niego (w geometrii struktury)
      //      std::cerr<<"policzone przes\n";
      n += pasma->pasmo_wal[i]->ilenosnikow(Fl + przes, T);
    }
  //  std::clog<<"Fl = "<<Fl<<"\tgest pow w 1/cm^2 = "<<przel_gest_na_cm2(n)<<"\n";
  return n;
}
/*****************************************************************************/
double gain::sieczne(double (gain::*f)(double), double pocz, double kon)
{
  std::clog.precision(12);
  //const double eps = 1e-14; // limit zmian x
  double dokl = 1e-6;
  double xp = kon;
  double xl = pocz;
  /*
  if((this->*f)(pocz)*(this->*f)(kon) > 0)
    {
      std::cerr<<"Złe krańce przedziału!\n";
      abort();
    }
  */
  double x, fc, fp, fl, xlp, xpp; // -p -- poprzednie krance 
  fl = (this->*f)(xl);
  fp = (this->*f)(xp);
  xlp = (xl + xp)/2;
  xpp = xlp; // żeby na pewno było na początku różne od prawdzwych końców
  do
    {
      x = xp - fp*(xp - xl)/(fp - fl);
      fc = (this->*f)(x);
      if(fc == 0)
	{
	  break;
	}
      if(fc*fl < 0) // c bedzie nowym prawym koncem
	{
	  //	  std::clog<<"xlp - xl = "<<(xlp - xl)<<"\n";
	  if(xlp == xl) // trzeba poprawic ten kraniec (metoda Illinois)
	    {
	      //	      std::clog<<"Lewy Illinois\n";
	      fl = fl/2;
	    }
	  xpp = xp;
	  xp = x;
	  xlp = xl;
	  fp = fc;
	}
      else // c bedzie nowym lewym koncem
	{
	  //	  std::clog<<"xpp - xp = "<<(xpp - xp)<<"\n";
	  if(xpp == xp) // trzeba poprawic ten kraniec (metoda Illinois)
	    {
	      //  std::clog<<"Prawy Illinois\n";
	      fp = fp/2;
	    }
	  xlp = xl;
	  xl = x;
	  xpp = xp;
	  fl = fc;
	  //	  fp = (this->*f)(kon);
	}
      //      std::clog<<"x = "<<x<<"\tf(x) = "<<fc<<"\txl = "<<xl<<" xp = "<<xp<<"\n";//<<"\txp - x = "<<(xp - x)<<"\n";
    }
  while(xp - xl >= dokl);
  return x;
}
/*****************************************************************************/
double gain::L(double x, double b)
{
  return 1/(M_PI*b)/(1 + x/b*x/b );
}
/*****************************************************************************/
double gain::wzmocnienie_calk_ze_splotem(double E, double b, double blad) // podzial na kawalek o promieniu Rb wokol 0 i reszte
{
  //  double blad = 0.005;
  // b energia do poszerzenia w lorentzu
  struktura * el = pasma->pasmo_przew[0];
  struktura * dziu = pasma->pasmo_wal[0];
  double E0pop = pasma->Egcv[0] - pasma->Egcc[0] + el->rozwiazania[0].poziom + dziu->rozwiazania[0].poziom; // energia potencjalna + energia prostopadla
  double E0min=E0pop;;
  for(int nr_c = 0; nr_c <= (int) pasma->pasmo_przew.size() - 1; nr_c++)
    {
      el = pasma->pasmo_przew[nr_c];
      for(int nr_v = 0; nr_v <= (int) pasma->pasmo_wal.size() - 1; nr_v++)
	{
	  dziu = pasma->pasmo_wal[nr_v];
	  E0min = pasma->Egcv[nr_c] - pasma->Egcc[nr_v] + el->rozwiazania[0].poziom + dziu->rozwiazania[0].poziom;
	  E0min = (E0pop >= E0min)? E0min: E0pop;
	}
    }
  double a = 2*(E0min - pasma->min_przerwa_energetyczna())*pasma->chrop;
  // maksima (oszacowne z góry) kolejnych pochodnych erfc(x/a)
  double em = 2.;
  double epm = 1.13/a;
  double eppm = 1./(a*a);
  double epppm = 2.5/(a*a*a);
  double eppppm = 5/(a*a*a*a);
  // maxima  (oszacowne z góry) kolejnych pochodnych lorentza (x/b), pomnożeone przez b
  double lm = 1/M_PI;
  double lpm = 0.2/b;
  double lppm = 0.7/(b*b);
  double lpppm = 1.5/(b*b*b);
  double lppppm = 24/M_PI/(b*b*b*b);
  double czwpoch0b = (em*lppppm + 4*epm*lpppm + 6*eppm*lppm + 4*epppm*lpm + eppppm*lm); // szacowanie (grube) czwartej pochodnej dla |x| < 1, pomnożone przez b
  double R = 3.; // w ilu b jest zmiana zagęszczenia
  double jedenplusR2 = 1 + R*R; 
  double lmR = 1/(M_PI*jedenplusR2); // oszacowania modułów kolejnych pochodnych przez wartość w R, bo dla R >=2 są malejące
  double lpmR = 2*R/(M_PI*jedenplusR2*jedenplusR2)/b;
  double lppmR = (6*R*R - 2)/(M_PI*jedenplusR2*jedenplusR2*jedenplusR2)/(b*b);
  double lpppmR = 24*R*(R*R - 1)/(M_PI*jedenplusR2*jedenplusR2*jedenplusR2*jedenplusR2)/(b*b*b);
  double lppppmR = (120*R*R*R*R - 240*R*R + 24)/(M_PI*jedenplusR2*jedenplusR2*jedenplusR2*jedenplusR2*jedenplusR2)/(b*b*b*b);
  double czwpochRb = (em*lppppmR + 4*epm*lpppmR + 6*eppm*lppmR + 4*epppm*lpmR + eppppm*lmR);

  int n0 = pow(2*R,5./4)*b*pow(czwpoch0b/(180*blad),0.25);
  int nR = pow((32-R),5./4)*b*pow(czwpochRb/(180*blad),0.25);
  if(n0%2) // ny powinny być parzyste
    {
      n0+=1;
    }
  else
    {
      n0 += 2;
    }
  if(nR%2) // ny powinny być parzyste
    {
      nR+=1;
    }
  else
    {
      nR += 2;
    }
  //  nR *= 2; // chwilowo, do testów
  double szer = 2*R*b;
  double h = szer/n0;
  double x2j, x2jm1, x2jm2;
  double calka1 = 0.;
  for(int j = 1; j <= n0/2; j++) //w promieniu Rb
    {
      x2j = -R*b + 2*j*h;
      x2jm1 = x2j - h;
      x2jm2 = x2jm1 - h;
      calka1 += L(x2jm2,b)*wzmocnienie_calk_bez_splotu(E-x2jm2) + 4*L(x2jm1,b)*wzmocnienie_calk_bez_splotu(E-x2jm1) + L(x2j,b)*wzmocnienie_calk_bez_splotu(E-x2j);
    }
  calka1 *= h/3;
  // dla -32b < x  -Rb
  szer = (32-R)*b;
  h = szer/nR;
  double calka2 = 0.;
  for(int j = 1; j <= nR/2; j++) // ujemne pół
    {
      x2j = -32*b + 2*j*h;
      x2jm1 = x2j - h;
      x2jm2 = x2jm1 - h;
      calka2 += L(x2jm2,b)*wzmocnienie_calk_bez_splotu(E-x2jm2) + 4*L(x2jm1,b)*wzmocnienie_calk_bez_splotu(E-x2jm1) + L(x2j,b)*wzmocnienie_calk_bez_splotu(E-x2j);
    }
  for(int j = 1; j <= nR/2; j++) // dodatnie pół
    {
      x2j = R*b + 2*j*h;
      x2jm1 = x2j - h;
      x2jm2 = x2jm1 - h;
      calka2 += L(x2jm2,b)*wzmocnienie_calk_bez_splotu(E-x2jm2) + 4*L(x2jm1,b)*wzmocnienie_calk_bez_splotu(E-x2jm1) + L(x2j,b)*wzmocnienie_calk_bez_splotu(E-x2j);
    }
  calka2 *= h/3;
  double calka = calka1 + calka2;
  //std::clog<<"\na = "<<a<<"\t4poch = "<<czwpoch0b<<"\tn0 = "<<n0<<"\tnR = "<<nR<<"\tcalka = "<<calka<<"\n";
  return calka;
}
/*****************************************************************************/
double gain::wzmocnienie_od_pary_pasm(double E, size_t nr_c, size_t nr_v)
{
  //  std::cerr<<"\npasmo walencyjna nr "<<nr_v<<"\n";
  if ( (nr_c >= pasma->pasmo_przew.size()) || (nr_v >= pasma->pasmo_wal.size()) ) // added by LUKASZ
    return 0.;
  else
  {
    struktura * el = pasma->pasmo_przew[nr_c];
    struktura * dziu = pasma->pasmo_wal[nr_v];
    A2D * m_prz = pasma->calki_przekrycia[nr_c][nr_v];
    double wzmoc = 0;
    double minimalna_przerwa = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->dol + dziu->dol;
    double min_E0 = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->rozwiazania[0].poziom + dziu->rozwiazania[0].poziom;
    double posz_en = 2*(min_E0 - minimalna_przerwa)*pasma->chrop;
    double E0;
    for(int nrpoz_el = 0; nrpoz_el <= int(el->rozwiazania.size()) - 1; nrpoz_el++)
      for(int nrpoz_dziu = 0; nrpoz_dziu <= int(dziu->rozwiazania.size()) - 1; nrpoz_dziu++)
      {
        //std::cerr<<"\nprzekrycie w "<<nrpoz_el<<", "<<nrpoz_dziu;
        //std::cerr<<" = "<<(*m_prz)[nrpoz_el][nrpoz_dziu];
        E0 = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->rozwiazania[nrpoz_el].poziom + dziu->rozwiazania[nrpoz_dziu].poziom;
        if( ((*m_prz)[nrpoz_el][nrpoz_dziu] > 0.005) && (E-E0 > -5*posz_en) ) // czy warto tracić czas
        {
          wzmoc += wzmocnienie_od_pary_poziomow(E, nr_c, nrpoz_el, nr_v, nrpoz_dziu);
          //std::cerr<<"\nnowy wzmoc = "<<wzmoc;
        }
      }
    return wzmoc;
  }
}
/*****************************************************************************/
double gain::spont_od_pary_pasm(double E, size_t nr_c, size_t nr_v)
{
  //  std::cerr<<"\npasmo walencyjna nr "<<nr_v<<"\n";
  struktura * el = pasma->pasmo_przew[nr_c];
  struktura * dziu = pasma->pasmo_wal[nr_v];
  A2D * m_prz = pasma->calki_przekrycia[nr_c][nr_v];
  double spont = 0;
  double minimalna_przerwa = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->dol + dziu->dol;
  double min_E0 = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->rozwiazania[0].poziom + dziu->rozwiazania[0].poziom;
  double posz_en = 2*(min_E0 - minimalna_przerwa)*pasma->chrop;
  double E0;
  for(int nrpoz_el = 0; nrpoz_el <= int(el->rozwiazania.size()) - 1; nrpoz_el++)
    for(int nrpoz_dziu = 0; nrpoz_dziu <= int(dziu->rozwiazania.size()) - 1; nrpoz_dziu++)
	{
	  //	  std::cerr<<"\nprzekrycie w "<<nrpoz_el<<", "<<nrpoz_dziu;
	  //	  std::cerr<<" = "<<(*m_prz)[nrpoz_el][nrpoz_dziu];
	  E0 = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->rozwiazania[nrpoz_el].poziom + dziu->rozwiazania[nrpoz_dziu].poziom;
	  if( ((*m_prz)[nrpoz_el][nrpoz_dziu] > 0.005) && (E-E0 > -5*posz_en) ) // czy warto tracić czas
	    {
	      spont += spont_od_pary_poziomow(E, nr_c, nrpoz_el, nr_v, nrpoz_dziu);
	      //	      std::cerr<<"\nnowy wzmoc = "<<wzmoc;
	    }
	}
  return spont;
}
/*****************************************************************************/
double gain::wzmocnienie_od_pary_poziomow(double E, size_t nr_c, int poz_c, size_t nr_v, int poz_v)
{
  double wynik;
  double cos2tet; // zmiana elementu macierzowego z k_prost
  struktura * el = pasma->pasmo_przew[nr_c];
  struktura * dziu = pasma->pasmo_wal[nr_v];
  double Eg; // lokalna przerwa energetyczna
  double E0 = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->rozwiazania[poz_c].poziom + dziu->rozwiazania[poz_v].poziom; // energia potencjalna + energia prostopadla
  //  std::cerr<<"\npoziom_el = "<<el->rozwiazania[poz_c].poziom<<"\n";
  //  std::cerr<<"\n\nE = "<<E<<" poziom c = "<<poz_c<<" poziom v = "<<poz_v<<" E0 = "<<E0<<"\n";
  double przekr_w_war;
  if (mInfo) std::cerr<<"\nTyp dziury = "<<dziu->typ; // LUKASZ
  std::vector<double> calki_kawalki;

  // Uśrednianie masy efektywnej
  //  std::cerr<<"\n prawd = "<<el->rozwiazania[poz_c].prawdopodobienstwa[0];
  double srednia_masa_el = el->rozwiazania[poz_c].prawdopodobienstwa[0]*el->lewa.masa_r;
  double srednia_masa_dziu = dziu->rozwiazania[poz_v].prawdopodobienstwa[0]*dziu->lewa.masa_r;
  for(int i = 0; i <= (int) el->kawalki.size() - 1 ; i++)
    {
      //      std::cerr<<"\n prawd = "<<el->rozwiazania[poz_c].prawdopodobienstwa[i+1];
      srednia_masa_el += el->rozwiazania[poz_c].prawdopodobienstwa[i+1]*el->kawalki[i].masa_r;
      srednia_masa_dziu += dziu->rozwiazania[poz_v].prawdopodobienstwa[i+1]*dziu->kawalki[i].masa_r;
    }
  int ost_ind = el->kawalki.size() + 1;
  //  std::cerr<<"\n prawd = "<<el->rozwiazania[poz_c].prawdopodobienstwa[ost_ind];
  srednia_masa_el += el->rozwiazania[poz_c].prawdopodobienstwa[ost_ind]*el->prawa.masa_r;
  srednia_masa_dziu += dziu->rozwiazania[poz_v].prawdopodobienstwa[ost_ind]*dziu->prawa.masa_r;
  //  std::cerr<<"\nŚrednie masy:\n elektron = "<< srednia_masa_el<<"\ndziura = "<<srednia_masa_dziu<<"\n";;
  double mnoznik_pol;
  // koniec uśredniania masy

  //  double srednie_k = (E-E0>0)?kodE(E-E0, srednia_masa_el, srednia_masa_dziu):0.;
  double srednie_k_zeznakiem = (E-E0>0)?kodE(E-E0, srednia_masa_el, srednia_masa_dziu):-kodE(E0-E, srednia_masa_el, srednia_masa_dziu);
  
  double minimalna_przerwa = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->dol + dziu->dol;
  double min_E0 = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->rozwiazania[0].poziom + dziu->rozwiazania[0].poziom;
  double posz_en = 2*(min_E0 - minimalna_przerwa)*pasma->chrop; // oszacowanie rozmycia poziomów z powodu chropowatości
  double sr_E_E0_dod = posz_en/(sqrt(2*struktura::pi))*exp(-(E-E0)*(E-E0)/(2*posz_en*posz_en)) + (E-E0)*erf_dorored(E, E0, posz_en);   //średnia energia kinetyczna w płaszczyźnie   
  Eg = pasma->Egcv[nr_v] - pasma->Egcc[nr_c];
  //      std::cerr<<"lewa Eg = "<<Eg<<"\n";
  //  cos2tet= (E0>Eg && E > E0)?(E0-Eg)/(E-Eg):1.0;
  cos2tet= (E0 > Eg)?(E0-Eg)/(sr_E_E0_dod + E0-Eg):1.;
  //      std::cerr<<"\ncos2tet = "<<cos2tet<<"\n";
  calki_kawalki =  (*(pasma->calki_przekrycia_kawalki[nr_c][nr_v]))[poz_c][poz_v]; 
  //      std::cerr<<"lewa po calkikawalki\n";
  przekr_w_war = calki_kawalki[0];
  //      std::cerr<<"lewa przed wynikiem\n";
  mnoznik_pol = (dziu->typ == struktura::hh)?(1 + cos2tet)/2:(5-3*cos2tet)/6; // polaryzacja TE
  wynik = sqrt(pasma->el_mac[0] * mnoznik_pol) * przekr_w_war;
  //      std::cerr<<"\nprzekr_w_war = "<<przekr_w_war<<" el_mac = "<<pasma->el_mac[0]<<" wynik = "<<wynik;
  for(int i = 0; i <= (int) el->kawalki.size() - 1; i++)
    {
      Eg = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->kawalki[i].y_pocz + dziu->kawalki[i].y_pocz; // y_pocz na szybko, może co innego powinno być
      //      cos2tet= (E0>Eg && E > E0)?(E0-Eg)/(E-Eg):1.0;
      cos2tet= (E0 > Eg)?(E0-Eg)/(sr_E_E0_dod + E0-Eg):1.;
      mnoznik_pol = (dziu->typ == struktura::hh)?(1 + cos2tet)/2:(5-3*cos2tet)/6;
      //	  std::cerr<<"\nkawalek "<<i;
      //	  std::cerr<<" mnoz_pol = "<<mnoznik_pol<<" cos2tet = "<<cos2tet;
      
      przekr_w_war = calki_kawalki[i + 1];
      wynik += sqrt(pasma->el_mac[i + 1] * mnoznik_pol) * przekr_w_war;
      //	  std::cerr<<" przekr_w_war = "<<przekr_w_war<< " wynik = "<<wynik;
      //	  std::cerr<<"\nprzekr_w_war = "<<przekr_w_war<<" el_mac = "<<pasma->el_mac[i+1]<<" wynik = "<<wynik;
    }
  przekr_w_war = calki_kawalki.back();
  double energia_elektronu = el->rozwiazania[poz_c].poziom + srednie_k_zeznakiem*abs(srednie_k_zeznakiem)/(2*srednia_masa_el);
  double energia_dziury = dziu->rozwiazania[poz_v].poziom + srednie_k_zeznakiem*abs(srednie_k_zeznakiem)/(2*srednia_masa_dziu);
  double rozn_obsadzen = fc(energia_elektronu - pasma->Egcc[nr_c]) - fv(-energia_dziury + pasma->Egcv[0] - pasma->Egcv[nr_v]);
  //      std::cerr<<"\nen_el = "<<energia_elektronu<<" en_dziu = "<<energia_dziury<<" rozn_obsadzen = "<<rozn_obsadzen<<"\n";
  Eg = pasma->Egcv[nr_v] - pasma->Egcc[nr_c];
  //  cos2tet= (E0>Eg && E > E0)?(E0-Eg)/(E-Eg):1.0;
  cos2tet= (E0 > Eg)?(E0-Eg)/(sr_E_E0_dod + E0-Eg):1.;
  mnoznik_pol = (dziu->typ == struktura::hh)?(1 + cos2tet)/2:(5-3*cos2tet)/6;	  
  wynik += sqrt(pasma->el_mac.back() * mnoznik_pol) * przekr_w_war;
  //  std::cerr<<"\nprzekr_w_war = "<<przekr_w_war<<" el_mac = "<<pasma->el_mac.back()<<" wynik = "<<wynik<<" rored = "<<rored(srednie_k, srednia_masa_el, srednia_masa_dziu)<<"\n";
  wynik *= wynik; // dopiero teraz kwadrat modułu
  //      std::cerr<<"\nwynik = "<<wynik;
  
  wynik *= rored_posz(E, E0, srednia_masa_el, srednia_masa_dziu, posz_en) * rozn_obsadzen;
      //      std::cerr<<"\nrored = "<<rored_posz(E, E0, srednia_masa_el, srednia_masa_dziu, posz_en);
  return wynik*struktura::pi/(struktura::c*n_r*struktura::eps0*E)/struktura::przelm*1e8;
}
/*****************************************************************************/
double gain::spont_od_pary_poziomow(double E, size_t nr_c, int poz_c, size_t nr_v, int poz_v)
{
  double wynik;
  double cos2tet; // zmiana elementu macierzowego z k_prost
  struktura * el = pasma->pasmo_przew[nr_c];
  struktura * dziu = pasma->pasmo_wal[nr_v];
  double Eg; // lokalna przerwa energetyczna
  double E0 = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->rozwiazania[poz_c].poziom + dziu->rozwiazania[poz_v].poziom; // energia potencjalna + energia prostopadla
  //  std::cerr<<"spont: poziom c = "<<poz_c<<" poziom v = "<<poz_v<<" E0 = "<<E0<<"\n";
  double przekr_w_war;
  std::vector<double> calki_kawalki;
  double srednia_masa_el = el->rozwiazania[poz_c].prawdopodobienstwa[0]*el->lewa.masa_r;
  double srednia_masa_dziu = dziu->rozwiazania[poz_v].prawdopodobienstwa[0]*dziu->lewa.masa_r;
  for(int i = 0; i <= (int) el->kawalki.size() - 1 ; i++)
    {
      srednia_masa_el += el->rozwiazania[poz_c].prawdopodobienstwa[i+1]*el->kawalki[i].masa_r;
      srednia_masa_dziu += dziu->rozwiazania[poz_v].prawdopodobienstwa[i+1]*dziu->kawalki[i].masa_r;
    }
  int ost_ind = el->kawalki.size() + 1;
  srednia_masa_el += el->rozwiazania[poz_c].prawdopodobienstwa[ost_ind]*el->prawa.masa_r;
  srednia_masa_dziu += dziu->rozwiazania[poz_v].prawdopodobienstwa[ost_ind]*dziu->prawa.masa_r;
  double mnoznik_pol;

  double minimalna_przerwa = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->dol + dziu->dol;
  double min_E0 = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->rozwiazania[0].poziom + dziu->rozwiazania[0].poziom;
  double posz_en = 2*(min_E0 - minimalna_przerwa)*pasma->chrop;
  //  double erf_dor = erf_dorored(E, E0, posz_en);
  double srednie_k_zeznakiem = (E-E0>0)?kodE(E-E0, srednia_masa_el, srednia_masa_dziu):-kodE(E0-E, srednia_masa_el, srednia_masa_dziu);
  Eg = pasma->Egcv[nr_v] - pasma->Egcc[nr_c];
  double sr_E_E0_dod = posz_en/(sqrt(2*struktura::pi))*exp(-(E-E0)*(E-E0)/(2*posz_en*posz_en)) + (E-E0)*erf_dorored(E, E0, posz_en);   //średnia energia kinetyczna w płaszczyźnie   
  //  std::clog<<(E-E0)<<" "<<sr_E_E0_dod<<"\n";
  //  cos2tet= (E0>Eg && E > E0)?(E0-Eg)/(E-Eg):1.0;
  //  cos2tet = 1.0; // na chwilę
  cos2tet= (E0 > Eg)?(E0-Eg)/(sr_E_E0_dod + E0-Eg):1.;
  calki_kawalki =  (*(pasma->calki_przekrycia_kawalki[nr_c][nr_v]))[poz_c][poz_v]; 
  przekr_w_war = calki_kawalki[0];
  mnoznik_pol = (dziu->typ == struktura::hh)?(1 + cos2tet)/2:(5-3*cos2tet)/6; // polaryzacja TE
  wynik = sqrt(pasma->el_mac[0] * mnoznik_pol) * przekr_w_war;
  for(int i = 0; i <= (int) el->kawalki.size() - 1; i++)
    {
      //      std::cerr<<"kawalek "<<i<<"\n";
      Eg = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->kawalki[i].y_pocz + dziu->kawalki[i].y_pocz; // y_pocz na szybko, może co innego powinno być
      //      cos2tet= (E0>Eg && E > E0)?(E0-Eg)/(E-Eg):1.0;
      //      cos2tet = 1.0; // na chwilę
      //      cos2tet= (E0>Eg && E > E0)?(E0-Eg)/(sr_E_E0_dod + E0-Eg):1.0;
      cos2tet= (E0 > Eg)?(E0-Eg)/(sr_E_E0_dod + E0-Eg):1.;
      mnoznik_pol = (dziu->typ == struktura::hh)?(1 + cos2tet)/2:(5-3*cos2tet)/6;
      //      std::cerr<<" mnoz_pol = "<<mnoznik_pol<<" cos2tet = "<<cos2tet<<"\n";
      przekr_w_war = calki_kawalki[i + 1];
      wynik += sqrt(pasma->el_mac[i + 1] * mnoznik_pol) * przekr_w_war;
    }
  przekr_w_war = calki_kawalki.back();
  double energia_elektronu = el->rozwiazania[poz_c].poziom + srednie_k_zeznakiem*abs(srednie_k_zeznakiem)/(2*srednia_masa_el); // abs, żeby mieć znak, i energie poniżej E0
  double energia_dziury = dziu->rozwiazania[poz_v].poziom + srednie_k_zeznakiem*abs(srednie_k_zeznakiem)/(2*srednia_masa_dziu);
  double obsadzenia = fc(energia_elektronu - pasma->Egcc[nr_c])*(1 - fv(-energia_dziury + pasma->Egcv[0] - pasma->Egcv[nr_v]));
  //  std::cerr<<"\nen_el = "<<energia_elektronu<<" en_dziu = "<<energia_dziury<<" obsadz el = "<<fc(energia_elektronu - pasma->Egcc[nr_c])<<" obsadz dziu = "<<(1 - fv(-energia_dziury + pasma->Egcv[0] - pasma->Egcv[nr_v]))<<" obsadzenia = "<<obsadzenia<<" przesunięcie w fv "<<(pasma->Egcv[0] - pasma->Egcv[nr_v])<<"\n";
  Eg = pasma->Egcv[nr_v] - pasma->Egcc[nr_c];
  //    cos2tet= (E0>Eg && E > E0)?(E0-Eg)/(E-Eg):1.0;
  //  cos2tet = 1.0; // na chwilę
  //  cos2tet= (E0>Eg && E > E0)?(E0-Eg)/(sr_E_E0_dod + E0-Eg);//:1.0;
  cos2tet= (E0 > Eg)?(E0-Eg)/(sr_E_E0_dod + E0-Eg):1.;
  mnoznik_pol = (dziu->typ == struktura::hh)?(1 + cos2tet)/2:(5-3*cos2tet)/6;	  
  wynik += sqrt(pasma->el_mac.back() * mnoznik_pol) * przekr_w_war;
  wynik *= wynik; // dopiero teraz kwadrat modułu
  //  double posz_en = 2*(E0 - minimalna_przerwa)*pasma->chrop; // oszacowanie rozmycia poziomów z powodu chropowatości
  wynik *= rored_posz(E, E0, srednia_masa_el, srednia_masa_dziu, posz_en)*obsadzenia;
  //  std::cerr<<"typ_"<<dziu->typ<<" "<<E<<" "<<fc(energia_elektronu - pasma->Egcc[nr_c])<<" "<<(1 - fv(-energia_dziury + pasma->Egcv[0] - pasma->Egcv[nr_v]))<<"\n";
  //  std::cerr<<"\nrored = "<<rored_posz(E, E0, srednia_masa_el, srednia_masa_dziu, posz_en);
  return wynik*E*E*n_r/(struktura::c*struktura::c*struktura::c*struktura::pi*struktura::eps0)/(struktura::przelm*struktura::przelm*struktura::przelm)*1e24/struktura::przels*1e12; // w 1/(cm^3 s)
}
/*****************************************************************************/
double gain::wzmocnienie_calk_bez_splotu(double E)
{
  double wynik = 0.;
  for(int nr_c = 0; nr_c <= (int) pasma->pasmo_przew.size() - 1; nr_c++)
    for(int nr_v = 0; nr_v <= (int) pasma->pasmo_wal.size() - 1; nr_v++)
      wynik += wzmocnienie_od_pary_pasm(E, nr_c, nr_v);
  return wynik;
}
/*****************************************************************************/
void gain::profil_wzmocnienia_bez_splotu_dopliku(std::ofstream & plik, double pocz, double kon, double krok)
{
  double wynik;
    for(double E = pocz; E <= kon; E += krok)
    {
      wynik = 0;
      for(int nr_c = 0; nr_c <= (int) pasma->pasmo_przew.size() - 1; nr_c++)
	for(int nr_v = 0; nr_v <= (int) pasma->pasmo_wal.size() - 1; nr_v++)
	  wynik += wzmocnienie_od_pary_pasm(E, nr_c, nr_v);
      plik<<E<<" "<<wynik<<"\n";
    }
}
/*****************************************************************************/
void gain::profil_wzmocnienia_ze_splotem_dopliku(std::ofstream & plik, double pocz, double kon, double krok, double b)
{
  for(double E = pocz; E <= kon; E += krok)
    {
      plik<<E<<" "<<wzmocnienie_calk_ze_splotem(E, b)<<"\n";
    }
}
/*****************************************************************************/
void gain::profil_lumin_dopliku(std::ofstream & plik, double pocz, double kon, double krok)
{
  //  double wynik;
  /*
  std::vector<double> wklady;
  wklady.resize(pasmo_wal.size());
  */
    for(double E = pocz; E <= kon; E += krok)
    {
      plik<<E;
      //     wynik = 0;
      for(int nr_c = 0; nr_c <= (int) pasma->pasmo_przew.size() - 1; nr_c++)
	for(int nr_v = 0; nr_v <= (int) pasma->pasmo_wal.size() - 1; nr_v++)
	  {
	    //	    wklady[nr_v] = spont_od_pary_pasm(E, nr_c, nr_v);
	    //	    wynik += spont_od_pary_pasm(E, nr_c, nr_v);
	    plik<<" "<<spont_od_pary_pasm(E, nr_c, nr_v);
	  }
      //      plik<<E<<" "<<wynik<<"\n";
      plik<<"\n";
    }
}
/*****************************************************************************/
double gain::moc_lumin()
{
  struktura * el = pasma->pasmo_przew[0];
  struktura * dziu = pasma->pasmo_wal[0];
  double min_E0 = pasma->Egcv[0] - pasma->Egcc[0] + el->rozwiazania[0].poziom + dziu->rozwiazania[0].poziom;
  double lok_min_E0;
  for(int nr_c = 0; nr_c <= (int) pasma->pasmo_przew.size() - 1; nr_c++)
    for(int nr_v = 0; nr_v <= (int) pasma->pasmo_wal.size() - 1; nr_v++)
      {
	lok_min_E0 = pasma->Egcv[nr_v] - pasma->Egcc[nr_c] + el->rozwiazania[0].poziom + dziu->rozwiazania[0].poziom;
	min_E0 = (min_E0 < lok_min_E0)?min_E0:lok_min_E0;
      }
  double minimalna_przerwa = pasma->min_przerwa_energetyczna();
  double posz_en = 2*(min_E0 - minimalna_przerwa)*pasma->chrop;
  double pocz = min_E0 - 2*posz_en;
  double kon = min_E0 + 6*struktura::kB*T;
  kon = (pocz<kon)?kon:pocz + 2*struktura::kB*T;
  std::clog<<"\nW mocy. pocz = "<<pocz<<" kon = "<<kon<<"\n";
  double krok = struktura::kB*T/30;
  double wynik = 0;
  for(double E = pocz; E <= kon; E += krok)
    {
      for(int nr_c = 0; nr_c <= (int) pasma->pasmo_przew.size() - 1; nr_c++)
	for(int nr_v = 0; nr_v <= (int) pasma->pasmo_wal.size() - 1; nr_v++)
	  {
	    wynik += spont_od_pary_pasm(E, nr_c, nr_v);
	  }
    }
  return wynik*krok;
}
/*****************************************************************************/
double gain::fc(double E)
{
  double arg=(E-qFlc)/(struktura::kB*T);
  return 1/(1 + exp(arg));
}
/*****************************************************************************/
double gain::fv(double E)
{
  double arg=(E-qFlv)/(struktura::kB*T);
  return 1/(1 + exp(arg));
}
/*****************************************************************************/
double gain::przel_gest_z_cm2(double gest_w_cm2) // gestosc powierzchniowa
{
  return gest_w_cm2*1e-16*struktura::przelm*struktura::przelm;
}
/*****************************************************************************/
double gain::przel_gest_na_cm2(double gest_w_wew) // gestosc powierzchniowa
{
  return gest_w_wew/(1e-16*struktura::przelm*struktura::przelm);
}
