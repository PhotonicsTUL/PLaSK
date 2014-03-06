#include <iostream>

#include "gainQW.h"

using QW::nosnik;
using QW::gain;
using QW::parametry;
using namespace std;

const double gain::kB=1.38062/1.60219*1e-4;
const double gain::przelm=10*1.05459/(sqrt(1.60219*9.10956));
const double gain::przels=1.05459/1.60219*1e-3;
const double gain::ep0=8.8542*1.05459/(100*1.60219*sqrt(1.60219*9.10956));
const double gain::c=300*sqrt(9.10956/1.60219);
const double gain::exprng=11100;

nosnik::nosnik(): poziomy(NULL)
{
}
/*****************************************************************************/
double nosnik::Eodk(double k) // E(k)
{
  return k*k/(2*masa_w_plaszcz);
}
/*****************************************************************************/
double nosnik::En(double k,int n)
{
  return Eodk(k)+poziomy[n]+gleb;
}
/*****************************************************************************/
double nosnik::pozoddna(int i)
{
  return (poziomy[i]>0)?-1:poziomy[i]+gleb;
}
/*****************************************************************************/
int nosnik::ilepoz()
{
  int k=0;
  while(poziomy[k++]<=0);
  return k-1;
}
/*****************************************************************************/
nosnik::~nosnik()
{
  delete [] poziomy;
}
/*****************************************************************************/
parametry::~parametry()
{
  delete [] ldopar;
}
/*****************************************************************************/
gain::gain()
{
  Break = 0;
  ilwyw=0;
  ustawione='n';
  Twzmoc=NULL;
  Tspont.resize(2);
  Mt=-1;
  T=-1.;
  n_r=-1.;
  szer=-1.;
  szerb=-1.;
  szer_fal=1.;
  Eg=-1.;
  tau=-1.;
  konc=-1.;
  deltaSO=0;
  bladb=100;
}
/*****************************************************************************/
double gain::En_to_len(double en)
{
  return przel_dlug_na_angstr(2*M_PI*c/en);
}
/*****************************************************************************/
double gain::przel_dlug_z_angstr(double dl_w_A)
{
  return dl_w_A/przelm;
}
/*****************************************************************************/
double gain::przel_dlug_na_angstr(double dl_w_wew)
{
  return dl_w_wew*przelm;
}
/*****************************************************************************/
double gain::przel_czas_z_psek(double czas_w_sek)
{
  return czas_w_sek/przels;
}
/*****************************************************************************/
double gain::przel_czas_na_psek(double czas_w_wew)
{
  return czas_w_wew*przels;
}
/*****************************************************************************/
double gain::przel_konc_z_cm(double konc_w_cm)
{
  return konc_w_cm*1e-24*przelm*przelm*przelm;
}
/*****************************************************************************/
double gain::przel_konc_na_cm(double konc_w_wew)
{
  return konc_w_wew/(przelm*przelm*przelm)*1e24;
}
/*****************************************************************************/
double gain::element()                                                                /// funkcja liczaca element macierzowy
{
  return (1/el.masa_w_kier_prost - 1)*(Eg+deltaSO)*Eg/(Eg+2*deltaSO/3)/2;
}
/*****************************************************************************/
double gain::fc(double E)                                                            /// rozklad fermiego dla pasma przewodnictwa
{
  double arg=(E-Efc)/(kB*T);
  return (arg<exprng)?1/(1+exp(arg)):0;
}
/*****************************************************************************/
double gain::fv(double E)                                                            /// rozklad fermiego dla pasma walencyjnego
{
  double arg=(E-Efv)/(kB*T);
  return (arg<exprng)?1/(1+exp(arg)):0;
}
/*****************************************************************************/
double gain::metsiecz(double (gain::*wf)(double,double *),double xl,double xp,double * param,double prec) /// metoda siecznych
{
  if( ((this->*wf)(xl,param))*((this->*wf)(xp,param))>0)
    {
      //      std::cerr<<"\nZłe krace!\n";
      throw -1;
    }
  double x,y,yp,yl;
  x=xl;
  y=xp;
  xl=mniej(x,y);
  xp=-mniej(-x,-y);
  yl=(this->*wf)(xl,param);
  yp=(this->*wf)(xp,param);
  char pt=1;
  do
    {
      x=(xl*yp-xp*yl)/(yp-yl);
      y=(this->*wf)(x,param);
      if( y*yl>0 )
        {
          yl=y;
          xl=x;
          if(yl*(this->*wf)(xl+prec,param)<=0) pt=0;
        }
      else
        {
          yp=y;
          xp=x;
          if(yp*(this->*wf)(xp-prec,param)<=0) pt=0;
        }
    }while(pt);
  return x;
}
/*****************************************************************************/
double gain::gdziepoziomy(double e, double *param) /// zera daja poziomy, e - energia
{
  double v=param[0];
  double m1=param[1];
  double m2=param[2];
  double kI=sqrt(-2*m1*e); /// sqrt dla long double (standardowa - math)
  double kII=sqrt(2*m2*(e+v));
  double ilormas=m1/m2; // porawione warunki sklejania pochodnych
  return 2*kI*kII*ilormas*cos(szer*kII)+(kI*kI - kII*kII*ilormas*ilormas)*sin(szer*kII);
}
/*****************************************************************************/
double gain::gdziepoziomy2A(double e, double *param) /// 2 - podwójna studnia
{
  double v=param[0];
  double m1=param[1];
  double m2=param[2];
  double kI=sqrt(-2*m1*e);
  double kII=sqrt(2*m2*(e+v));
  return 2*kI*kII/(m1*m2)*cos(szer*kII)+(kI*kI/(m1*m1)-kII*kII/(m2*m2))*sin(szer*kII)-exp(-kI*szerb)*(kI*kI/(m1*m1)+kII*kII/(m2*m2))*sin(kII*szer);
}

/*****************************************************************************/
double gain::gdziepoziomy2B(double e, double *param) // ma zawsze 0 w 0
{
  double v=param[0];
  double m1=param[1];
  double m2=param[2];
  double kI=sqrt(-2*m1*e);
  double kII=sqrt(2*m2*(e+v));
  return 2*kI*kII/(m1*m2)*cos(szer*kII)+(kI*kI/(m1*m1)-kII*kII/(m2*m2))*sin(szer*kII)+exp(-kI*szerb)*(kI*kI/(m1*m1)+kII*kII/(m2*m2))*sin(kII*szer);
}
/*****************************************************************************/
double gain::krance(int n,double v,double m2) /// krance przedzialu w którym szuka sie n-tego poziomu
{
  return (n*M_PI/szer)*(n*M_PI/szer)/(2*m2)-v;
}
/*****************************************************************************/
double * gain::znajdzpoziomy(nosnik & no) /// przy pomocy gdziepoziomy znajduje poziomy
{
  double par[]={no.gleb,no.masabar,no.masa_w_kier_prost};
  double * wsk;
  if(no.masabar<=0 || no.gleb<=0 || no.masa_w_kier_prost<=0)
    {
      wsk=new double[1];
      wsk[0]=1;
    }
  else
    {
      int n=(int)ceil(szer*sqrt(2*no.masa_w_kier_prost*no.gleb)/M_PI);
      wsk=new double [n+1];
      if(!wsk)
        throw CriticalException("Error in gain module");
      double p,q;
      p=mniej(this->krance(1,no.gleb,no.masa_w_kier_prost),(double)0);
      double fp=this->gdziepoziomy(p,par);
      q=p;
      do
        {
          q=(q-no.gleb)/2;
        }while(this->gdziepoziomy(q,par)*fp>0);
      wsk[0]=this->metsiecz(& gain::gdziepoziomy,q,p,par);
      int i;
      for(i=1;i<=n-2;i++)
        wsk[i]=this->metsiecz(& gain::gdziepoziomy,this->krance(i,no.gleb,no.masa_w_kier_prost),this->krance(i+1,no.gleb,no.masa_w_kier_prost),par);
      wsk[n-1]=(n>1)?this->metsiecz(& gain::gdziepoziomy,this->krance(n-1,no.gleb,no.masa_w_kier_prost),0.0,par):wsk[0];
      wsk[n]=1;
    }
  return wsk;
}
/*****************************************************************************/
double * gain::znajdzpoziomy2(nosnik & no) /// j.w. dla podwójnej studni
{
  double przes=1e-7; // startowy punkt zamiast 0 i głębokości
  double par[]={no.gleb,no.masabar,no.masa_w_kier_prost};
  /*
  for(double E=0.;E>=-no.gleb;E-=.00005)
    {
      std::cerr<<E<<" "<<gdziepoziomy2A(E,par)<<" "<<gdziepoziomy2B(E,par)<<"\n";
    }
  */

  int n=(int)ceil(szer*sqrt(2*no.masa_w_kier_prost*no.gleb)/M_PI);
  //  std::cerr<<"\n n="<<n<<"\n";
  double * wsk=new double [2*n+1];
  if(!wsk)
    throw CriticalException("Error in gain module");
  double p,pom,ostkr;
  p=mniej(this->krance(1,no.gleb,no.masa_w_kier_prost),-przes);
  //  std::cerr<<"\n"<<this->gdziepoziomy2A(-.99*no.gleb,par)<<" "<<this->gdziepoziomy2A(p,par)<<"\n";
  if(n>1)
    {
      wsk[0]=this->metsiecz(& gain::gdziepoziomy2A,-no.gleb+przes,p,par); /// w podwójnej studni mamy rozszczepienie poziomów
      //      std::cerr<<"\n po pierwszym\n";
      wsk[1]=this->metsiecz(& gain::gdziepoziomy2B,-no.gleb+przes,p,par);
      if(wsk[0]>wsk[1])
        {
          pom=wsk[0];
          wsk[0]=wsk[1];
          wsk[1]=pom;
        }
      int i;
      for(i=1;i<=n-2;i++)
        {
          wsk[2*i]=this->metsiecz(& gain::gdziepoziomy2A,this->krance(i,no.gleb,no.masa_w_kier_prost),this->krance(i+1,no.gleb,no.masa_w_kier_prost),par);
          wsk[2*i+1]=this->metsiecz(& gain::gdziepoziomy2B,this->krance(i,no.gleb,no.masa_w_kier_prost),this->krance(i+1,no.gleb,no.masa_w_kier_prost),par);
          if(wsk[2*i]>wsk[2*i+1])
            {
              pom=wsk[2*i];
              wsk[2*i]=wsk[2*i+1];
              wsk[2*i+1]=pom;
            }
        }
      ostkr=this->krance(n-1,no.gleb,no.masa_w_kier_prost);
      //          std::cerr<<"\nA "<<this->gdziepoziomy2A(ostkr,par)<<" "<<this->gdziepoziomy2A(ostkr/2,par)<<" "<<this->gdziepoziomy2A(0.,par)<<"\n";
      if(gdziepoziomy2A(ostkr,par)*gdziepoziomy2A(0.0,par)<0)
        {
          wsk[2*n-2]=this->metsiecz(& gain::gdziepoziomy2A,ostkr,0.0,par);
          //                  std::cerr<<"\nOstatni A="<<wsk[2*n-2]<<"\n";
        }
      else wsk[2*n-2]=1;
      //    std::cerr<<"\nB "<<this->gdziepoziomy2B(ostkr,par)<<" "<<this->gdziepoziomy2B(ostkr/2,par)<<" "<<this->gdziepoziomy2B(-przes,par)<<"\n";
      if(gdziepoziomy2B(ostkr,par)*gdziepoziomy2B(-przes,par)<0)
        {
          wsk[2*n-1]=this->metsiecz(& gain::gdziepoziomy2B,ostkr,-przes,par);
          //  std::cerr<<"\nOstatni B="<<wsk[2*n-1]<<"\n";
        }
      else wsk[2*n-1]=1;
      if(wsk[2*n-2]>wsk[2*n-1])
        {
          pom=wsk[2*n-2];
          wsk[2*i]=wsk[2*n-1];
          wsk[2*n-1]=pom;
        }
    }
  else
    {
      if(gdziepoziomy2A(p,par)*gdziepoziomy2A(-no.gleb+przes,par)<0)
        {
          wsk[0]=this->metsiecz(& gain::gdziepoziomy2A,-no.gleb+przes,p,par);
        }
      else wsk[0]=1;
      if(gdziepoziomy2B(p,par)*gdziepoziomy2B(-no.gleb+przes,par)<0)
        {
          wsk[1]=this->metsiecz(& gain::gdziepoziomy2B,-no.gleb+przes,p,par);
        }
      else wsk[1]=1;
    }
    wsk[2*n]=1;
  return wsk;
}
/*****************************************************************************/
double gain::gdzieqflc(double ef,double *) /// zero wyznacza kwazi poziom fermiego w pasmie przewodnictwa
{
  double f=0;
  double en;
  double kT=kB*T;
  double gam32 = sqrt(M_PI)/2; // Gamma(3/2)
  double k;
  //  std::cerr<<"\nszer w gdzieqflc_n="<<szer<<"\n";
  f+=szer*kT*gam32*sqrt(kT)*2*sqrt(2*el.masabar)*el.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((ef-el.gleb-el.gleb_fal)/(kB*T)); // w sztukach na powierzchnię
  //    std::cerr<<"\n3D = "<<f<<" dla ef = "<<ef;
  if(el.gleb_fal>0)
    {
      for(int j=(int)ceil(szer_fal*sqrt(2*el.masabar*el.gleb_fal)/M_PI);j>=1;j--)
        {
          k = j*M_PI/(szer_fal);
          en=k*k/(2*el.masabar)+el.gleb;
          f+=szer/szer_fal*(el.masabar*kT)/M_PI*log(1+exp(-(en-ef)/(kB*T)));// spin jest
        }
    } // Poziomy nad studnią przybliżone studnią nieskończoną.
  //  f*=szer/szer_fal;  // stosunek objętości falowodu i studni
  //  std::clog<<"\nkocwbar = "<<f;
  barkonc_c=f/szer;
  for(int i=0;i<=el.ilepoz()-1;i++)
    f+=(el.masa_w_plaszcz*kB*T)/M_PI*log(1+exp(-(el.pozoddna(i)-ef)/(kB*T)));
  //  std::clog<<"\nkoccalk = "<<f;
  f-=konc*szer;
  return f;
}
/*****************************************************************************/
double gain::gdzieqflc2(double ef,double *) /// j.w. dla podwójnej studni
{
  double f=0;
  double en;
  for(int j=(int)ceil(2*szer_fal*sqrt(2*el.masa_w_plaszcz*el.gleb_fal)/M_PI);j>=1;j--)
    {
      en=(j*M_PI/(2*szer_fal))*(j*M_PI/(2*szer_fal))/(2*el.masa_w_plaszcz)+el.gleb;
      f+=log(1+exp(-(en-ef)/(kB*T)));
    } // Poziomy nad studnią przybliżone studnią nieskończoną.
  f*=2*szer/szer_fal;  // stosunek objętości falowodu i studni
  for(int i=0;i<=el.ilepoz()-1;i++)
    f+=log(1+exp(-(el.pozoddna(i)-ef)/(kB*T)));
  f-=konc*M_PI*2*szer/(el.masa_w_plaszcz*kB*T);
  return f;
}
/*****************************************************************************/
double gain::gdzieqflc_n(double ef,double * wsk_sszer) /// dla n studni - poziomy podane z zewnatrz
{
  double f=0;
  double en;
  double kT=kB*T;
  double sumaszer=(*wsk_sszer);
  //  std::cerr<<"\nsumaszer w gdzieqflc_n="<<sumaszer<<"\n";
  double gam32 = sqrt(M_PI)/2; // Gamma(3/2)
   double k;
  f+=sumaszer*kT*gam32*sqrt(kT)*2*sqrt(2*el.masabar)*el.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((ef-el.gleb-el.gleb_fal)/(kB*T)); // w sztukach na powierzchnię
  //  std::cerr<<"\n3D_n = "<<f<<" dla ef = "<<ef;
  for(int j=(int)ceil(szer_fal*sqrt(2*el.masabar*el.gleb_fal)/M_PI);j>=1;j--)
    {
      k = j*M_PI/(szer_fal);
      en=k*k/(2*el.masabar)+el.gleb;
      //      en=(j*M_PI/(2*szer_fal))*(j*M_PI/(2*szer_fal))/(2*el.masabar)+el.gleb;
      f+=sumaszer/szer_fal*(el.masabar*kT)/M_PI*log(1+exp(-(en-ef)/(kB*T))); // spin jest
    } // Poziomy nad studnią przybliżone studnią nieskończoną.
  //  std::clog<<"\nkocwbar = "<<f;
  barkonc_c=f/sumaszer;
  for(int i=0;i<=el.ilepoz()-1;i++)
    f+=(el.masa_w_plaszcz*kB*T)/M_PI*log(1+exp(-(el.pozoddna(i)-ef)/(kB*T)));
  f-=konc*sumaszer;
  return f;
}
/*****************************************************************************/
double gain::gdzieqflv(double ef,double *) /// zero wyznacza kwazi poziom fermiego w pasmie walencyjnym
{
  double f=0;
  double en;
  double kT=kB*T;
  double gam32 = sqrt(M_PI)/2; // Gamma(3/2)
  double k;
  f+=szer*kT*gam32*sqrt(kT)*2*sqrt(2*lh.masabar)*lh.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((-ef-lh.gleb-lh.gleb_fal)/(kB*T));
  if(lh.gleb_fal>0)
    {
      for(int j=(int)ceil(szer_fal*sqrt(2*lh.masabar*lh.gleb_fal)/M_PI);j>=1;j--)
        {
          k = j*M_PI/(szer_fal);
          en=k*k/(2*lh.masabar)+lh.gleb;
          f+=szer/szer_fal*lh.masabar*kT/M_PI*log(1+exp((-en-ef)/(kB*T)));
        }
    }
  //  f*=szer/szer_fal;  // stosunek objętości falowodu i studni
  f+=szer*gam32*kT*sqrt(kT)*2*sqrt(2*hh.masabar)*hh.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((-ef-hh.gleb-hh.gleb_fal)/(kB*T));
  if(hh.gleb_fal>0)
    {
      for(int j=(int)ceil(szer_fal*sqrt(2*hh.masabar*hh.gleb_fal)/M_PI);j>=1;j--)
        {
          k = j*M_PI/(szer_fal);
          en= k*k/(2*hh.masabar)+hh.gleb;
          f+=szer/szer_fal*hh.masabar*kT/M_PI*log(1+exp((-en-ef)/(kB*T)));
        } // Poziomy nad studnią przybliżone studnią nieskończoną.
    }
  barkonc_v=f/szer;
  //  std::clog<<"\nkocvwbar = "<<f;
  for(int i=0;i<=hh.ilepoz()-1;i++)
    f+=hh.masa_w_plaszcz*kB*T/M_PI*log(1+exp((-hh.pozoddna(i)-ef)/(kB*T)));
  for(int j=0;j<=lh.ilepoz()-1;j++)
    f+=lh.masa_w_plaszcz*kB*T/M_PI*log(1+exp((-lh.pozoddna(j)-ef)/(kB*T)));
  //  std::clog<<"\nkocvcalk = "<<f;
  f-=konc*szer;
  return f;
}
/*****************************************************************************
double gain::gdzieqflv2(double ef,double *)
{
  double f=0;
  double en;
  for(int j=(int)ceil(2*szer_fal*sqrt(2*lh.masa_w_plaszcz*lh.gleb_fal)/M_PI);j>=1;j--)
    {
      en=(j*M_PI/(2*szer_fal))*(j*M_PI/(2*szer_fal))/(2*lh.masa_w_plaszcz)+lh.gleb;
      f+=log(1+exp((-en-ef)/(kB*T)));
    }
  f*=lh.masa_w_plaszcz;
  f*=2*szer/szer_fal;  // stosunek objętości falowodu i studni
  for(int j=(int)ceil(2*szer_fal*sqrt(2*hh.masa_w_plaszcz*hh.gleb_fal)/M_PI);j>=1;j--)
    {
      en=(j*M_PI/(2*szer_fal))*(j*M_PI/(2*szer_fal))/(2*hh.masa_w_plaszcz)+hh.gleb;
      f+=hh.masa_w_plaszcz*log(1+exp((-en-ef)/(kB*T)))*2*szer/szer_fal;
    } // Poziomy nad studnią przybliżone studnią nieskończoną.
  for(int i=0;i<=hh.ilepoz()-1;i++)
    f+=hh.masa_w_plaszcz*log(1+exp((-hh.pozoddna(i)-ef)/(kB*T)));
  for(int j=0;j<=lh.ilepoz()-1;j++)
    f+=lh.masa_w_plaszcz*log(1+exp((-lh.pozoddna(j)-ef)/(kB*T)));
  f-=konc*M_PI*2*szer/(kB*T);
  return f;
}
*****************************************************************************/
double gain::gdzieqflv_n(double ef,double * wsk_sszer)
{
  double f=0;
  double en;
  double sumaszer=(*wsk_sszer);
  double kT=kB*T;
  double gam32 = sqrt(M_PI)/2; // Gamma(3/2)
  double k;
  f+=sumaszer*kT*gam32*sqrt(kT)*2*sqrt(2*lh.masabar)*lh.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((-ef-lh.gleb-lh.gleb_fal)/(kB*T));
  for(int j=(int)ceil(szer_fal*sqrt(2*lh.masabar*lh.gleb_fal)/M_PI);j>=1;j--)
    {
      k = j*M_PI/(szer_fal);
      en=k*k/(2*lh.masabar)+lh.gleb;
      f+=sumaszer/szer_fal*lh.masabar*kT/M_PI*log(1+exp((-en-ef)/(kB*T)));
    }
  f+=sumaszer*gam32*kT*sqrt(kT)*2*sqrt(2*hh.masabar)*hh.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((-ef-hh.gleb-hh.gleb_fal)/(kB*T));
  for(int j=(int)ceil(szer_fal*sqrt(2*hh.masabar*hh.gleb_fal)/M_PI);j>=1;j--)
    {
      k = j*M_PI/(szer_fal);
      en= k*k/(2*hh.masabar)+hh.gleb;
      f+=sumaszer/szer_fal*hh.masabar*kT/M_PI*log(1+exp((-en-ef)/(kB*T)));
    } // Poziomy nad studnią przybliżone studnią nieskończoną.
  barkonc_v=f/sumaszer;
  for(int i=0;i<=hh.ilepoz()-1;i++)
    f+=hh.masa_w_plaszcz*kB*T/M_PI*log(1+exp((-hh.pozoddna(i)-ef)/(kB*T)));
  for(int j=0;j<=lh.ilepoz()-1;j++)
    f+=lh.masa_w_plaszcz*kB*T/M_PI*log(1+exp((-lh.pozoddna(j)-ef)/(kB*T)));
  //  std::clog<<"\nkocvcalk = "<<f;
  f-=konc*sumaszer;
  return f;
}
/*****************************************************************************/
double gain::qFlc() /// poziomy na podstawie gdzieqflc
{
  double e1=-el.gleb/10;
  double k=el.gleb/100;
  double stare=e1;

  while(gdzieqflc(e1,NULL)>0)
    {
      stare=e1;
      e1-=k;
    }
  double e2=stare;
  while(gdzieqflc(e2,NULL)<0)
    {
      e2+=k;
    }
  double wyn=metsiecz(& gain::gdzieqflc,e1,e2);
  return wyn;
}
/*****************************************************************************/
double gain::qFlc2()
{
  double e1=-el.gleb/10;
  double k=el.gleb/100;
  double stare=e1;
  while(gdzieqflc2(e1,NULL)>0)
    {
      stare=e1;
      e1-=k;
    }
  double e2=stare;
  while(gdzieqflc2(e2,NULL)<0)
    {
      e2+=k;
    }
  double wyn=metsiecz(& gain::gdzieqflc2,e1,e2);
  return wyn;
}
/*****************************************************************************/
double gain::qFlc_n(double sszer)
{
  double e1=-el.gleb/10;
  double k=el.gleb/100;
  double stare=e1;
  while(gdzieqflc_n(e1,&sszer)>0)
    {
      stare=e1;
      e1-=k;
    }
  double e2=stare;
  while(gdzieqflc_n(e2,&sszer)<0)
    {
      e2+=k;
    }
  double wyn=metsiecz(& gain::gdzieqflc_n,e1,e2,&sszer);
  return wyn;
}
/*****************************************************************************/
double gain::qFlv()
{
  double e1=el.gleb/10;
  double k=el.gleb/10;
  double stare=e1;
  while(gdzieqflv(e1,NULL)>0)
    {
      stare=e1;
      e1+=k;
    }
  double e2=stare;
  while(gdzieqflv(e2,NULL)<0)
    {
      e2-=k;
    }
  double wyn=metsiecz(& gain::gdzieqflv,e1,e2);
  return wyn;
}
/*****************************************************************************
double gain::qFlv2()
{
  double e1=el.gleb/10;
  double k=el.gleb/10;
  double stare=e1;
  while(gdzieqflv2(e1,NULL)>0)
    {
      stare=e1;
      e1+=k;
    }
  double e2=stare;
  while(gdzieqflv2(e2,NULL)<0)
    {
      e2-=k;
    }
  double wyn=metsiecz(& gain::gdzieqflv2,e1,e2);
  return wyn;
}
*****************************************************************************/
double gain::qFlv_n(double sszer)
{
  double e1=el.gleb/10;
  double k=el.gleb/10;
  double stare=e1;
  while(gdzieqflv_n(e1,&sszer)>0)
    {
      stare=e1;
      e1+=k;
    }
  double e2=stare;
  while(gdzieqflv_n(e2,&sszer)<0)
    {
      e2-=k;
    }
  double wyn=metsiecz(& gain::gdzieqflv_n,e1,e2,&sszer);
  return wyn;
}
/*****************************************************************************/
inline double gain::L(double x,double b) /// poszerzenie lorentzowskie
{
  return b/(M_PI*(x*x+b*b));
}
/*****************************************************************************/
inline double gain::Lpr(double x,double b) /// pochodna poszerzenia lorentzowskiego
{
  return -2*x*b/(M_PI*(x*x+b*b)*(x*x+b*b));
}
/*****************************************************************************/
double gain::kodE(double E,double mc,double mv) /// k(E) (k nosnika od energii fotonu)
{
  double m=1/(1/mc+1/mv);
  return sqrt(2*m*E);
}
/*****************************************************************************/
double gain::rored(double,double mc,double mv) /// dwuwymiarowa zredukowana gestosc stanów
{
  double m=1/(1/mc+1/mv);
  return m/(2*M_PI*szer);
}
/*****************************************************************************/
double gain::rored2(double,double mc,double mv)
{
  double m=1/(1/mc+1/mv);
  return m/(4*M_PI*szer);
}
/*****************************************************************************/
double gain::rored_n(double,double mc,double mv, double sumaszer)
{
  double m=1/(1/mc+1/mv);
  return m/(2*M_PI*sumaszer);
}
/*****************************************************************************/
double gain::dosplotu(double E, parametry * param) /// splot lorentza ze wzmonieniem (nie poszerzonym) - funkcja podcalkowa do splotu dla emisji wymuszonej
{
  double *par=param->ldopar;
  double E0=par[0];
  int i=(int)par[3];
  double b=par[1];
  double t=par[2];

  double h_masa_w_plaszcz=(param->rdziury=='h')?hh.masa_w_plaszcz:lh.masa_w_plaszcz;
  double k=kodE(E-E0,el.masa_w_plaszcz,h_masa_w_plaszcz);
  double el_En=el.En(k,i);
  double h_En=(param->rdziury=='h')?hh.En(k,i):lh.En(k,i);
  double cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
  double wspelema=(param->rdziury=='h')?(1+cos2tet)/2:(5-3*cos2tet)/6; // modyfikacja elementu macierzowego w zal. od k
  double f=wspelema*rored(k,el.masa_w_plaszcz,h_masa_w_plaszcz)*( fc(el_En)-fv(-h_En) )/E;
  return f*L(E-t,b);
}
/*****************************************************************************/
double gain::dosplotu2(double E, parametry * param)
{
  double *par=param->ldopar;
  double E0=par[0];
  int i=(int)par[3];
  double b=par[1];
  double t=par[2];

  double h_masa_w_plaszcz=(param->rdziury=='h')?hh.masa_w_plaszcz:lh.masa_w_plaszcz;
  double k=kodE(E-E0,el.masa_w_plaszcz,h_masa_w_plaszcz);
  double el_En=el.En(k,i);
  double h_En=(param->rdziury=='h')?hh.En(k,i):lh.En(k,i);
  double cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
  double wspelema=(param->rdziury=='h')?(1+cos2tet)/2:(5-3*cos2tet)/6; // modyfikacja elementu macierzowego w zal. od k
  double f=wspelema*rored2(k,el.masa_w_plaszcz,h_masa_w_plaszcz)*( fc(el_En)-fv(-h_En) )/E;
  return f*L(E-t,b);
}
/*****************************************************************************/
double gain::dosplotu_n(double E, parametry * param)
{
  double *par=param->ldopar;
  double E0=par[0];
  int i=(int)par[3];
  double b=par[1];
  double t=par[2];
  double sumszer=par[4];

  double h_masa_w_plaszcz=(param->rdziury=='h')?hh.masa_w_plaszcz:lh.masa_w_plaszcz;
  double k=kodE(E-E0,el.masa_w_plaszcz,h_masa_w_plaszcz);
  double el_En=el.En(k,i);
  double h_En=(param->rdziury=='h')?hh.En(k,i):lh.En(k,i);
  double cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
  double wspelema=(param->rdziury=='h')?(1+cos2tet)/2:(5-3*cos2tet)/6; // modyfikacja elementu macierzowego w zal. od k
  double f=wspelema*rored_n(k,el.masa_w_plaszcz,h_masa_w_plaszcz,sumszer)*( fc(el_En)-fv(-h_En) )/E;
  return f*L(E-t,b);
}
/*****************************************************************************/
double gain::wzmoc_z_posz(double t) /// wykonuje calke (splot) z funkcja dosplotu
{
  int i=0;
  double Ec,Ev;
  double g=0;
  Ev=hh.pozoddna(0);
  Ec=el.pozoddna(0);
  double E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  double stala=M_PI/(c*n_r*ep0)/przelm*1e8;
  double epsb;
  double * ldpar=new double [4];
  parametry * param=new parametry;
  param->ldopar=ldpar;
  param->rdziury='h';
  double b=1/tau;
  ldpar[1]=b;
  ldpar[2]=t;
  double lc=1/(1+el.masa_w_plaszcz/hh.masa_w_plaszcz);
  double lv=1/(1+hh.masa_w_plaszcz/el.masa_w_plaszcz);
  double M=(1/Eg*( 2/(Eg*Eg) + 2/(Eg*kB*T)*(lc+lv) + (lc*lc+lv*lv)/(kB*T*kB*T)))/(b*M_PI);
  M+=.75*sqrt(3.)/(M_PI*b*b*Eg)*(1/Eg+lc/(kB*T)+lv/(kB*T));
  M+=2/(Eg*b*b*b*M_PI);
  epsb=bladb/(stala*3*Mt*el.ilepoz()/2);
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(double)i;
      if(E0<t+32*b)
        g+=Mt*Prost(& gain::dosplotu,M,-mniej(-E0,-t+32*b),t+32*b,param,epsb);
      i++;
      Ec=el.pozoddna(i);
      Ev=hh.pozoddna(i);
      E0=Eg+Ec+Ev;
    }
  Ev=lh.pozoddna(0);
  Ec=el.pozoddna(0);
  E0=Eg+el.pozoddna(0)+lh.pozoddna(0);
  i=0;
  param->rdziury='l';
  epsb=bladb/(stala*Mt*el.ilepoz());
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(double)i;
      if(E0<t+32*b)
        g+=Mt*Prost(& gain::dosplotu,M,-mniej(-E0,-t+32*b),t+32*b,param,epsb);
      i++;
      Ec=el.pozoddna(i);
      Ev=lh.pozoddna(i);
      E0=Eg+Ec+Ev;
    }
  delete param;
  return stala*g;
}
/*****************************************************************************/
double gain::wzmoc_z_posz2(double t)
{
  int i=0;
  double Ec,Ev;
  double g=0;
  Ev=hh.pozoddna(0);
  Ec=el.pozoddna(0);
  double E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  double stala=M_PI/(c*n_r*ep0)/przelm*1e8;
  double epsb;
  double * ldpar=new double [4];
  parametry * param=new parametry;
  param->ldopar=ldpar;
  param->rdziury='h';
  double b=1/tau;
  ldpar[1]=b;
  ldpar[2]=t;
  double lc=1/(1+el.masa_w_plaszcz/hh.masa_w_plaszcz);
  double lv=1/(1+hh.masa_w_plaszcz/el.masa_w_plaszcz);
  double M=(1/Eg*( 2/(Eg*Eg) + 2/(Eg*kB*T)*(lc+lv) + (lc*lc+lv*lv)/(kB*T*kB*T)))/(b*M_PI);
  M+=.75*sqrt(3.)/(M_PI*b*b*Eg)*(1/Eg+lc/(kB*T)+lv/(kB*T));
  M+=2/(Eg*b*b*b*M_PI);
  epsb=bladb/(stala*3*Mt*el.ilepoz()/2);
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(double)i;
      if(E0<t+32*b)
        g+=Mt*Prost(& gain::dosplotu2,M,-mniej(-E0,-t+32*b),t+32*b,param,epsb);
      i++;
      Ec=el.pozoddna(i);
      Ev=hh.pozoddna(i);
      E0=Eg+Ec+Ev;
    }
  Ev=lh.pozoddna(0);
  Ec=el.pozoddna(0);
  E0=Eg+el.pozoddna(0)+lh.pozoddna(0);
  i=0;
  param->rdziury='l';
  epsb=bladb/(stala*Mt*el.ilepoz());
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(double)i;
      if(E0<t+32*b)
        g+=Mt*Prost(& gain::dosplotu2,M,-mniej(-E0,-t+32*b),t+32*b,param,epsb);
      i++;
      Ec=el.pozoddna(i);
      Ev=lh.pozoddna(i);
      E0=Eg+Ec+Ev;
    }
  delete param;
  return stala*g;
}
/*****************************************************************************/
double gain::wzmoc_z_posz_n(double t, double sumszer)
{
  int i=0;
  double Ec,Ev;
  double g=0;
  Ev=hh.pozoddna(0);
  Ec=el.pozoddna(0);
  double E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  double stala=M_PI/(c*n_r*ep0)/przelm*1e8;
  double epsb;
  double * ldpar=new double [5];
  parametry * param=new parametry;
  param->ldopar=ldpar;
  param->rdziury='h';
  double b=1/tau;
  ldpar[1]=b;
  ldpar[2]=t;
  ldpar[4]=sumszer;

  double lc=1/(1+el.masa_w_plaszcz/hh.masa_w_plaszcz);
  double lv=1/(1+hh.masa_w_plaszcz/el.masa_w_plaszcz);
  double M=(1/Eg*( 2/(Eg*Eg) + 2/(Eg*kB*T)*(lc+lv) + (lc*lc+lv*lv)/(kB*T*kB*T)))/(b*M_PI);
  M+=.75*sqrt(3.)/(M_PI*b*b*Eg)*(1/Eg+lc/(kB*T)+lv/(kB*T));
  M+=2/(Eg*b*b*b*M_PI);
  epsb=bladb/(stala*3*Mt*el.ilepoz()/2);
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(double)i;
      if(E0<t+32*b)
        g+=Mt*Prost(& gain::dosplotu_n,M,-mniej(-E0,-t+32*b),t+32*b,param,epsb);
      i++;
      Ec=el.pozoddna(i);
      Ev=hh.pozoddna(i);
      E0=Eg+Ec+Ev;
    }
  Ev=lh.pozoddna(0);
  Ec=el.pozoddna(0);
  E0=Eg+el.pozoddna(0)+lh.pozoddna(0);
  i=0;
  param->rdziury='l';
  epsb=bladb/(stala*Mt*el.ilepoz());
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(double)i;
      if(E0<t+32*b)
        g+=Mt*Prost(& gain::dosplotu_n,M,-mniej(-E0,-t+32*b),t+32*b,param,epsb);
      i++;
      Ec=el.pozoddna(i);
      Ev=lh.pozoddna(i);
      E0=Eg+Ec+Ev;
    }
  delete param;
  return stala*g;
}
/*****************************************************************************/
double gain::dosplotu_spont(double E, parametry * param) /// funkcja podcalkowa do splotu dla emisji spont.
{
  double *par=param->ldopar;
  double E0=par[0];
  int i=(int)par[3];
  double b=par[1];
  double t=par[2];

  double h_masa_w_plaszcz=(param->rdziury=='h')?hh.masa_w_plaszcz:lh.masa_w_plaszcz;
  double k=kodE(E-E0,el.masa_w_plaszcz,h_masa_w_plaszcz);
  double el_En=el.En(k,i);
  double h_En=(param->rdziury=='h')?hh.En(k,i):lh.En(k,i);
  /*
  double cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
  double wspelema=(param->rdziury=='h')?(1+cos2tet)/2:(5-3*cos2tet)/6; // modyfikacja elementu macierzowego w zal. od k
  */
  //  double f=wspelema*rored(k,el.masa_w_plaszcz,h_masa_w_plaszcz)*( fc(el_En)*(1-fv(-h_En)) )/E;
  double f=E*E*rored(k,el.masa_w_plaszcz,h_masa_w_plaszcz)*( fc(el_En)*(1-fv(-h_En)) );
  return f*L(E-t,b);
}
/*****************************************************************************/
double gain::spont_z_posz(double t) /// to samo co wzmoc_posz tylko, ze spontaniczne
{
  int i=0;
  double Ec,Ev;
  double g=0;
  Ev=hh.pozoddna(0);
  Ec=el.pozoddna(0);
  double E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  double stala=n_r/(M_PI*c*c*c*ep0); // Nie ma przelicznikow, bo Get przeliczy na zwykle jednostki
  double epsb;
  double * ldpar=new double [4];
  parametry * param=new parametry;
  param->ldopar=ldpar;
  param->rdziury='h';
  double b=1/tau;
  ldpar[1]=b;
  ldpar[2]=t;
  double lc=1/(1+el.masa_w_plaszcz/hh.masa_w_plaszcz);
  double lv=1/(1+hh.masa_w_plaszcz/el.masa_w_plaszcz);
  double M=(1/Eg*( 2/(Eg*Eg) + 2/(Eg*kB*T)*(lc+lv) + (lc*lc+lv*lv)/(kB*T*kB*T)))/(b*M_PI); //Oszacowanie pochodnej (chyba)
  M+=.75*sqrt(3.)/(M_PI*b*b*Eg)*(1/Eg+lc/(kB*T)+lv/(kB*T));
  M+=2/(Eg*b*b*b*M_PI);
  epsb=bladb/(stala*3*Mt*el.ilepoz()/2);
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(double)i;
      if(E0<t+32*b)
        g+=Mt*Prost(& gain::dosplotu_spont,M,-mniej(-E0,-t+32*b),t+32*b,param,epsb);
      i++;
      Ec=el.pozoddna(i);
      Ev=hh.pozoddna(i);
      E0=Eg+Ec+Ev;
    }
  Ev=lh.pozoddna(0);
  Ec=el.pozoddna(0);
  E0=Eg+el.pozoddna(0)+lh.pozoddna(0);
  i=0;
  param->rdziury='l';
  epsb=bladb/(stala*Mt*el.ilepoz());
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(double)i;
      if(E0<t+32*b)
        g+=Mt*Prost(& gain::dosplotu_spont,M,-mniej(-E0,-t+32*b),t+32*b,param,epsb);
      i++;
      Ec=el.pozoddna(i);
      Ev=lh.pozoddna(i);
      E0=Eg+Ec+Ev;
    }
  delete param;
  return stala*g;
}
/*****************************************************************************/
double gain::Prost(double (gain::*F)(double, parametry *),double M, double a, double b, parametry * par, double bld) /// metoda prostokatów (calkowania)
{
  double szer=b-a;
  long N=(long)ceil(szer*sqrt(szer*M/(24*bld)));
  double podz=szer/N;
  double wyn=0;
  for(long k=0;k<=N-1;k++)
    {
      wyn+=(this->*F)(a+podz*((double)k+.5),par);
    }
  return podz*wyn;
}
/*****************************************************************************/
double gain::wzmoc0(double E) /// liczy wzmocnienie bez poszerzenia
{
  int i=0;
  double Ec,Ev;
  double g=0;
  double E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  double k;
  double cos2tet;
  while(E0<=E)
    {
      k=kodE(E-E0,el.masa_w_plaszcz,hh.masa_w_plaszcz);
      cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
      g+=(1+cos2tet)/2*Mt*rored(k,el.masa_w_plaszcz,hh.masa_w_plaszcz)*( fc(el.En(k,i))-fv(-hh.En(k,i)) );
      i++;
      Ec=el.pozoddna(i);
      Ev=hh.pozoddna(i);
      if(Ec<0||Ev<0) break;
      E0=Eg+Ec+Ev;
    }
  E0=Eg+el.pozoddna(0)+lh.pozoddna(0);
  i=0;
  while(E0<=E)
    {
      k=kodE(E-E0,el.masa_w_plaszcz,lh.masa_w_plaszcz);
      cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
      g+=(5-3*cos2tet)/6*Mt*rored(k,el.masa_w_plaszcz,lh.masa_w_plaszcz)*(fc(el.En(k,i))-fv(-lh.En(k,i)));
      i++;
      Ec=el.pozoddna(i);
      Ev=lh.pozoddna(i);
      if(Ec<0||Ev<0) break;
      E0=Eg+Ec+Ev;
    }
  return M_PI*g/(c*n_r*ep0*E)/przelm*1e8;
}
/*****************************************************************************/
double gain::wzmoc02(double E)
{
  int i=0;
  double Ec,Ev;
  double g=0;
  double E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  double k;
  double cos2tet;
  while(E0<=E)
    {
      k=kodE(E-E0,el.masa_w_plaszcz,hh.masa_w_plaszcz);
      cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
      g+=(1+cos2tet)/2*Mt*rored2(k,el.masa_w_plaszcz,hh.masa_w_plaszcz)*( fc(el.En(k,i))-fv(-hh.En(k,i)) );
      i++;
      Ec=el.pozoddna(i);
      Ev=hh.pozoddna(i);
      if(Ec<0||Ev<0) break;
      E0=Eg+Ec+Ev;
    }
  E0=Eg+el.pozoddna(0)+lh.pozoddna(0);
  i=0;
  while(E0<=E)
    {
      k=kodE(E-E0,el.masa_w_plaszcz,lh.masa_w_plaszcz);
      cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
      g+=(5-3*cos2tet)/6*Mt*rored2(k,el.masa_w_plaszcz,lh.masa_w_plaszcz)*(fc(el.En(k,i))-fv(-lh.En(k,i)));
      i++;
      Ec=el.pozoddna(i);
      Ev=lh.pozoddna(i);
      if(Ec<0||Ev<0) break;
      E0=Eg+Ec+Ev;
    }
  return M_PI*g/(c*n_r*ep0*E)/przelm*1e8;
}
/*****************************************************************************/
double gain::wzmoc0_n(double E, double sumszer)
{
  int i=0;
  double Ec,Ev;
  double g=0;
  double E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  double k;
  double cos2tet;
  while(E0<=E)
    {
      k=kodE(E-E0,el.masa_w_plaszcz,hh.masa_w_plaszcz);
      cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
      g+=(1+cos2tet)/2*Mt*rored_n(k,el.masa_w_plaszcz,hh.masa_w_plaszcz,sumszer)*( fc(el.En(k,i))-fv(-hh.En(k,i)) );
      i++;
      Ec=el.pozoddna(i);
      Ev=hh.pozoddna(i);
      if(Ec<0||Ev<0) break;
      E0=Eg+Ec+Ev;
    }
  E0=Eg+el.pozoddna(0)+lh.pozoddna(0);
  i=0;
  while(E0<=E)
    {
      k=kodE(E-E0,el.masa_w_plaszcz,lh.masa_w_plaszcz);
      cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
      g+=(5-3*cos2tet)/6*Mt*rored_n(k,el.masa_w_plaszcz,lh.masa_w_plaszcz,sumszer)*(fc(el.En(k,i))-fv(-lh.En(k,i)));
      i++;
      Ec=el.pozoddna(i);
      Ev=lh.pozoddna(i);
      if(Ec<0||Ev<0) break;
      E0=Eg+Ec+Ev;
    }
  return M_PI*g/(c*n_r*ep0*E)/przelm*1e8;
}
/*****************************************************************************/
double gain::spont0(double E) /// liczy emisje spont. bez poszerzenia
{
  int i=0;
  double Ec,Ev;
  double g=0;
  double E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  double k;
  double cos2tet;
  while(E0<=E)
    {
      k=kodE(E-E0,el.masa_w_plaszcz,hh.masa_w_plaszcz);
      cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
      g+=(1+cos2tet)/2*Mt*rored(k,el.masa_w_plaszcz,hh.masa_w_plaszcz)*( fc(el.En(k,i))*(1-fv(-hh.En(k,i))) );
      i++;
      Ec=el.pozoddna(i);
      Ev=hh.pozoddna(i);
      if(Ec<0||Ev<0) break;
      E0=Eg+Ec+Ev;
    }
  E0=Eg+el.pozoddna(0)+lh.pozoddna(0);
  i=0;
  while(E0<=E)
    {
      k=kodE(E-E0,el.masa_w_plaszcz,lh.masa_w_plaszcz);
      cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
      g+=(5-3*cos2tet)/6*Mt*rored(k,el.masa_w_plaszcz,lh.masa_w_plaszcz)*(fc(el.En(k,i))*(1-fv(-lh.En(k,i))));
      i++;
      Ec=el.pozoddna(i);
      Ev=lh.pozoddna(i);
      if(Ec<0||Ev<0) break;
      E0=Eg+Ec+Ev;
    }
  return n_r*E*E*g/(M_PI*c*c*c*ep0);
}
/*****************************************************************************/
void gain::przygobl() /// przygotuj obliczenia (znajduje kp fermiego, poziomy energetyczne w studni etc)
{
  if(Mt<=0)
    {
      Mt=element();
    }
  if(T<0 || n_r<0 || szer<0 || szer_fal<0 || Eg<0 || Mt<0 || tau<0 || konc<0)
    {
      throw CriticalException("Error in gain module");
    }
  el.~nosnik();
  el.poziomy=znajdzpoziomy(el);
  hh.~nosnik();
  hh.poziomy=znajdzpoziomy(hh);
  lh.~nosnik();
  lh.poziomy=znajdzpoziomy(lh);
  kasuj_poziomy = true;
  Efc=qFlc();
  Efv=qFlv();
  /*  std::cerr<<"\nszer="<<szer<<"\n";
  std::cerr<<"\nqflc1="<<Efc<<"\n";
  std::cerr<<"\nqflv1="<<Efv<<"\n";*/
  ustawione='t';
}
/*****************************************************************************/
void gain::przygobl_n(double sumaszer)
{
  //  std::cerr<<"\nW n\n";
  if(Mt<=0) Mt=element();
  if(T<0 || n_r<0 || szer<0 || szer_fal<0 || Eg<0 || Mt<0 || tau<0 || konc<0)
    throw CriticalException("Error in gain module");
  if (kasuj_poziomy) el.~nosnik();
  el.poziomy = znajdzpoziomy(el);
  if (kasuj_poziomy) hh.~nosnik();
  hh.poziomy = znajdzpoziomy(hh);
  if (kasuj_poziomy) lh.~nosnik();
  lh.poziomy = znajdzpoziomy(lh);
  Efc=qFlc_n(sumaszer);
  Efv=qFlv_n(sumaszer);
  ustawione='t';
}
/*****************************************************************************
void gain::przygobl2()
{
  //  std::cerr<<"\nW 2\n";
  if(Mt<=0)
    {
      Mt=element();
    }
  if(T<0 || n_r<0 || szer<0 || szer_fal<0 || Eg<0 || Mt<0 || tau<0 || konc<0)
    {
      throw CriticalException("Error in gain module");
    }
  if (kasuj_poziomy) el.~nosnik();
  el.poziomy=znajdzpoziomy2(el);
  //  std::cerr<<"\nel2 poziomy "<<el.ilepoz()<<"\n";
  if (kasuj_poziomy) hh.~nosnik();
  hh.poziomy=znajdzpoziomy2(hh);
  //  std::cerr<<"\nhh2 poziomy "<<hh.ilepoz()<<"\n";
  if (kasuj_poziomy) lh.~nosnik();
  lh.poziomy=znajdzpoziomy2(lh);
  //  std::cerr<<"\nlh2 poziomy "<<lh.ilepoz()<<"\n";
  kasuj_poziomy = true;
  Efc=qFlc2();
  Efv=qFlv2();
  ustawione='t';
}
*****************************************************************************/
void gain::przygobl_n(const ExternalLevels& zewpoziomy, double sumaszer)
{
  //  std::cerr<<"\nW n\n";
  if(Mt<=0)
    {
      Mt=element();
    }
  if(T<0 || n_r<0 || szer<0 || szer_fal<0 || Eg<0 || Mt<0 || tau<0 || konc<0)
    {
      throw CriticalException("Error in gain module");
    }
  if (kasuj_poziomy) el.~nosnik();
  el.poziomy = zewpoziomy.el;
  //  std::cerr<<"\neln poziomy "<<el.ilepoz()<<"\n";
  if (kasuj_poziomy) hh.~nosnik();
  hh.poziomy = zewpoziomy.hh;
  //  std::cerr<<"\nhhn poziomy "<<hh.ilepoz()<<"\n";
  if (kasuj_poziomy) lh.~nosnik();
  lh.poziomy = zewpoziomy.lh;
  kasuj_poziomy = false;
  //  std::cerr<<"\nlhn poziomy "<<lh.ilepoz()<<"\n";
  Efc=qFlc_n(sumaszer);
  Efv=qFlv_n(sumaszer);
  /*
  std::cerr<<"\nsumaszer="<<sumaszer<<"\n";
  std::cerr<<"\nqflc_n="<<Efc<<"\n";
  std::cerr<<"\nqflv_n="<<Efv<<"\n";
  */
  ustawione='t';
}
/*****************************************************************************/
long gain::Calculate_Spont_Profile() /// liczy widmo emisji spont. (od energii) (pocz, koniec, krok), zwraca liczbe punktów
{
  if(ilwyw>0) return Tspont[0].size();
  ilwyw++;
  if(!Tspont[0].empty())
    {
      Tspont[0].resize(0);
      Tspont[1].resize(0);
    }
  if(ustawione=='n')
    przygobl();
  double (gain::*wzmoc)(double);
  wzmoc=(tau)?& gain::spont_z_posz:& gain::spont0;
  double g;
  double d=100*8e-7;
  for(double en=enpo;en<enko;en+=krok)
    {
      if(Break) break;
      Tspont[0].push_back(en);
      g=Get_gain_at(en);
      Tspont[1].push_back((this->*wzmoc)(en)*(exp(g*d)-1)/g);
    }
  return Tspont[0].size();
}
/*****************************************************************************/
long gain::Calculate_Gain_Profile() /// liczy widmo wzmocnienia (od energii) (pocz, koniec, krok), zwraca liczbe punktów
{
  if(ilwyw>0) return ilpt;
  ilwyw++;
  if(Twzmoc)
    {
      delete [] Twzmoc [0];
      delete [] Twzmoc [1];
      delete [] Twzmoc;
    }
  if(ustawione=='n')
    przygobl();
  long j=0;
  long ilemabyc=(long)floor((enko-enpo)/krok)+2;
  Twzmoc = new double * [2];
  Twzmoc[0] = new double [ilemabyc];
  Twzmoc[1] = new double [ilemabyc];
  double (gain::*wzmoc)(double);
  wzmoc=(tau)?& gain::wzmoc_z_posz:& gain::wzmoc0;
  for(double en=enpo;en<enko;en+=krok)
    {
      if(Break) break;
      Twzmoc[0][j]=en;
      Twzmoc[1][j]=(this->*wzmoc)(en);
      j++;
    }
  ilpt=j;
  return j;
}
/*****************************************************************************
long gain::Calculate_Gain_Profile2()
{
  //  if(ilwyw>0) return ilpt;
  //  ilwyw++;
  if(Twzmoc)
    {
      delete [] Twzmoc [0];
      delete [] Twzmoc [1];
      delete [] Twzmoc;
    }
  //  if(ustawione=='n')
  //  std::cerr<<"\nW Calc 2"<<"\n";
    przygobl2();
  long j=0;
  long ilemabyc=(long)floor((enko-enpo)/krok)+2;
  Twzmoc = new double * [2];
  Twzmoc[0] = new double [ilemabyc];
  Twzmoc[1] = new double [ilemabyc];
  double (gain::*wzmoc)(double);
  wzmoc=(tau)?& gain::wzmoc_z_posz2:& gain::wzmoc02;
  for(double en=enpo;en<enko;en+=krok)
    {
      if(Break) break;
      Twzmoc[0][j]=en;
      Twzmoc[1][j]=(this->*wzmoc)(en);
      j++;
    }
  ilpt=j;
  return j;
}
*****************************************************************************/
long gain::Calculate_Gain_Profile_n(const ExternalLevels& zewpoziomy, double sumaszer)
{
  //  if(ilwyw>0) return ilpt;
  //  ilwyw++;
  if(Twzmoc)
    {
      delete [] Twzmoc [0];
      delete [] Twzmoc [1];
      delete [] Twzmoc;
    }
  //  if(ustawione=='n')
  //  std::cerr<<"\nW Calc 2"<<"\n";
  double sszer=przel_dlug_z_angstr(sumaszer);
  przygobl_n(zewpoziomy, sszer);
  long j=0;
  long ilemabyc=(long)floor((enko-enpo)/krok)+2;
  Twzmoc = new double * [2];
  Twzmoc[0] = new double [ilemabyc];
  Twzmoc[1] = new double [ilemabyc];
  double (gain::*wzmoc)(double,double);
  wzmoc=(tau)?& gain::wzmoc_z_posz_n:& gain::wzmoc0_n;
  for(double en=enpo;en<enko;en+=krok)
    {
      if(Break) break;
      Twzmoc[0][j]=en;
      Twzmoc[1][j]=(this->*wzmoc)(en,sszer);
      j++;
    }
  ilpt=j;
  return j;
}
/*****************************************************************************/
double gain::Find_max_gain() /// szuka maksimum wzmocnienia
{
  int iter=0, it_max=200;
  const gsl_min_fminimizer_type *T;
  gsl_min_fminimizer *s;
  vector<double> min;
  if(ustawione=='n')
    przygobl();
  int k=0;
  while(el.pozoddna(k)>0 && hh.pozoddna(k)>0)
    {
      min.push_back(el.pozoddna(k)+hh.pozoddna(k)+Eg);
      k++;
    }
  k=0;
  while(el.pozoddna(k)>0 && lh.pozoddna(k)>0)
    {
      min.push_back(el.pozoddna(k)+lh.pozoddna(k)+Eg);
      k++;
    }
  sort(min.begin(),min.end());
  double m= min[0];
  unsigned int gdzie=0;
  for(unsigned int i=1;i<=min.size()-1;i++)
    {
      if(Get_gain_at(m)<Get_gain_at(min[i]))
        {
          m=min[i];
          gdzie=i;
        }
    }
  double max=0;
  if(Get_gain_at(m)>0)
    {
      double u=(gdzie==min.size()-1)?Eg+Efc+Efv:min[gdzie+1];
      double l=(gdzie==0)?Eg:min[gdzie-1];
      gsl_function F;
      F.function=min_wzmoc;
      F.params=this;
      T=gsl_min_fminimizer_brent;
      s=gsl_min_fminimizer_alloc(T);
      gsl_min_fminimizer_set(s,&F,m,l,u);
      int /*stat_it,*/ stat_przedz;
      do{
        iter++;
        /*stat_it=*/gsl_min_fminimizer_iterate(s);
        m=gsl_min_fminimizer_minimum(s);
        l=gsl_min_fminimizer_x_lower(s);
        u=gsl_min_fminimizer_x_upper(s);
        stat_przedz=gsl_min_test_interval(l,u,1e-5,0);
        if(stat_przedz==GSL_SUCCESS)
          max=m;
      }while(iter<it_max && stat_przedz!=GSL_SUCCESS);
    }
  else max=-1.;
  return max;
}
/*****************************************************************************/
double gain::Find_max_gain_n(const ExternalLevels& zewpoziomy, double sumaszer)
{
  int iter=0, it_max=200;
  const gsl_min_fminimizer_type *T;
  gsl_min_fminimizer *s;
  vector<double> min;
  double sszer=przel_dlug_z_angstr(sumaszer);
  if(ustawione=='n')
    przygobl_n(zewpoziomy, sszer);
  int k=0;
  while(el.pozoddna(k)>0 && hh.pozoddna(k)>0)
    {
      min.push_back(el.pozoddna(k)+hh.pozoddna(k)+Eg);
      k++;
    }
  k=0;
  while(el.pozoddna(k)>0 && lh.pozoddna(k)>0)
    {
      min.push_back(el.pozoddna(k)+lh.pozoddna(k)+Eg);
      k++;
    }
  sort(min.begin(),min.end());
  double m= min[0];
  unsigned int gdzie=0;
  for(unsigned int i=1;i<=min.size()-1;i++)
    {
      if(Get_gain_at_n(m, zewpoziomy, sumaszer)<Get_gain_at_n(min[i], zewpoziomy, sumaszer))
        {
          m=min[i];
          gdzie=i;
        }
    }
  double max=0;
  if(Get_gain_at_n(m, zewpoziomy, sumaszer)>0)
    {
      double u=(gdzie==min.size()-1)?Eg+Efc+Efv:min[gdzie+1];
      double l=(gdzie==0)?Eg:min[gdzie-1];
      gsl_function F;
      F.function=min_wzmoc;
      F.params=this;
      T=gsl_min_fminimizer_brent;
      s=gsl_min_fminimizer_alloc(T);
      gsl_min_fminimizer_set(s,&F,m,l,u);
      int /*stat_it,*/ stat_przedz;
      do{
        iter++;
        /*stat_it=*/gsl_min_fminimizer_iterate(s);
        m=gsl_min_fminimizer_minimum(s);
        l=gsl_min_fminimizer_x_lower(s);
        u=gsl_min_fminimizer_x_upper(s);
        stat_przedz=gsl_min_test_interval(l,u,1e-5,0);
        if(stat_przedz==GSL_SUCCESS)
          max=m;
      }while(iter<it_max && stat_przedz!=GSL_SUCCESS);
    }
  else max=-1.;
  return max;
}
/*****************************************************************************/
void gain::przygoblE() // LUKASZ
{
  el.~nosnik();
  el.poziomy=znajdzpoziomy(el);
}
/*****************************************************************************/
void gain::przygoblHH() // LUKASZ
{
  hh.~nosnik();
  hh.poziomy=znajdzpoziomy(hh);
}
/*****************************************************************************/
void gain::przygoblLH() // LUKASZ
{
  lh.~nosnik();
  lh.poziomy=znajdzpoziomy(lh);
}
/*****************************************************************************/
double *gain::sendLev(std::vector<double> &zewpoziomy) // LUKASZ
{
  int rozm=zewpoziomy.size();
  double *wsk = new double[rozm+1];
  for(int i=0; i<rozm; ++i)
    wsk[i]=-zewpoziomy[i];
  wsk[rozm]=1.;
  return wsk;
}
/*****************************************************************************/
void gain::przygoblHHc(std::vector<double> &iLevHH) // LUKASZ
{
  hh.poziomy=sendLev(iLevHH);
}
/*****************************************************************************/
void gain::przygoblLHc(std::vector<double> &iLevLH) // LUKASZ
{
  lh.poziomy=sendLev(iLevLH);
}
/*****************************************************************************/
void gain::przygoblQFL(double iTotalWellH) // LUKASZ
{
  double sszer=przel_dlug_z_angstr(iTotalWellH);
	
  Efc=qFlc_n(sszer);
  Efv=qFlv_n(sszer);
  //std::cout << "Quasi Fermi level for electrons = " << Efc << "\n";
  //std::cout << "Quasi Fermi level for holes = " << Efv << "\n";
  ustawione='t';
}
/*****************************************************************************/
double QW::min_wzmoc(double E,void * klasa) /// ?
{
  gain * wzmoc = (gain *)klasa;
  return -wzmoc->Get_gain_at(E);
}
/*****************************************************************************/
double gain::Get_gain_at(double E) /// wzmocnienie dla energii E
{
  if(ustawione=='n')
    przygobl();
  return (tau)? wzmoc_z_posz(E):wzmoc0(E);
}
/*****************************************************************************/
double gain::Get_gain_at_n(double E, double sumaszer)
{
  double sszer=przel_dlug_z_angstr(sumaszer);
  if(ustawione=='n')
    przygobl_n(sszer);
  double (gain::*wzmoc)(double,double);
  wzmoc=(tau)?& gain::wzmoc_z_posz_n:& gain::wzmoc0_n;
  return(this->*wzmoc)(E,sszer);
}
/*****************************************************************************/
double gain::Get_gain_at_n(double E, const ExternalLevels& zewpoziomy, double sumaszer)
{
  double sszer=przel_dlug_z_angstr(sumaszer);
  if(ustawione=='n')
    przygobl_n(zewpoziomy, sszer);
  double (gain::*wzmoc)(double,double);
  wzmoc=(tau)?& gain::wzmoc_z_posz_n:& gain::wzmoc0_n;
  return(this->*wzmoc)(E,sszer);
}

/*****************************************************************************/
double gain::Get_bar_gain_at(double E) /// wzmocnienie (absorpcja) w barierze dla energii E
{
  if(ustawione=='n')
    przygobl();
  double g;
  double k;
  double mi = 1/(1/el.masabar + 1/hh.masabar);
  double deltaE = E-(Eg+el.gleb+hh.gleb);
  if(deltaE<=0)
    g=0;
  else
    {
      k = sqrt(2*mi*deltaE);
      g=M_PI/(c*n_r*ep0*E)/przelm*1e8 * Mt*sqrt(2*mi*deltaE)*mi/(M_PI*M_PI)* (fc(el.gleb + k*k/(2*el.masabar))-fv(-(hh.gleb + k*k/(2*hh.masabar))));
      std::clog<<"\nEe = "<<(el.gleb+k*k/(2*el.masabar))<<" Ehh = "<<(hh.gleb+k*k/(2*hh.masabar))<<std::endl;
      mi = 1/(1/el.masabar + 1/lh.masabar);
      k = sqrt(2*mi*deltaE);
      g+=M_PI/(c*n_r*ep0*E)/przelm*1e8 *Mt*sqrt(2*mi*deltaE)*mi/(M_PI*M_PI)* (fc(el.gleb + k*k/(2*el.masabar))-fv(-(lh.gleb + k*k/(2*lh.masabar))));
    }
  return g;
}
/*****************************************************************************/
double gain::Get_spont_at(double E) /// emisja spontaniczna (intensywnosc [W/m^2 ?]) dla energii E
{
  if(ustawione=='n')
    przygobl();
  double wynik = (tau)? spont_z_posz(E):spont0(E); // w 1/(s cm^3) ma być
  return wynik/(przelm*przelm*przelm)*1e24/przels*1e12;
}
/*****************************************************************************/
void gain::Set_temperature(double temp)
{
  T=temp;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_temperature()
{
  return T;
}
/*****************************************************************************/
void gain::Set_refr_index(double zal)
{
  n_r=zal;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_refr_index()
{
  return n_r;
}
/*****************************************************************************/
void gain::Set_well_width(double szA)
{
  szer=przel_dlug_z_angstr(szA);
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_well_width()
{
  return przel_dlug_na_angstr(szer);
}
/*****************************************************************************/
void gain::Set_barrier_width(double szA)
{
  szerb=przel_dlug_z_angstr(szA);
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_barrier_width()
{
  return przel_dlug_na_angstr(szerb);
}
/*****************************************************************************/
void gain::Set_waveguide_width(double sz)
{
  szer_fal=przel_dlug_z_angstr(sz);
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_waveguide_width()
{
  return przel_dlug_na_angstr(szer_fal);
}
/*****************************************************************************/
void gain::Set_bandgap(double prz)
{
  Eg=prz;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_bandgap()
{
  return Eg;
}
/*****************************************************************************/
void gain::Set_split_off(double de)
{
  deltaSO=de;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_split_off()
{
  return deltaSO;
}
/*****************************************************************************/
void gain::Set_lifetime(double t)
{
  tau=przel_czas_z_psek(t);
  ilwyw=0;
  ustawione='n'; /// wskaznik, ze trzeba cos przeliczyc wewnatrz
}
/*****************************************************************************/
double gain::Get_lifetime()
{
  return przel_czas_na_psek(tau);
}
/*****************************************************************************/
void gain::Set_koncentr(double konce)
{
  konc=przel_konc_z_cm(konce);
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_koncentr()
{
  return przel_konc_na_cm(konc);
}
/*****************************************************************************/
double gain::Get_bar_konc_c()
{
  return przel_konc_na_cm(barkonc_c);
}
/*****************************************************************************/
double gain::Get_bar_konc_v()
{
  return przel_konc_na_cm(barkonc_v);
}
/*****************************************************************************/
double gain::Get_qFlc()
{
  return Efc;
}
/*****************************************************************************/
double gain::Get_qFlv()
{
  return Efv;
}
/*****************************************************************************/
void gain::Set_step(double step)
{
  krok=step;
  ilwyw=0;
}
/*****************************************************************************/
double gain::Get_step()
{
  return krok;
}
/*****************************************************************************/
void gain::Set_first_point(double pierw)
{
  enpo=pierw;
  ilwyw=0;
}
/*****************************************************************************/
double gain::Get_first_point()
{
  return enpo;
}
/*****************************************************************************/
void gain::Set_last_point(double ost)
{
  enko=ost;
  ilwyw=0;
}
/*****************************************************************************/
double gain::Get_last_point()
{
  return enko;
}
/*****************************************************************************/
void gain::Set_conduction_depth(double gle)
{
  el.gleb=gle;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_conduction_depth()
{
  return el.gleb;
}
/*****************************************************************************/
double gain::Get_electron_level_depth(int i)
{
  return (i<el.ilepoz())?-el.poziomy[i]:-1;
}
/*****************************************************************************/
double gain::Get_electron_level_from_bottom(int i)
{
  return (i<el.ilepoz())?el.gleb+el.poziomy[i]:-1;
}
/*****************************************************************************/
double gain::Get_heavy_hole_level_depth(int i)
{
  return (i<hh.ilepoz())?-hh.poziomy[i]:-1;
}
/*****************************************************************************/
double gain::Get_heavy_hole_level_from_bottom(int i)
{
  return (i<hh.ilepoz())?hh.gleb+hh.poziomy[i]:-1;
}
/*****************************************************************************/
double gain::Get_light_hole_level_depth(int i)
{
  return (i<lh.ilepoz())?-lh.poziomy[i]:-1;
}
/*****************************************************************************/
double gain::Get_light_hole_level_from_bottom(int i)
{
  return (i<lh.ilepoz())?lh.gleb+lh.poziomy[i]:-1;
}
/*****************************************************************************/
int gain::Get_number_of_electron_levels()
{
  return el.ilepoz();
}
/*****************************************************************************/
int gain::Get_number_of_heavy_hole_levels()
{
  return hh.ilepoz();
}
/*****************************************************************************/
int gain::Get_number_of_light_hole_levels()
{
  return lh.ilepoz();
}
/*****************************************************************************/
void gain::Set_cond_waveguide_depth(double gle)
{
  el.gleb_fal=gle;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_cond_waveguide_depth()
{
  return el.gleb_fal;
}
/*****************************************************************************/
void gain::Set_valence_depth(double gle)
{
  lh.gleb=gle;
  hh.gleb=gle;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_valence_depth()
{
  return hh.gleb;
}
/*****************************************************************************/
void gain::Set_vale_waveguide_depth(double gle)
{
  hh.gleb_fal=gle;
  lh.gleb_fal=gle;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_vale_waveguide_depth()
{
  return hh.gleb_fal;
}
/*****************************************************************************/
void gain::Set_electron_mass_in_plain(double ma)
{
  el.masa_w_plaszcz=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_electron_mass_in_plain()
{
  return el.masa_w_plaszcz;
}
/*****************************************************************************/
void gain::Set_electron_mass_transverse(double ma)
{
  el.masa_w_kier_prost=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_electron_mass_transverse()
{
  return el.masa_w_kier_prost;
}
/*****************************************************************************/
void gain::Set_heavy_hole_mass_in_plain(double ma)
{
  hh.masa_w_plaszcz=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_heavy_hole_mass_in_plain()
{
  return hh.masa_w_plaszcz;
}
/*****************************************************************************/
void gain::Set_heavy_hole_mass_transverse(double ma)
{
  hh.masa_w_kier_prost=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_heavy_hole_mass_transverse()
{
  return hh.masa_w_kier_prost;
}
/*****************************************************************************/
void gain::Set_light_hole_mass_in_plain(double ma)
{
  lh.masa_w_plaszcz=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_light_hole_mass_in_plain()
{
  return lh.masa_w_plaszcz;
}
/*****************************************************************************/
void gain::Set_light_hole_mass_transverse(double ma)
{
  lh.masa_w_kier_prost=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_light_hole_mass_transverse()
{
  return lh.masa_w_kier_prost;
}
/*****************************************************************************/
void gain::Set_electron_mass_in_barrier(double ma)
{
  el.masabar=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_electron_mass_in_barrier()
{
  return el.masabar;
}
/*****************************************************************************/
void gain::Set_heavy_hole_mass_in_barrier(double ma)
{
  hh.masabar=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_heavy_hole_mass_in_barrier()
{
  return hh.masabar;
}
/*****************************************************************************/
void gain::Set_light_hole_mass_in_barrier(double ma)
{
  lh.masabar=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_light_hole_mass_in_barrier()
{
  return lh.masabar;
}
/*****************************************************************************/
void gain::Set_momentum_matrix_element(double elem)
{
  Mt=elem;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
double gain::Get_momentum_matrix_element()
{
  return Mt;
}
/*****************************************************************************/
double ** gain::Get_gain_tab()
{
  return Twzmoc;
}
/*****************************************************************************/
double gain::Get_inversion(double E, int i)
{
  double E0=Eg+el.pozoddna(i)+hh.pozoddna(i);
  double k=kodE(E-E0,el.masa_w_plaszcz,hh.masa_w_plaszcz);
  return (fc(el.En(k,i))-fv(-hh.En(k,i)));
}
/*****************************************************************************/
std::vector<std::vector<double> > & gain::Get_spont_wek()
{
  return Tspont;
}
/*****************************************************************************/
gain::~gain()
{
  if(Twzmoc)
    {
      if(Twzmoc[0]) delete [] Twzmoc [0];
      if(Twzmoc[1]) delete [] Twzmoc [1];
      delete [] Twzmoc;
    }
  if (!kasuj_poziomy) {
      el.poziomy = hh.poziomy = lh.poziomy = NULL;
  }
}

