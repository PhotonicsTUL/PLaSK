#include <iostream>
#include "gainQW.h"

using QW::nosnik;
using QW::gain;
using QW::parametry;
using namespace std;

const ldouble gain::kB=1.38062/1.60219*1e-4;
const ldouble gain::przelm=10*1.05459/(sqrtl(1.60219*9.10956));
const ldouble gain::przels=1.05459/1.60219*1e-3;
const ldouble gain::ep0=8.8542*1.05459/(100*1.60219*sqrtl(1.60219*9.10956));
const ldouble gain::c=300*sqrtl(9.10956/1.60219);
const ldouble gain::exprng=11100;
int gain::Break=0;

nosnik::nosnik()
{
  poziomy=NULL;
}
/*****************************************************************************/
ldouble nosnik::Eodk(ldouble k) // E(k)
{
  return k*k/(2*masa_w_plaszcz);
}
/*****************************************************************************/
ldouble nosnik::En(ldouble k,int n)
{
  return Eodk(k)+poziomy[n]+gleb;
}
/*****************************************************************************/
ldouble nosnik::pozoddna(int i)
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
  if(poziomy)
    delete [] poziomy;
}
/*****************************************************************************/
parametry::~parametry()
{
  if(ldopar)
    delete [] ldopar;
}
/*****************************************************************************/
gain::gain()
{
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
ldouble gain::En_to_len(ldouble en)
{
  return przel_dlug_na_angstr(2*M_PI*c/en);
}
/*****************************************************************************/
ldouble gain::przel_dlug_z_angstr(ldouble dl_w_A)
{
  return dl_w_A/przelm;
}
/*****************************************************************************/
ldouble gain::przel_dlug_na_angstr(ldouble dl_w_wew)
{
  return dl_w_wew*przelm;
}
/*****************************************************************************/
ldouble gain::przel_czas_z_psek(ldouble czas_w_sek)
{
  return czas_w_sek/przels;
}
/*****************************************************************************/
ldouble gain::przel_czas_na_psek(ldouble czas_w_wew)
{
  return czas_w_wew*przels;
}
/*****************************************************************************/
ldouble gain::przel_konc_z_cm(ldouble konc_w_cm)
{
  return konc_w_cm*1e-24*przelm*przelm*przelm;
}
/*****************************************************************************/
ldouble gain::przel_konc_na_cm(ldouble konc_w_wew)
{
  return konc_w_wew/(przelm*przelm*przelm)*1e24;
}
/*****************************************************************************/
ldouble gain::element()                                                                /// funkcja licząca element macierzowy
{
  return (1/el.masa_w_kier_prost - 1)*(Eg+deltaSO)*Eg/(Eg+2*deltaSO/3)/2;
}
/*****************************************************************************/
ldouble gain::fc(ldouble E)                                                            /// rozkład fermiego dla pasma przewodnictwa
{
  ldouble arg=(E-Efc)/(kB*T);
  return (arg<exprng)?1/(1+expl(arg)):0;
}
/*****************************************************************************/
ldouble gain::fv(ldouble E)                                                            /// rozkład fermiego dla pasma walencyjnego
{
  ldouble arg=(E-Efv)/(kB*T);
  return (arg<exprng)?1/(1+expl(arg)):0;
}
/*****************************************************************************/
ldouble gain::metsiecz(ldouble (gain::*wf)(ldouble,ldouble *),ldouble xl,ldouble xp,ldouble * param,ldouble prec) /// metoda siecznych
{
  if( ((this->*wf)(xl,param))*((this->*wf)(xp,param))>0)
    {
      //      std::cerr<<"\nZ³e krace!\n";
      throw -1;
    }
  ldouble x,y,yp,yl;
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
ldouble gain::gdziepoziomy(ldouble e, ldouble *param) /// zera dają poziomy, e - energia
{
  ldouble v=param[0];
  ldouble m1=param[1];
  ldouble m2=param[2];
  ldouble kI=sqrtl(-2*m1*e); /// sqrt dla long double (standardowa - math)
  ldouble kII=sqrtl(2*m2*(e+v));
  ldouble ilormas=m1/m2; // porawione warunki sklejania pochodnych
  return 2*kI*kII*ilormas*cosl(szer*kII)+(kI*kI - kII*kII*ilormas*ilormas)*sinl(szer*kII);
}
/*****************************************************************************/
ldouble gain::gdziepoziomy2A(ldouble e, ldouble *param) /// 2 - podwójna studnia
{
  ldouble v=param[0];
  ldouble m1=param[1];
  ldouble m2=param[2];
  ldouble kI=sqrtl(-2*m1*e);
  ldouble kII=sqrtl(2*m2*(e+v));
  return 2*kI*kII/(m1*m2)*cosl(szer*kII)+(kI*kI/(m1*m1)-kII*kII/(m2*m2))*sinl(szer*kII)-expl(-kI*szerb)*(kI*kI/(m1*m1)+kII*kII/(m2*m2))*sinl(kII*szer);
}

/*****************************************************************************/
ldouble gain::gdziepoziomy2B(ldouble e, ldouble *param) // ma zawsze 0 w 0
{
  ldouble v=param[0];
  ldouble m1=param[1];
  ldouble m2=param[2];
  ldouble kI=sqrtl(-2*m1*e);
  ldouble kII=sqrtl(2*m2*(e+v));
  return 2*kI*kII/(m1*m2)*cosl(szer*kII)+(kI*kI/(m1*m1)-kII*kII/(m2*m2))*sinl(szer*kII)+expl(-kI*szerb)*(kI*kI/(m1*m1)+kII*kII/(m2*m2))*sinl(kII*szer);
}
/*****************************************************************************/
ldouble gain::krance(int n,ldouble v,ldouble m2) /// krańce przedziału w którym szuka się n-tego poziomu
{
  return (n*M_PI/szer)*(n*M_PI/szer)/(2*m2)-v;
}
/*****************************************************************************/
ldouble * gain::znajdzpoziomy(nosnik & no) /// przy pomocy gdziepoziomy znajduje poziomy
{
  ldouble par[]={no.gleb,no.masabar,no.masa_w_kier_prost};
  ldouble * wsk;
  if(no.masabar<=0 || no.gleb<=0 || no.masa_w_kier_prost<=0)
    {
      wsk=new ldouble[1];
      wsk[0]=1;
    }
  else
    {
      int n=(int)ceill(szer*sqrtl(2*no.masa_w_kier_prost*no.gleb)/M_PI);
      wsk=new ldouble [n+1];
      if(!wsk)
        exit(1);
      ldouble p,q;
      p=mniej(this->krance(1,no.gleb,no.masa_w_kier_prost),(ldouble)0);
      ldouble fp=this->gdziepoziomy(p,par);
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
ldouble * gain::znajdzpoziomy2(nosnik & no) /// j.w. dla podwójnej studni
{
  ldouble przes=1e-7; // startowy punkt zamiast 0 i g³êboko¶ci
  ldouble par[]={no.gleb,no.masabar,no.masa_w_kier_prost};
  /*
  for(double E=0.;E>=-no.gleb;E-=.00005)
    {
      std::cerr<<E<<" "<<gdziepoziomy2A(E,par)<<" "<<gdziepoziomy2B(E,par)<<"\n";
    }
  */

  int n=(int)ceill(szer*sqrtl(2*no.masa_w_kier_prost*no.gleb)/M_PI);
  //  std::cerr<<"\n n="<<n<<"\n";
  ldouble * wsk=new ldouble [2*n+1];
  if(!wsk)
    exit(1);
  ldouble p,pom,ostkr;
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
ldouble gain::gdzieqflc(ldouble ef,ldouble *) /// zero wyznacza kwazi poziom fermiego w pasmie przewodnictwa
{
  ldouble f=0;
  ldouble en;
  ldouble kT=kB*T;
  ldouble gam32 = sqrt(M_PI)/2; // Gamma(3/2)
  ldouble k;
  //  std::cerr<<"\nszer w gdzieqflc_n="<<szer<<"\n";
  f+=szer*kT*gam32*sqrt(kT)*2*sqrt(2*el.masabar)*el.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((ef-el.gleb-el.gleb_fal)/(kB*T)); // w sztukach na powierzchniê
  //    std::cerr<<"\n3D = "<<f<<" dla ef = "<<ef;
  if(el.gleb_fal>0)
    {
      for(int j=(int)ceill(szer_fal*sqrtl(2*el.masabar*el.gleb_fal)/M_PI);j>=1;j--)
        {
          k = j*M_PI/(szer_fal);
          en=k*k/(2*el.masabar)+el.gleb;
          f+=szer/szer_fal*(el.masabar*kT)/M_PI*logl(1+expl(-(en-ef)/(kB*T)));// spin jest
        }
    } // Poziomy nad studini± przybli¿one studni± nieskoñczon±.
  //  f*=szer/szer_fal;  // stosunek objêto¶ci falowodu i studni
  //  std::clog<<"\nkocwbar = "<<f;
  barkonc_c=f/szer;
  for(int i=0;i<=el.ilepoz()-1;i++)
    f+=(el.masa_w_plaszcz*kB*T)/M_PI*logl(1+expl(-(el.pozoddna(i)-ef)/(kB*T)));
  //  std::clog<<"\nkoccalk = "<<f;
  f-=konc*szer;
  return f;
}
/*****************************************************************************/
ldouble gain::gdzieqflc2(ldouble ef,ldouble *) /// j.w. dla podwójnej studni
{
  ldouble f=0;
  ldouble en;
  for(int j=(int)ceill(2*szer_fal*sqrtl(2*el.masa_w_plaszcz*el.gleb_fal)/M_PI);j>=1;j--)
    {
      en=(j*M_PI/(2*szer_fal))*(j*M_PI/(2*szer_fal))/(2*el.masa_w_plaszcz)+el.gleb;
      f+=logl(1+expl(-(en-ef)/(kB*T)));
    } // Poziomy nad studini± przybli¿one studni± nieskoñczon±.
  f*=2*szer/szer_fal;  // stosunek objêto¶ci falowodu i studni
  for(int i=0;i<=el.ilepoz()-1;i++)
    f+=logl(1+expl(-(el.pozoddna(i)-ef)/(kB*T)));
  f-=konc*M_PI*2*szer/(el.masa_w_plaszcz*kB*T);
  return f;
}
/*****************************************************************************/
ldouble gain::gdzieqflc_n(ldouble ef,ldouble * wsk_sszer) /// dla n studni - poziomy podane z zewnątrz
{
  ldouble f=0;
  ldouble en;
  ldouble kT=kB*T;
  ldouble sumaszer=(*wsk_sszer);
  //  std::cerr<<"\nsumaszer w gdzieqflc_n="<<sumaszer<<"\n";
  ldouble gam32 = sqrt(M_PI)/2; // Gamma(3/2)
   ldouble k;
  f+=sumaszer*kT*gam32*sqrt(kT)*2*sqrt(2*el.masabar)*el.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((ef-el.gleb-el.gleb_fal)/(kB*T)); // w sztukach na powierzchniê
  //  std::cerr<<"\n3D_n = "<<f<<" dla ef = "<<ef;
  for(int j=(int)ceill(szer_fal*sqrtl(2*el.masabar*el.gleb_fal)/M_PI);j>=1;j--)
    {
      k = j*M_PI/(szer_fal);
      en=k*k/(2*el.masabar)+el.gleb;
      //      en=(j*M_PI/(2*szer_fal))*(j*M_PI/(2*szer_fal))/(2*el.masabar)+el.gleb;
      f+=sumaszer/szer_fal*(el.masabar*kT)/M_PI*logl(1+expl(-(en-ef)/(kB*T))); // spin jest
    } // Poziomy nad studini± przybli¿one studni± nieskoñczon±.
  //  std::clog<<"\nkocwbar = "<<f;
  barkonc_c=f/sumaszer;
  for(int i=0;i<=el.ilepoz()-1;i++)
    f+=(el.masa_w_plaszcz*kB*T)/M_PI*logl(1+expl(-(el.pozoddna(i)-ef)/(kB*T)));
  f-=konc*sumaszer;
  return f;
}
/*****************************************************************************/
ldouble gain::gdzieqflv(ldouble ef,ldouble *) /// zero wyznacza kwazi poziom fermiego w pasmie walencyjnym
{
  ldouble f=0;
  ldouble en;
  ldouble kT=kB*T;
  ldouble gam32 = sqrt(M_PI)/2; // Gamma(3/2)
  ldouble k;
  f+=szer*kT*gam32*sqrt(kT)*2*sqrt(2*lh.masabar)*lh.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((-ef-lh.gleb-lh.gleb_fal)/(kB*T));
  if(lh.gleb_fal>0)
    {
      for(int j=(int)ceill(szer_fal*sqrtl(2*lh.masabar*lh.gleb_fal)/M_PI);j>=1;j--)
        {
          k = j*M_PI/(szer_fal);
          en=k*k/(2*lh.masabar)+lh.gleb;
          f+=szer/szer_fal*lh.masabar*kT/M_PI*logl(1+expl((-en-ef)/(kB*T)));
        }
    }
  //  f*=szer/szer_fal;  // stosunek objêto¶ci falowodu i studni
  f+=szer*gam32*kT*sqrt(kT)*2*sqrt(2*hh.masabar)*hh.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((-ef-hh.gleb-hh.gleb_fal)/(kB*T));
  if(hh.gleb_fal>0)
    {
      for(int j=(int)ceill(szer_fal*sqrtl(2*hh.masabar*hh.gleb_fal)/M_PI);j>=1;j--)
        {
          k = j*M_PI/(szer_fal);
          en= k*k/(2*hh.masabar)+hh.gleb;
          f+=szer/szer_fal*hh.masabar*kT/M_PI*logl(1+expl((-en-ef)/(kB*T)));
        } // Poziomy nad studini± przybli¿one studni± nieskoñczon±.
    }
  barkonc_v=f/szer;
  //  std::clog<<"\nkocvwbar = "<<f;
  for(int i=0;i<=hh.ilepoz()-1;i++)
    f+=hh.masa_w_plaszcz*kB*T/M_PI*logl(1+expl((-hh.pozoddna(i)-ef)/(kB*T)));
  for(int j=0;j<=lh.ilepoz()-1;j++)
    f+=lh.masa_w_plaszcz*kB*T/M_PI*logl(1+expl((-lh.pozoddna(j)-ef)/(kB*T)));
  //  std::clog<<"\nkocvcalk = "<<f;
  f-=konc*szer;
  return f;
}
/*****************************************************************************
ldouble gain::gdzieqflv2(ldouble ef,ldouble *)
{
  ldouble f=0;
  ldouble en;
  for(int j=(int)ceill(2*szer_fal*sqrtl(2*lh.masa_w_plaszcz*lh.gleb_fal)/M_PI);j>=1;j--)
    {
      en=(j*M_PI/(2*szer_fal))*(j*M_PI/(2*szer_fal))/(2*lh.masa_w_plaszcz)+lh.gleb;
      f+=logl(1+expl((-en-ef)/(kB*T)));
    }
  f*=lh.masa_w_plaszcz;
  f*=2*szer/szer_fal;  // stosunek objêto¶ci falowodu i studni
  for(int j=(int)ceill(2*szer_fal*sqrtl(2*hh.masa_w_plaszcz*hh.gleb_fal)/M_PI);j>=1;j--)
    {
      en=(j*M_PI/(2*szer_fal))*(j*M_PI/(2*szer_fal))/(2*hh.masa_w_plaszcz)+hh.gleb;
      f+=hh.masa_w_plaszcz*logl(1+expl((-en-ef)/(kB*T)))*2*szer/szer_fal;
    } // Poziomy nad studini± przybli¿one studni± nieskoñczon±.
  for(int i=0;i<=hh.ilepoz()-1;i++)
    f+=hh.masa_w_plaszcz*logl(1+expl((-hh.pozoddna(i)-ef)/(kB*T)));
  for(int j=0;j<=lh.ilepoz()-1;j++)
    f+=lh.masa_w_plaszcz*logl(1+expl((-lh.pozoddna(j)-ef)/(kB*T)));
  f-=konc*M_PI*2*szer/(kB*T);
  return f;
}
*****************************************************************************/
ldouble gain::gdzieqflv_n(ldouble ef,ldouble * wsk_sszer)
{
  ldouble f=0;
  ldouble en;
  ldouble sumaszer=(*wsk_sszer);
  ldouble kT=kB*T;
  ldouble gam32 = sqrt(M_PI)/2; // Gamma(3/2)
  ldouble k;
  f+=sumaszer*kT*gam32*sqrt(kT)*2*sqrt(2*lh.masabar)*lh.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((-ef-lh.gleb-lh.gleb_fal)/(kB*T));
  for(int j=(int)ceill(szer_fal*sqrtl(2*lh.masabar*lh.gleb_fal)/M_PI);j>=1;j--)
    {
      k = j*M_PI/(szer_fal);
      en=k*k/(2*lh.masabar)+lh.gleb;
      f+=sumaszer/szer_fal*lh.masabar*kT/M_PI*logl(1+expl((-en-ef)/(kB*T)));
    }
  f+=sumaszer*gam32*kT*sqrt(kT)*2*sqrt(2*hh.masabar)*hh.masabar/(2*M_PI*M_PI)* gsl_sf_fermi_dirac_half ((-ef-hh.gleb-hh.gleb_fal)/(kB*T));
  for(int j=(int)ceill(szer_fal*sqrtl(2*hh.masabar*hh.gleb_fal)/M_PI);j>=1;j--)
    {
      k = j*M_PI/(szer_fal);
      en= k*k/(2*hh.masabar)+hh.gleb;
      f+=sumaszer/szer_fal*hh.masabar*kT/M_PI*logl(1+expl((-en-ef)/(kB*T)));
    } // Poziomy nad studini± przybli¿one studni± nieskoñczon±.
  barkonc_v=f/sumaszer;
  for(int i=0;i<=hh.ilepoz()-1;i++)
    f+=hh.masa_w_plaszcz*kB*T/M_PI*logl(1+expl((-hh.pozoddna(i)-ef)/(kB*T)));
  for(int j=0;j<=lh.ilepoz()-1;j++)
    f+=lh.masa_w_plaszcz*kB*T/M_PI*logl(1+expl((-lh.pozoddna(j)-ef)/(kB*T)));
  //  std::clog<<"\nkocvcalk = "<<f;
  f-=konc*sumaszer;
  return f;
}
/*****************************************************************************/
ldouble gain::qFlc() /// poziomy na podstawie gdzieqflc
{
  ldouble e1=-el.gleb/10;
  ldouble k=el.gleb/100;
  ldouble stare=e1;
  while(gdzieqflc(e1,NULL)>0)
    {
      stare=e1;
      e1-=k;
    }
  ldouble e2=stare;
  while(gdzieqflc(e2,NULL)<0)
    {
      e2+=k;
    }
  ldouble wyn=metsiecz(& gain::gdzieqflc,e1,e2);
  return wyn;
}
/*****************************************************************************/
ldouble gain::qFlc2()
{
  ldouble e1=-el.gleb/10;
  ldouble k=el.gleb/100;
  ldouble stare=e1;
  while(gdzieqflc2(e1,NULL)>0)
    {
      stare=e1;
      e1-=k;
    }
  ldouble e2=stare;
  while(gdzieqflc2(e2,NULL)<0)
    {
      e2+=k;
    }
  ldouble wyn=metsiecz(& gain::gdzieqflc2,e1,e2);
  return wyn;
}
/*****************************************************************************/
ldouble gain::qFlc_n(ldouble sszer)
{
  ldouble e1=-el.gleb/10;
  ldouble k=el.gleb/100;
  ldouble stare=e1;
  while(gdzieqflc_n(e1,&sszer)>0)
    {
      stare=e1;
      e1-=k;
    }
  ldouble e2=stare;
  while(gdzieqflc_n(e2,&sszer)<0)
    {
      e2+=k;
    }
  ldouble wyn=metsiecz(& gain::gdzieqflc_n,e1,e2,&sszer);
  return wyn;
}
/*****************************************************************************/
ldouble gain::qFlv()
{
  ldouble e1=el.gleb/10;
  ldouble k=el.gleb/10;
  ldouble stare=e1;
  while(gdzieqflv(e1,NULL)>0)
    {
      stare=e1;
      e1+=k;
    }
  ldouble e2=stare;
  while(gdzieqflv(e2,NULL)<0)
    {
      e2-=k;
    }
  ldouble wyn=metsiecz(& gain::gdzieqflv,e1,e2);
  return wyn;
}
/*****************************************************************************
ldouble gain::qFlv2()
{
  ldouble e1=el.gleb/10;
  ldouble k=el.gleb/10;
  ldouble stare=e1;
  while(gdzieqflv2(e1,NULL)>0)
    {
      stare=e1;
      e1+=k;
    }
  ldouble e2=stare;
  while(gdzieqflv2(e2,NULL)<0)
    {
      e2-=k;
    }
  ldouble wyn=metsiecz(& gain::gdzieqflv2,e1,e2);
  return wyn;
}
*****************************************************************************/
ldouble gain::qFlv_n(ldouble sszer)
{
  ldouble e1=el.gleb/10;
  ldouble k=el.gleb/10;
  ldouble stare=e1;
  while(gdzieqflv_n(e1,&sszer)>0)
    {
      stare=e1;
      e1+=k;
    }
  ldouble e2=stare;
  while(gdzieqflv_n(e2,&sszer)<0)
    {
      e2-=k;
    }
  ldouble wyn=metsiecz(& gain::gdzieqflv_n,e1,e2,&sszer);
  return wyn;
}
/*****************************************************************************/
inline ldouble gain::L(ldouble x,ldouble b) /// poszerzenie lorentzowskie
{
  return b/(M_PI*(x*x+b*b));
}
/*****************************************************************************/
inline ldouble gain::Lpr(ldouble x,ldouble b) /// pochodna poszerzenia lorentzowskiego
{
  return -2*x*b/(M_PI*(x*x+b*b)*(x*x+b*b));
}
/*****************************************************************************/
ldouble gain::kodE(ldouble E,ldouble mc,ldouble mv) /// k(E) (k nośnika od energii fotonu)
{
  ldouble m=1/(1/mc+1/mv);
  return sqrtl(2*m*E);
}
/*****************************************************************************/
ldouble gain::rored(ldouble,ldouble mc,ldouble mv) /// dwuwymiarowa zredukowana gęstość stanów
{
  ldouble m=1/(1/mc+1/mv);
  return m/(2*M_PI*szer);
}
/*****************************************************************************/
ldouble gain::rored2(ldouble,ldouble mc,ldouble mv)
{
  ldouble m=1/(1/mc+1/mv);
  return m/(4*M_PI*szer);
}
/*****************************************************************************/
ldouble gain::rored_n(ldouble,ldouble mc,ldouble mv, ldouble sumaszer)
{
  ldouble m=1/(1/mc+1/mv);
  return m/(2*M_PI*sumaszer);
}
/*****************************************************************************/
ldouble gain::dosplotu(ldouble E, parametry * param) /// splot lorentza ze wzmonieniem (nie poszerzonym) - funkcja podcałkowa do splotu dla emisji wymuszonej
{
  ldouble *par=param->ldopar;
  ldouble E0=par[0];
  int i=(int)par[3];
  ldouble b=par[1];
  ldouble t=par[2];

  ldouble h_masa_w_plaszcz=(param->rdziury=='h')?hh.masa_w_plaszcz:lh.masa_w_plaszcz;
  ldouble k=kodE(E-E0,el.masa_w_plaszcz,h_masa_w_plaszcz);
  ldouble el_En=el.En(k,i);
  ldouble h_En=(param->rdziury=='h')?hh.En(k,i):lh.En(k,i);
  ldouble cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
  ldouble wspelema=(param->rdziury=='h')?(1+cos2tet)/2:(5-3*cos2tet)/6; // modyfikacja elementu macierzowego w zal. od k
  ldouble f=wspelema*rored(k,el.masa_w_plaszcz,h_masa_w_plaszcz)*( fc(el_En)-fv(-h_En) )/E;
  return f*L(E-t,b);
}
/*****************************************************************************/
ldouble gain::dosplotu2(ldouble E, parametry * param)
{
  ldouble *par=param->ldopar;
  ldouble E0=par[0];
  int i=(int)par[3];
  ldouble b=par[1];
  ldouble t=par[2];

  ldouble h_masa_w_plaszcz=(param->rdziury=='h')?hh.masa_w_plaszcz:lh.masa_w_plaszcz;
  ldouble k=kodE(E-E0,el.masa_w_plaszcz,h_masa_w_plaszcz);
  ldouble el_En=el.En(k,i);
  ldouble h_En=(param->rdziury=='h')?hh.En(k,i):lh.En(k,i);
  ldouble cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
  ldouble wspelema=(param->rdziury=='h')?(1+cos2tet)/2:(5-3*cos2tet)/6; // modyfikacja elementu macierzowego w zal. od k
  ldouble f=wspelema*rored2(k,el.masa_w_plaszcz,h_masa_w_plaszcz)*( fc(el_En)-fv(-h_En) )/E;
  return f*L(E-t,b);
}
/*****************************************************************************/
ldouble gain::dosplotu_n(ldouble E, parametry * param)
{
  ldouble *par=param->ldopar;
  ldouble E0=par[0];
  int i=(int)par[3];
  ldouble b=par[1];
  ldouble t=par[2];
  ldouble sumszer=par[4];

  ldouble h_masa_w_plaszcz=(param->rdziury=='h')?hh.masa_w_plaszcz:lh.masa_w_plaszcz;
  ldouble k=kodE(E-E0,el.masa_w_plaszcz,h_masa_w_plaszcz);
  ldouble el_En=el.En(k,i);
  ldouble h_En=(param->rdziury=='h')?hh.En(k,i):lh.En(k,i);
  ldouble cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
  ldouble wspelema=(param->rdziury=='h')?(1+cos2tet)/2:(5-3*cos2tet)/6; // modyfikacja elementu macierzowego w zal. od k
  ldouble f=wspelema*rored_n(k,el.masa_w_plaszcz,h_masa_w_plaszcz,sumszer)*( fc(el_En)-fv(-h_En) )/E;
  return f*L(E-t,b);
}
/*****************************************************************************/
ldouble gain::wzmoc_z_posz(ldouble t) /// wykonuje całkę (splot) z funkcją dosplotu
{
  int i=0;
  ldouble Ec,Ev;
  ldouble g=0;
  Ev=hh.pozoddna(0);
  Ec=el.pozoddna(0);
  ldouble E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  ldouble stala=M_PI/(c*n_r*ep0)/przelm*1e8;
  ldouble epsb;
  ldouble * ldpar=new ldouble [4];
  parametry * param=new parametry;
  param->ldopar=ldpar;
  param->rdziury='h';
  ldouble b=1/tau;
  ldpar[1]=b;
  ldpar[2]=t;
  ldouble lc=1/(1+el.masa_w_plaszcz/hh.masa_w_plaszcz);
  ldouble lv=1/(1+hh.masa_w_plaszcz/el.masa_w_plaszcz);
  ldouble M=(1/Eg*( 2/(Eg*Eg) + 2/(Eg*kB*T)*(lc+lv) + (lc*lc+lv*lv)/(kB*T*kB*T)))/(b*M_PI);
  M+=.75*sqrt(3.)/(M_PI*b*b*Eg)*(1/Eg+lc/(kB*T)+lv/(kB*T));
  M+=2/(Eg*b*b*b*M_PI);
  epsb=bladb/(stala*3*Mt*el.ilepoz()/2);
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(ldouble)i;
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
      ldpar[3]=(ldouble)i;
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
ldouble gain::wzmoc_z_posz2(ldouble t)
{
  int i=0;
  ldouble Ec,Ev;
  ldouble g=0;
  Ev=hh.pozoddna(0);
  Ec=el.pozoddna(0);
  ldouble E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  ldouble stala=M_PI/(c*n_r*ep0)/przelm*1e8;
  ldouble epsb;
  ldouble * ldpar=new ldouble [4];
  parametry * param=new parametry;
  param->ldopar=ldpar;
  param->rdziury='h';
  ldouble b=1/tau;
  ldpar[1]=b;
  ldpar[2]=t;
  ldouble lc=1/(1+el.masa_w_plaszcz/hh.masa_w_plaszcz);
  ldouble lv=1/(1+hh.masa_w_plaszcz/el.masa_w_plaszcz);
  ldouble M=(1/Eg*( 2/(Eg*Eg) + 2/(Eg*kB*T)*(lc+lv) + (lc*lc+lv*lv)/(kB*T*kB*T)))/(b*M_PI);
  M+=.75*sqrt(3.)/(M_PI*b*b*Eg)*(1/Eg+lc/(kB*T)+lv/(kB*T));
  M+=2/(Eg*b*b*b*M_PI);
  epsb=bladb/(stala*3*Mt*el.ilepoz()/2);
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(ldouble)i;
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
      ldpar[3]=(ldouble)i;
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
ldouble gain::wzmoc_z_posz_n(ldouble t, ldouble sumszer)
{
  int i=0;
  ldouble Ec,Ev;
  ldouble g=0;
  Ev=hh.pozoddna(0);
  Ec=el.pozoddna(0);
  ldouble E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  ldouble stala=M_PI/(c*n_r*ep0)/przelm*1e8;
  ldouble epsb;
  ldouble * ldpar=new ldouble [5];
  parametry * param=new parametry;
  param->ldopar=ldpar;
  param->rdziury='h';
  ldouble b=1/tau;
  ldpar[1]=b;
  ldpar[2]=t;
  ldpar[4]=sumszer;

  ldouble lc=1/(1+el.masa_w_plaszcz/hh.masa_w_plaszcz);
  ldouble lv=1/(1+hh.masa_w_plaszcz/el.masa_w_plaszcz);
  ldouble M=(1/Eg*( 2/(Eg*Eg) + 2/(Eg*kB*T)*(lc+lv) + (lc*lc+lv*lv)/(kB*T*kB*T)))/(b*M_PI);
  M+=.75*sqrt(3.)/(M_PI*b*b*Eg)*(1/Eg+lc/(kB*T)+lv/(kB*T));
  M+=2/(Eg*b*b*b*M_PI);
  epsb=bladb/(stala*3*Mt*el.ilepoz()/2);
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(ldouble)i;
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
      ldpar[3]=(ldouble)i;
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
ldouble gain::dosplotu_spont(ldouble E, parametry * param) /// funkcja podcałkowa do splotu dla emisji spont.
{
  ldouble *par=param->ldopar;
  ldouble E0=par[0];
  int i=(int)par[3];
  ldouble b=par[1];
  ldouble t=par[2];

  ldouble h_masa_w_plaszcz=(param->rdziury=='h')?hh.masa_w_plaszcz:lh.masa_w_plaszcz;
  ldouble k=kodE(E-E0,el.masa_w_plaszcz,h_masa_w_plaszcz);
  ldouble el_En=el.En(k,i);
  ldouble h_En=(param->rdziury=='h')?hh.En(k,i):lh.En(k,i);
  /*
  ldouble cos2tet=(E>Eg)?(E0-Eg)/(E-Eg):1.0;
  ldouble wspelema=(param->rdziury=='h')?(1+cos2tet)/2:(5-3*cos2tet)/6; // modyfikacja elementu macierzowego w zal. od k
  */
  //  ldouble f=wspelema*rored(k,el.masa_w_plaszcz,h_masa_w_plaszcz)*( fc(el_En)*(1-fv(-h_En)) )/E;
  ldouble f=E*E*rored(k,el.masa_w_plaszcz,h_masa_w_plaszcz)*( fc(el_En)*(1-fv(-h_En)) );
  return f*L(E-t,b);
}
/*****************************************************************************/
ldouble gain::spont_z_posz(ldouble t) /// to samo co wzmoc_posz tylko, że spontaniczne
{
  int i=0;
  ldouble Ec,Ev;
  ldouble g=0;
  Ev=hh.pozoddna(0);
  Ec=el.pozoddna(0);
  ldouble E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  ldouble stala=n_r/(M_PI*c*c*c*ep0); // Nie ma przelicznikow, bo Get przeliczy na zwykle jednostki
  ldouble epsb;
  ldouble * ldpar=new ldouble [4];
  parametry * param=new parametry;
  param->ldopar=ldpar;
  param->rdziury='h';
  ldouble b=1/tau;
  ldpar[1]=b;
  ldpar[2]=t;
  ldouble lc=1/(1+el.masa_w_plaszcz/hh.masa_w_plaszcz);
  ldouble lv=1/(1+hh.masa_w_plaszcz/el.masa_w_plaszcz);
  ldouble M=(1/Eg*( 2/(Eg*Eg) + 2/(Eg*kB*T)*(lc+lv) + (lc*lc+lv*lv)/(kB*T*kB*T)))/(b*M_PI); //Oszacowanie pochodnej (chyba)
  M+=.75*sqrt(3.)/(M_PI*b*b*Eg)*(1/Eg+lc/(kB*T)+lv/(kB*T));
  M+=2/(Eg*b*b*b*M_PI);
  epsb=bladb/(stala*3*Mt*el.ilepoz()/2);
  while(Ec>0 && Ev>0)
    {
      ldpar[0]=E0;
      ldpar[3]=(ldouble)i;
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
      ldpar[3]=(ldouble)i;
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
ldouble gain::Prost(ldouble (gain::*F)(ldouble, parametry *),ldouble M, ldouble a, ldouble b, parametry * par, ldouble bld) /// metoda prostokątów (całkowania)
{
  ldouble szer=b-a;
  long N=(long)ceill(szer*sqrtl(szer*M/(24*bld)));
  ldouble podz=szer/N;
  ldouble wyn=0;
  for(long k=0;k<=N-1;k++)
    {
      wyn+=(this->*F)(a+podz*((ldouble)k+.5),par);
    }
  return podz*wyn;
}
/*****************************************************************************/
ldouble gain::wzmoc0(ldouble E) /// liczy wzmocnienie bez poszerzenia
{
  int i=0;
  ldouble Ec,Ev;
  ldouble g=0;
  ldouble E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  ldouble k;
  ldouble cos2tet;
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
ldouble gain::wzmoc02(ldouble E)
{
  int i=0;
  ldouble Ec,Ev;
  ldouble g=0;
  ldouble E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  ldouble k;
  ldouble cos2tet;
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
ldouble gain::wzmoc0_n(ldouble E, ldouble sumszer)
{
  int i=0;
  ldouble Ec,Ev;
  ldouble g=0;
  ldouble E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  ldouble k;
  ldouble cos2tet;
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
ldouble gain::spont0(ldouble E) /// liczy emisję spont. bez poszerzenia
{
  int i=0;
  ldouble Ec,Ev;
  ldouble g=0;
  ldouble E0=Eg+el.pozoddna(0)+hh.pozoddna(0);
  ldouble k;
  ldouble cos2tet;
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
      exit(1);
    }
  el.~nosnik();
  el.poziomy=znajdzpoziomy(el);
  hh.~nosnik();
  hh.poziomy=znajdzpoziomy(hh);
  lh.~nosnik();
  lh.poziomy=znajdzpoziomy(lh);
  Efc=qFlc();
  Efv=qFlv();
  /*  std::cerr<<"\nszer="<<szer<<"\n";
  std::cerr<<"\nqflc1="<<Efc<<"\n";
  std::cerr<<"\nqflv1="<<Efv<<"\n";*/
  ustawione='t';
}
/*****************************************************************************/
// Marcin Gebski 21.02.2013
void gain::runPrzygobl()
{
    this->przygobl();
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
      exit(1);
    }
  el.~nosnik();
  el.poziomy=znajdzpoziomy2(el);
  //  std::cerr<<"\nel2 poziomy "<<el.ilepoz()<<"\n";
  hh.~nosnik();
  hh.poziomy=znajdzpoziomy2(hh);
  //  std::cerr<<"\nhh2 poziomy "<<hh.ilepoz()<<"\n";
  lh.~nosnik();
  lh.poziomy=znajdzpoziomy2(lh);
  //  std::cerr<<"\nlh2 poziomy "<<lh.ilepoz()<<"\n";
  Efc=qFlc2();
  Efv=qFlv2();
  ustawione='t';
}
*****************************************************************************/
ldouble * gain::z_vec_wsk(std::vector<std::vector<ldouble> > & zewpoziomy, int k) /// z wektora wskaźnik
{
  size_t rozm=zewpoziomy[k].size();
  ldouble * wsk = new ldouble [rozm+1];
  for(size_t i=0; i<=rozm-1; i++)
    {
      wsk[i]=zewpoziomy[k][i];
    }
  wsk[rozm]=1.;
  return wsk;
}
/*****************************************************************************/
void gain::przygobl_n(std::vector<std::vector<ldouble> > & zewpoziomy, ldouble sumaszer)
{
  //  std::cerr<<"\nW n\n";
  if(Mt<=0)
    {
      Mt=element();
    }
  if(T<0 || n_r<0 || szer<0 || szer_fal<0 || Eg<0 || Mt<0 || tau<0 || konc<0)
    {
      exit(1);
    }
  el.~nosnik();
  el.poziomy=z_vec_wsk(zewpoziomy, 0);
  //  std::cerr<<"\neln poziomy "<<el.ilepoz()<<"\n";
  hh.~nosnik();
  hh.poziomy=z_vec_wsk(zewpoziomy, 1);
  //  std::cerr<<"\nhhn poziomy "<<hh.ilepoz()<<"\n";
  lh.~nosnik();
  lh.poziomy=z_vec_wsk(zewpoziomy, 2);
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
long gain::Calculate_Spont_Profile() /// liczy widmo emisji spont. (od energii) (pocz, koniec, krok), zwraca liczbę punktów
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
  ldouble (gain::*wzmoc)(ldouble);
  wzmoc=(tau)?& gain::spont_z_posz:& gain::spont0;
  double g;
  double d=100*8e-7;
  for(ldouble en=enpo;en<enko;en+=krok)
    {
      if(Break) break;
      Tspont[0].push_back(en);
      g=Get_gain_at(en);
      Tspont[1].push_back((this->*wzmoc)(en)*(exp(g*d)-1)/g);
    }
  return Tspont[0].size();
}
/*****************************************************************************/
long gain::Calculate_Gain_Profile() /// liczy widmo wzmocnienia (od energii) (pocz, koniec, krok), zwraca liczbę punktów
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
  Twzmoc = new ldouble * [2];
  Twzmoc[0] = new ldouble [ilemabyc];
  Twzmoc[1] = new ldouble [ilemabyc];
  ldouble (gain::*wzmoc)(ldouble);
  wzmoc=(tau)?& gain::wzmoc_z_posz:& gain::wzmoc0;
  for(ldouble en=enpo;en<enko;en+=krok)
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
  Twzmoc = new ldouble * [2];
  Twzmoc[0] = new ldouble [ilemabyc];
  Twzmoc[1] = new ldouble [ilemabyc];
  ldouble (gain::*wzmoc)(ldouble);
  wzmoc=(tau)?& gain::wzmoc_z_posz2:& gain::wzmoc02;
  for(ldouble en=enpo;en<enko;en+=krok)
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
long gain::Calculate_Gain_Profile_n(std::vector<std::vector<ldouble> > & zewpoziomy, ldouble sumaszer)
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
  ldouble sszer=przel_dlug_z_angstr(sumaszer);
  przygobl_n(zewpoziomy, sszer);
  long j=0;
  long ilemabyc=(long)floor((enko-enpo)/krok)+2;
  Twzmoc = new ldouble * [2];
  Twzmoc[0] = new ldouble [ilemabyc];
  Twzmoc[1] = new ldouble [ilemabyc];
  ldouble (gain::*wzmoc)(ldouble,ldouble);
  wzmoc=(tau)?& gain::wzmoc_z_posz_n:& gain::wzmoc0_n;
  for(ldouble en=enpo;en<enko;en+=krok)
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
ldouble gain::Find_max_gain() /// szuka maksimum wzmocnienia
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
      int stat_it, stat_przedz;
      do{
        iter++;
        stat_it=gsl_min_fminimizer_iterate(s);
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
ldouble gain::Find_max_gain_n(std::vector<std::vector<ldouble> > & zewpoziomy, ldouble sumaszer)
{
  int iter=0, it_max=200;
  const gsl_min_fminimizer_type *T;
  gsl_min_fminimizer *s;
  vector<double> min;
  ldouble sszer=przel_dlug_z_angstr(sumaszer);
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
      int stat_it, stat_przedz;
      do{
        iter++;
        stat_it=gsl_min_fminimizer_iterate(s);
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
double QW::min_wzmoc(double E,void * klasa) /// ?
{
  gain * wzmoc = (gain *)klasa;
  return -wzmoc->Get_gain_at(E);
}
/*****************************************************************************/
ldouble gain::Get_gain_at(ldouble E) /// wzmocnienie dla energii E
{
  if(ustawione=='n')
    przygobl();
  return (tau)? wzmoc_z_posz(E):wzmoc0(E);
}
/*****************************************************************************/
ldouble gain::Get_gain_at_n(ldouble E,std::vector<std::vector<ldouble> > & zewpoziomy, double sumaszer)
{
  ldouble sszer=przel_dlug_z_angstr(sumaszer);
  if(ustawione=='n')
    przygobl_n(zewpoziomy, sszer);
  ldouble (gain::*wzmoc)(ldouble,ldouble);
  wzmoc=(tau)?& gain::wzmoc_z_posz_n:& gain::wzmoc0_n;
  return(this->*wzmoc)(E,sszer);
}

/*****************************************************************************/
ldouble gain::Get_bar_gain_at(ldouble E) /// wzmocnienie (absorpcja) w barierze dla energii E
{
  if(ustawione=='n')
    przygobl();
  ldouble g;
  ldouble k;
  ldouble mi = 1/(1/el.masabar + 1/hh.masabar);
  ldouble deltaE = E-(Eg+el.gleb+hh.gleb);
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
ldouble gain::Get_spont_at(ldouble E) /// emisja spontaniczna (intensywność [W/m^2 ?]) dla energii E
{
  if(ustawione=='n')
    przygobl();
  double wynik = (tau)? spont_z_posz(E):spont0(E); // w 1/(s cm^3) ma byæ
  return wynik/(przelm*przelm*przelm)*1e24/przels*1e12;
}
/*****************************************************************************/
void gain::Set_temperature(ldouble temp)
{
  T=temp;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_temperature()
{
  return T;
}
/*****************************************************************************/
void gain::Set_refr_index(ldouble zal)
{
  n_r=zal;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_refr_index()
{
  return n_r;
}
/*****************************************************************************/
void gain::Set_well_width(ldouble szA)
{
  szer=przel_dlug_z_angstr(szA);
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_well_width()
{
  return przel_dlug_na_angstr(szer);
}
/*****************************************************************************/
void gain::Set_barrier_width(ldouble szA)
{
  szerb=przel_dlug_z_angstr(szA);
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_barrier_width()
{
  return przel_dlug_na_angstr(szerb);
}
/*****************************************************************************/
void gain::Set_waveguide_width(ldouble sz)
{
  szer_fal=przel_dlug_z_angstr(sz);
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_waveguide_width()
{
  return przel_dlug_na_angstr(szer_fal);
}
/*****************************************************************************/
void gain::Set_bandgap(ldouble prz)
{
  Eg=prz;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_bandgap()
{
  return Eg;
}
/*****************************************************************************/
void gain::Set_split_off(ldouble de)
{
  deltaSO=de;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_split_off()
{
  return deltaSO;
}
/*****************************************************************************/
void gain::Set_lifetime(ldouble t)
{
  tau=przel_czas_z_psek(t);
  ilwyw=0;
  ustawione='n'; /// wskaźnik, że trzeba coś przeliczyć wewnątrz
}
/*****************************************************************************/
ldouble gain::Get_lifetime()
{
  return przel_czas_na_psek(tau);
}
/*****************************************************************************/
void gain::Set_koncentr(ldouble konce)
{
  konc=przel_konc_z_cm(konce);
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_koncentr()
{
  return przel_konc_na_cm(konc);
}
/*****************************************************************************/
ldouble gain::Get_bar_konc_c()
{
  return przel_konc_na_cm(barkonc_c);
}
/*****************************************************************************/
ldouble gain::Get_bar_konc_v()
{
  return przel_konc_na_cm(barkonc_v);
}
/*****************************************************************************/
ldouble gain::Get_qFlc()
{
  return Efc;
}
/*****************************************************************************/
ldouble gain::Get_qFlv()
{
  return Efv;
}
/*****************************************************************************/
void gain::Set_step(ldouble step)
{
  krok=step;
  ilwyw=0;
}
/*****************************************************************************/
ldouble gain::Get_step()
{
  return krok;
}
/*****************************************************************************/
void gain::Set_first_point(ldouble pierw)
{
  enpo=pierw;
  ilwyw=0;
}
/*****************************************************************************/
ldouble gain::Get_first_point()
{
  return enpo;
}
/*****************************************************************************/
void gain::Set_last_point(ldouble ost)
{
  enko=ost;
  ilwyw=0;
}
/*****************************************************************************/
ldouble gain::Get_last_point()
{
  return enko;
}
/*****************************************************************************/
void gain::Set_conduction_depth(ldouble gle)
{
  el.gleb=gle;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_conduction_depth()
{
  return el.gleb;
}
/*****************************************************************************/
ldouble gain::Get_electron_level_depth(int i)
{
  return (i<el.ilepoz())?-el.poziomy[i]:-1;
}
/*****************************************************************************/
ldouble gain::Get_electron_level_from_bottom(int i)
{
  return (i<el.ilepoz())?el.gleb+el.poziomy[i]:-1;
}
/*****************************************************************************/
ldouble gain::Get_heavy_hole_level_depth(int i)
{
  return (i<hh.ilepoz())?-hh.poziomy[i]:-1;
}
/*****************************************************************************/
ldouble gain::Get_heavy_hole_level_from_bottom(int i)
{
  return (i<hh.ilepoz())?hh.gleb+hh.poziomy[i]:-1;
}
/*****************************************************************************/
ldouble gain::Get_light_hole_level_depth(int i)
{
  return (i<lh.ilepoz())?-lh.poziomy[i]:-1;
}
/*****************************************************************************/
ldouble gain::Get_light_hole_level_from_bottom(int i)
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
void gain::Set_cond_waveguide_depth(ldouble gle)
{
  el.gleb_fal=gle;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_cond_waveguide_depth()
{
  return el.gleb_fal;
}
/*****************************************************************************/
void gain::Set_valence_depth(ldouble gle)
{
  lh.gleb=gle;
  hh.gleb=gle;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_valence_depth()
{
  return hh.gleb;
}
/*****************************************************************************/
void gain::Set_vale_waveguide_depth(ldouble gle)
{
  hh.gleb_fal=gle;
  lh.gleb_fal=gle;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_vale_waveguide_depth()
{
  return hh.gleb_fal;
}
/*****************************************************************************/
void gain::Set_electron_mass_in_plain(ldouble ma)
{
  el.masa_w_plaszcz=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_electron_mass_in_plain()
{
  return el.masa_w_plaszcz;
}
/*****************************************************************************/
void gain::Set_electron_mass_transverse(ldouble ma)
{
  el.masa_w_kier_prost=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_electron_mass_transverse()
{
  return el.masa_w_kier_prost;
}
/*****************************************************************************/
void gain::Set_heavy_hole_mass_in_plain(ldouble ma)
{
  hh.masa_w_plaszcz=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_heavy_hole_mass_in_plain()
{
  return hh.masa_w_plaszcz;
}
/*****************************************************************************/
void gain::Set_heavy_hole_mass_transverse(ldouble ma)
{
  hh.masa_w_kier_prost=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_heavy_hole_mass_transverse()
{
  return hh.masa_w_kier_prost;
}
/*****************************************************************************/
void gain::Set_light_hole_mass_in_plain(ldouble ma)
{
  lh.masa_w_plaszcz=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_light_hole_mass_in_plain()
{
  return lh.masa_w_plaszcz;
}
/*****************************************************************************/
void gain::Set_light_hole_mass_transverse(ldouble ma)
{
  lh.masa_w_kier_prost=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_light_hole_mass_transverse()
{
  return lh.masa_w_kier_prost;
}
/*****************************************************************************/
void gain::Set_electron_mass_in_barrier(ldouble ma)
{
  el.masabar=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_electron_mass_in_barrier()
{
  return el.masabar;
}
/*****************************************************************************/
void gain::Set_heavy_hole_mass_in_barrier(ldouble ma)
{
  hh.masabar=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_heavy_hole_mass_in_barrier()
{
  return hh.masabar;
}
/*****************************************************************************/
void gain::Set_light_hole_mass_in_barrier(ldouble ma)
{
  lh.masabar=ma;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_light_hole_mass_in_barrier()
{
  return lh.masabar;
}
/*****************************************************************************/
void gain::Set_momentum_matrix_element(ldouble elem)
{
  Mt=elem;
  ilwyw=0;
  ustawione='n';
}
/*****************************************************************************/
ldouble gain::Get_momentum_matrix_element()
{
  return Mt;
}
/*****************************************************************************/
ldouble ** gain::Get_gain_tab()
{
  return Twzmoc;
}
/*****************************************************************************/
ldouble gain::Get_inversion(ldouble E, int i)
{
  ldouble E0=Eg+el.pozoddna(i)+hh.pozoddna(i);
  ldouble k=kodE(E-E0,el.masa_w_plaszcz,hh.masa_w_plaszcz);
  return (fc(el.En(k,i))-fv(-hh.En(k,i)));
}
/*****************************************************************************/
std::vector<std::vector<ldouble> > & gain::Get_spont_wek()
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
}

