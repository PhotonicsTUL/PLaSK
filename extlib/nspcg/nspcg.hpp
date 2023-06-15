#include <plask/config.hpp>


//////////////////////// Available preconditioners ////////////////////////

#define rich1 F77_GLOBAL(rich1,RICH1)
F77SUB rich1(...);

#define rich2 F77_GLOBAL(rich2,RICH2)
F77SUB rich2(...);

#define rich3 F77_GLOBAL(rich3,RICH3)
F77SUB rich3(...);

#define rich4 F77_GLOBAL(rich4,RICH4)
F77SUB rich4(...);

#define rich5 F77_GLOBAL(rich5,RICH5)
F77SUB rich5(...);

#define jac1 F77_GLOBAL(jac1,JAC1)
F77SUB jac1(...);

#define jac2 F77_GLOBAL(jac2,JAC2)
F77SUB jac2(...);

#define jac3 F77_GLOBAL(jac3,JAC3)
F77SUB jac3(...);

#define jac4 F77_GLOBAL(jac4,JAC4)
F77SUB jac4(...);

#define jac5 F77_GLOBAL(jac5,JAC5)
F77SUB jac5(...);

#define ljac2 F77_GLOBAL(ljac2,LJAC2)
F77SUB ljac2(...);

#define ljac3 F77_GLOBAL(ljac3,LJAC3)
F77SUB ljac3(...);

#define ljacx2 F77_GLOBAL(ljacx2,LJACX2)
F77SUB ljacx2(...);

#define ljacx3 F77_GLOBAL(ljacx3,LJACX3)
F77SUB ljacx3(...);

#define sor1 F77_GLOBAL(sor1,SOR1)
F77SUB sor1(...);

#define sor2 F77_GLOBAL(sor2,SOR2)
F77SUB sor2(...);

#define sor3 F77_GLOBAL(sor3,SOR3)
F77SUB sor3(...);

#define sor6 F77_GLOBAL(sor6,SOR6)
F77SUB sor6(...);

#define sor7 F77_GLOBAL(sor7,SOR7)
F77SUB sor7(...);

#define ssor1 F77_GLOBAL(ssor1,SSOR1)
F77SUB ssor1(...);

#define ssor2 F77_GLOBAL(ssor2,SSOR2)
F77SUB ssor2(...);

#define ssor3 F77_GLOBAL(ssor3,SSOR3)
F77SUB ssor3(...);

#define ssor6 F77_GLOBAL(ssor6,SSOR6)
F77SUB ssor6(...);

#define ssor7 F77_GLOBAL(ssor7,SSOR7)
F77SUB ssor7(...);

#define ic1 F77_GLOBAL(ic1,IC1)
F77SUB ic1(...);

#define ic2 F77_GLOBAL(ic2,IC2)
F77SUB ic2(...);

#define ic3 F77_GLOBAL(ic3,IC3)
F77SUB ic3(...);

#define ic6 F77_GLOBAL(ic6,IC6)
F77SUB ic6(...);

#define mic1 F77_GLOBAL(mic1,MIC1)
F77SUB mic1(...);

#define mic2 F77_GLOBAL(mic2,MIC2)
F77SUB mic2(...);

#define mic3 F77_GLOBAL(mic3,MIC3)
F77SUB mic3(...);

#define mic6 F77_GLOBAL(mic6,MIC6)
F77SUB mic6(...);

#define lsp1 F77_GLOBAL(lsp1,LSP1)
F77SUB lsp1(...);

#define lsp2 F77_GLOBAL(lsp2,LSP2)
F77SUB lsp2(...);

#define lsp3 F77_GLOBAL(lsp3,LSP3)
F77SUB lsp3(...);

#define lsp4 F77_GLOBAL(lsp4,LSP4)
F77SUB lsp4(...);

#define lsp5 F77_GLOBAL(lsp5,LSP5)
F77SUB lsp5(...);

#define neu1 F77_GLOBAL(neu1,NEU1)
F77SUB neu1(...);

#define neu2 F77_GLOBAL(neu2,NEU2)
F77SUB neu2(...);

#define neu3 F77_GLOBAL(neu3,NEU3)
F77SUB neu3(...);

#define neu4 F77_GLOBAL(neu4,NEU4)
F77SUB neu4(...);

#define neu5 F77_GLOBAL(neu5,NEU5)
F77SUB neu5(...);

#define lsor2 F77_GLOBAL(lsor2,LSOR2)
F77SUB lsor2(...);

#define lsor3 F77_GLOBAL(lsor3,LSOR3)
F77SUB lsor3(...);

#define lssor2 F77_GLOBAL(lssor2,LSSOR2)
F77SUB lssor2(...);

#define lssor3 F77_GLOBAL(lssor3,LSSOR3)
F77SUB lssor3(...);

#define llsp2 F77_GLOBAL(llsp2,LLSP2)
F77SUB llsp2(...);

#define llsp3 F77_GLOBAL(llsp3,LLSP3)
F77SUB llsp3(...);

#define lneu2 F77_GLOBAL(lneu2,LNEU2)
F77SUB lneu2(...);

#define lneu3 F77_GLOBAL(lneu3,LNEU3)
F77SUB lneu3(...);

#define bic2 F77_GLOBAL(bic2,BIC2)
F77SUB bic2(...);

#define bic3 F77_GLOBAL(bic3,BIC3)
F77SUB bic3(...);

#define bic7 F77_GLOBAL(bic7,BIC7)
F77SUB bic7(...);

#define bicx2 F77_GLOBAL(bicx2,BICX2)
F77SUB bicx2(...);

#define bicx3 F77_GLOBAL(bicx3,BICX3)
F77SUB bicx3(...);

#define bicx7 F77_GLOBAL(bicx7,BICX7)
F77SUB bicx7(...);

#define mbic2 F77_GLOBAL(mbic2,MBIC2)
F77SUB mbic2(...);

#define mbic3 F77_GLOBAL(mbic3,MBIC3)
F77SUB mbic3(...);

#define mbic7 F77_GLOBAL(mbic7,MBIC7)
F77SUB mbic7(...);

#define mbicx2 F77_GLOBAL(mbicx2,MBICX2)
F77SUB mbicx2(...);

#define mbicx3 F77_GLOBAL(mbicx3,MBICX3)
F77SUB mbicx3(...);

#define mbicx7 F77_GLOBAL(mbicx7,MBICX7)
F77SUB mbicx7(...);

#define rs6 F77_GLOBAL(rs6,RS6)
F77SUB rs6(...);

#define rs7 F77_GLOBAL(rs7,RS7)
F77SUB rs7(...);


//////////////////////// Available accelerators ////////////////////////

#define cg F77_GLOBAL(cg,CG)
F77SUB cg(...);

#define si F77_GLOBAL(si,SI)
F77SUB si(...);

#define sor F77_GLOBAL(sor,SOR)
F77SUB sor(...);

#define srcg F77_GLOBAL(srcg,SRCG)
F77SUB srcg(...);

#define srsi F77_GLOBAL(srsi,SRSI)
F77SUB srsi(...);

#define basic F77_GLOBAL(basic,BASIC)
F77SUB basic(...);

#define me F77_GLOBAL(me,ME)
F77SUB me(...);

#define cgnr F77_GLOBAL(cgnr,CGNR)
F77SUB cgnr(...);

#define lsqr F77_GLOBAL(lsqr,LSQR)
F77SUB lsqr(...);

#define odir F77_GLOBAL(odir,ODIR)
F77SUB odir(...);

#define omin F77_GLOBAL(omin,OMIN)
F77SUB omin(...);

#define ores F77_GLOBAL(ores,ORES)
F77SUB ores(...);

#define iom F77_GLOBAL(iom,IOM)
F77SUB iom(...);

#define gmres F77_GLOBAL(gmres,GMRES)
F77SUB gmres(...);

#define usymlq F77_GLOBAL(usymlq,USYMLQ)
F77SUB usymlq(...);

#define usymqr F77_GLOBAL(usymqr,USYMQR)
F77SUB usymqr(...);

#define landir F77_GLOBAL(landir,LANDIR)
F77SUB landir(...);

#define lanmin F77_GLOBAL(lanmin,LANMIN)
F77SUB lanmin(...);

#define lanres F77_GLOBAL(lanres,LANRES)
F77SUB lanres(...);

#define cgcr F77_GLOBAL(cgcr,CGCR)
F77SUB cgcr(...);

#define bcgs F77_GLOBAL(bcgs,BCGS)
F77SUB bcgs(...);

//////////////////////// Parameters structures ////////////////////////

struct iparm_t {
    int ntest;      // (default: 2)
    int itmax;      // (default: 100)
    int level;      // (default: 0)
    int nout;       // (default: 6)
    int idgts;      // (default: 0)
    int maxadp;     // (default: 1)
    int minadp;     // (default: 1)
    int iomgad;     // (default: 1)
    int ns1;        // (default: 5)
    int ns2;        // (default: 100000)
    int ns3;        // (default: 0)
    int nstore;     // (default: 2)
    int iscale;     // (default: 0)
    int iperm;      // (default: 0)
    int ifact;      // (default: 1)
    int lvfill;     // (default: 0)
    int ltrunc;     // (default: 0)
    int ipropa;     // (default: 2)
    int kblsz;      // (default: −1)
    int nbl2d;      // (default: −1)
    int ifctv;      // (default: 1)
    int iqlr;       // (default: 1)
    int isymm;      // (default: 2)
    int ielim;      // (default: 0)
    int ndeg;       // (default: 1)
    int __pad[5];   // to total 30 integers

    // Allow to use iparm as an array like in Fortran
    int& operator[](int i) { return *((int*)this + i); }
    const int& operator[](int i) const { return *((int*)this + i); }
};

struct rparm_t {
    double zeta;    // (default: 10e−6)
    double emax;    // (default: 2.0)
    double emin;    // (default: 1.0)
    double ff;      // (default: 0.75)
    double fff;     // (default: 0.75)
    double timit;   // (default: 0.0)
    double digit1;  // (default: 0.0)
    double digit2;  // (default: 0.0)
    double omega;   // (default: 1.0)
    double alphab;  // (default: 0.0)
    double betab;   // (default: 0.25)
    double specr;   // (default: 0.0)
    double timfac;  // (default: 0.0)
    double timtot;  // (default: 0.0)
    double tol;     // (default: 500 ∗ srelpr)
    double ainf;    // (default: 0.0)
    double __pad[14]; // to total 30 doubles

    // Allow to use rparm as an array like in Fortran
    double& operator[](int i) { return *((double*)this + i); }
    const double& operator[](int i) const { return *((double*)this + i); }
};

//////////////////////// Main procedures ////////////////////////

#define nspcg F77_GLOBAL(nspcg,NSPCG)
/**
 * Driver for the nspcg package.
 *
 * \param precon        preconditioning module
 * \param accel         acceleration module
 * \param ndim          row dimension of the \c coef array
 * \param mdim          column dimension of the \c coef array
 * \param n             order of the system
 * \param[inout] maxnz  active column width of the \c coef array
 * \param coef          matrix data array
 * \param jcoef         matrix data array
 * \param p, ip         pivot and inverse pivot information (or certain solvers, these vectors may not be necessary)
 * \param[inout] u      on input the initial guess to the solution, on output the latest estimate to the solution
 * \param ubar          optional input quantity containing the true solution
 * \param rhs           right hand side of the matrix problem
 * \param wksp          real workspace array
 * \param iwksp         integer workspace array
 * \param nw            length of \c wksp
 * \param inw           length of \c iwksp
 * \param[inout] iparm  some integer parameters which affect the method [30]
 * \param[inout] rparm  some real parameters which affect the method [30]
 * \param[out] ier      the error flag
 */
F77SUB nspcg(void (*precon)(...), void (*accel)(...), const int& ndim, const int& mdim, const int& n, int& maxnz,
             double* coef, int* jcoef, int* p, int* ip, double* u, double* ubar, double* rhs,
             double* wksp, int* iwksp, int& nw, int& inw, iparm_t& iparm, rparm_t& rparm, int& ier);

#define dfault F77_GLOBAL(dfault,DFAULT)
/**
 * Sets the default values of \c iparm and \c rparm.
 */
F77SUB dfault(iparm_t& iparm, rparm_t& rparm);
