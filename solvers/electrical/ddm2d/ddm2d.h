#ifndef PLASK__MODULE_ELECTRICAL_DDM_H
#define PLASK__MODULE_ELECTRICAL_DDM_H

#include <plask/plask.hpp>
#include "element2D.h"
#include "layer2D.h"
#include "Lapack/lapack.h"
#include <fstream>
#include <sstream>

namespace plask { namespace solvers { namespace electrical {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2DType>
struct PLASK_SOLVER_API DriftDiffusionModel2DSolver: public SolverWithMesh<Geometry2DType, RectangularMesh<2>> {

  protected:

    int solvePart(std::vector<double> &ipAA, std::vector<double> &ipDD, std::vector<double> &ipYY, const std::string& iType, int iLoopLim, double iCorrLim);
    int solveEquationAYD(std::vector<double> &ipAA, std::vector<double> &ipDD, std::vector<double> &ipYY, int n);
    int setSolverData(std::vector<double> &ipAA, std::vector<double> &ipDD, std::vector<double> &ipYY, const std::string& iType); //Set Solver Data for Calculation Type (T,Psi,Fn,Fp) // 1D2D
    void updVoltNodes(double iVolt); //Update Nodes (T,Psi,Fn,Fp Values) Setting Voltage Step
    double updNodes(std::vector<double> &iDeltaVal, const std::string& iType);
    void updElements(); //Update Elements N, P, Ionized ND And NA Values

    bool mRsrh; // SRH recombination is taken into account
    bool mRrad; // radiative recombination is taken into account
    bool mRaug; // Auger recombination is taken into account
    bool mPol; // polarization (GaN is the substrate)
    bool mFullIon; // dopant ionization = 100%
    std::string mFnFpCalc; // "one"/"many"
    std::string fnStructure; // file with structure
    std::vector<double> vX; // vector of mesh x-points
    std::vector<double> vY; // vector of mesh x-points
    std::vector<Node2D> vN; // vector of nodes
    std::vector<Elem2D> vE; // vector of elements
    std::vector<Layer2D> vL; // vector of layers
    int nx, ny, nn, ne, nl; // number of nodes/elements/layers
    double PsiP; // normalised e*psi at p-contact at U=0V

    void setStructure(); /// set structure
    void setScaleParam(); /// set scalling parameters
    int setMeshPoints(); /// set mesh
    int setNodes(); /// set nodes
    char checkPos(double iX); /// check position of node (boundary/interface)
    int setElements(); /// set elements
    Layer2D* checkLay(double iX); /// check layer for a given element
    void calcPsiI(); /// calculate initial potential
    int solve(); /// solve AX=B
    int saveResJ(double iU); /// calculate Jn and Jp (without values at the interfaces)

    double findPsiI(double iEc0, double iEv0, double iNc, double iNv, double iNd, double iNa, double iEd, double iEa, double iFnEta, double iFpKsi, double iT); /// find initial potential
    double calcN(double iNc, double iFnEta, double iPsi, double iEc0, double iT); /// calculate electron concentration
    double calcP(double iNv, double iFpKsi, double iPsi, double iEv0, double iT); /// calculate hole concentration

    /// Fermi-Dirac integral of grade 1/2
    double calcFD12(double iEta);

    double scaleE, scaleX, scaleT, scaleN, scaleEpsR, scaleK, scaleMi, scaleJ, scaleR, scalet, scaleB, scaleC, scaleH;

/*    /// scale coordinate (x)
    double scaleX(double iX);
    /// rescale coordinate (x)
    double rescaleX(double iX);
    /// scale temperature (T)
    double scaleT(double iT);
    /// rescale temperature (T)
    double rescaleT(double iT);
    /// scale energy (Fn, Fp, Eg, Chi)
    double scaleE(double iE);
    /// rescale energy (Fn, Fp, Eg, Chi)
    double rescaleE(double iE);
    /// scale concentration (n, p, Nd, Na, Nc, Nv)
    double scaleN(double iN);
    /// rescale concentration (n, p, Nd, Na, Nc, Nv)
    double rescaleN(double iN);
    /// scale thermal conductivity (K)
    double scaleK(double iK);
    /// scale dielectric constant (epsR)
    double scaleEpsR(double iEpsR);
    /// rescale dielectric constant (epsR)
    double rescaleEpsR(double iEpsR);
    /// scale mobility (miN, miP)
    double scaleMi(double iEpsR);
    /// rescale mobility (miN, miP)
    double rescaleMi(double iEpsR);
    /// rescale current density (Jn, Jp)
    double rescaleJ(double iJ); // get J (kA/cm^2)
    /// scale SRH recombination lifetime (Tn, Tp)
    double scaleTime(double iTime);
    /// scale radiative recombination coefficient (B)
    double scaleB(double iB);
    /// scale Auger recombination coefficient (Cn, Cp)
    double scaleC(double iC);
    /// scale heat source (H)
    double scaleH(double iH);
    /// rescale recombination rate (Rsrh, Rrad, Raug)
    double rescaleR(double iR); // get R (1/(cm^3*s))
    /// scale polarization (Psp, Ppz, P)
    double scaleP(double iP);*/
    /// are equal?
    bool areEq(double iVal1, double iVal2, double iTol);


    /// Initialize the solver
    virtual void onInitialize() override;

    /// Invalidate the data
    virtual void onInvalidate() override;

    const double
        accT = 1e-6,			    // accuracy for temperature calculations (K) (see: functions.h)
        accPsiI = 1e-12,			// accuracy for initial potential calculations (eV) (see: functions.h)
        accPsi0 = 1e-12,			// accuracy for potential calculations (eV) (see: functions.h) dPsi0/Psi0_old
        accPsi = 1e-6,			    // accuracy for potential calculations (eV) dPsi/Psi_old
        accFnFp = 1e-6,			    // accuracy for quasi-Fermi levels calculations (eV) dFnEta/FnEta_old dFpKsi/FpKsi_old
        bigNum = 1e15,              // big number (when boundary conditions are taken into account)
        dxMin = 1e-5,               // x-step at structure edge or interface (um)
        dxMax = 0.0001,             // acceptable x-step in the structure (um)
        dxDiv = 1.1,                // used in mesh refining (-)
        dxEq = 0.0001,              // x-step for identical x-step lengths (um) /def: 0.0001/
        dyEq = 0.0001,              // x-step for identical x-step lengths (um) /def: 0.0001/
        dxAcc = 1e-7,               // below this value x-values are the same (um)
        dyAcc = 1e-7,               // below this value y-values are the same (um)
        U0 = 0.,                    // initial voltage (V)
        dU = 0.002,                 // voltage step (V)
        Umax = 1.000,               // maximal voltage (V)
        maxcorrT = 100.,            // maximal correction for temperature calculations (K) (100)
        maxcorrPsi0 = 10.*dU,       // maximal correction for initial potential calculations (eV) (10 dU)
        maxcorrPsi = 0.1*dU,        // maximal correction for potential calculations (eV) (0.1 dU)
        maxcorrFnFp = 1e20*dU;      // maximal correction for quasi-Fermi levels calculations (eV) (1e20 dU)

    const int
        loopT = 100,		        // number of loops for temperature calculations (see: functions.h) (10)
        loopPsiI = 10000,		    // number of loops for initial potential calculations (see: functions.h) (10000)
        loopPsi0 = 2000,		    // number of loops for potential calculations (2000)
        loopPsi = 3,			    // number of loops for potential calculations (3)
        loopFnFp = 3,			    // number of loops for quasi-Fermi levels calculations (3)
        loopPsiFnFp = 20,			// number of loops for potential-quasi-Fermi levels calculations (20)
        dUsav = 1,                  // how often save results
        mshgen = 1;                 // mesh generator

    const std::string
        stat = "MB";                // Maxwell-Boltzmann "MB" / Fermi-Dirac "FD"

    const double
        T = 300.,                                                                   // ambient temperature (K)
        EpsR = 10.,                                                                 // maximal dielectric constant (-)
        N = 1e18,                                                                   // maximal doping concentration (1/cm^3)
        x = sqrt((phys::epsilon0*phys::kB_J*T*EpsR)/(phys::qe*phys::qe*N))*1e3,       // sometimes denoted as LD (um)
        Mi = 1000.,                                                                 // maximal mobility (cm^2/Vs)
        Jx = ((phys::kB_J*N)*T*Mi/x)*10.,                                           // current density parameter (kA/cm2)
        Rx = ((phys::kB_J*T*Mi*N)/(phys::qe*x*x))*1e8,                               // recombination parameter (1/(cm^3*s))
        Time = N/Rx,                                                                // SRH recombination lifetime (s)
        Bx = Rx/(N*N),                                                              // radiative recombination coefficient (cm^3/s)
        Cx = Rx/(N*N*N),                                                            // Auger recombination coefficient (cm^6/s)
        K = 150.,                                                                   // thermal conductivity (W/mK)
        Px = x*phys::qe*N,                                                           // polarization (C/m^2)
        Hx = ((K*T)/(x*x))*1e12;                                                    // heat source (W/(m^3))

  public:

    typename ProviderFor<Potential, Geometry2DType>::Delegate outPotential;    
    
    ReceiverFor<Temperature, Geometry2DType> inTemperature;
    
    void loadFile(std::string filename); /// load file with structure
    
    double compute(); /// run calculations
    
    void saveResN(std::string filename); /// save results for nodes
    
    void saveResE(std::string filename); /// save results for nodes
    
    void saveResP(std::string filename); /// save polarizations

    virtual void loadConfiguration(XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    DriftDiffusionModel2DSolver(const std::string& name="");

    virtual std::string getClassName() const override;

    ~DriftDiffusionModel2DSolver();
};

}} //namespaces

} // namespace plask

#endif

