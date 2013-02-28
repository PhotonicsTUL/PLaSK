#include <type_traits>

#include "femT.h"

#include "band_matrix.h"
#include "dcg.h"

// LAPACK routines to solve set of linear equations
#define dpbtrf F77_GLOBAL(dpbtrf,DPBTRF)
F77SUB dpbtrf(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, int& info);

#define dpbtf2 F77_GLOBAL(dpbtf2,DPBTF2)
F77SUB dpbtf2(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, int& info);

#define dpbtrs F77_GLOBAL(dpbtrs,DPBTRS)
F77SUB dpbtrs(const char& uplo, const int& n, const int& kd, const int& nrhs, double* ab, const int& ldab, double* b, const int& ldb, int& info);


namespace plask { namespace solvers { namespace thermal3d {

FiniteElementMethodThermal3DSolver::FiniteElementMethodThermal3DSolver(const std::string& name) :
    SolverWithMesh<Geometry3D, RectilinearMesh3D>(name),
    algorithm(ALGORITHM_BLOCK),
    loopno(0),
    bignum(1e15),
    inittemp(300.),
    corrlim(0.05),
    corrtype(CORRECTION_ABSOLUTE),
    outTemperature(this, &FiniteElementMethodThermal3DSolver::getTemperatures),
    outHeatFlux(this, &FiniteElementMethodThermal3DSolver::getHeatFluxes)
{
    temperatures.reset();
    fluxes.reset();
    inHeatDensity = 0.;
}


FiniteElementMethodThermal3DSolver::~FiniteElementMethodThermal3DSolver() {
}


void FiniteElementMethodThermal3DSolver::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "temperature")
            this->readBoundaryConditions(manager, source, temperature_boundary);

        else if (param == "heatflux")
            this->readBoundaryConditions(manager, source, heatflux_boundary);

        else if (param == "convection")
            this->readBoundaryConditions(manager, source, convection_boundary);

        else if (param == "radiation")
            this->readBoundaryConditions(manager, source, radiation_boundary);

        else if (param == "loop") {
            inittemp = source.getAttribute<double>("inittemp", inittemp);
            corrlim = source.getAttribute<double>("corrlim", corrlim);
            corrtype = source.enumAttribute<CorrectionType>("corrtype")
                .value("absolute", CORRECTION_ABSOLUTE, 3)
                .value("relative", CORRECTION_RELATIVE, 3)
                .get(corrtype);
            source.requireTagEnd();
        }

        else if (param == "matrix") {
            bignum = source.getAttribute<double>("bignum", bignum);
            algorithm = source.enumAttribute<Algorithm>("algorithm")
                .value("block", ALGORITHM_BLOCK)
                .value("iterative", ALGORITHM_ITERATIVE)
                .get(algorithm);
            source.requireTagEnd();
        } else
            this->parseStandardConfiguration(source, manager);
    }
}


void FiniteElementMethodThermal3DSolver::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    loopno = 0;
    temperatures.reset(this->mesh->size(), inittemp);
}


void FiniteElementMethodThermal3DSolver::onInvalidate() {
    temperatures.reset();
    fluxes.reset();
}

void FiniteElementMethodThermal3DSolver::setAlgorithm(Algorithm alg) {
    //TODO
    algorithm = alg;
}

// /**
//     * Helper function for applying boundary conditions of element edges to stiffness matrix.
//     * Boundary conditions must be set for both nodes at the element edge.
//     * \param boundary_conditions boundary conditions holder
//     * \param i1, i2, i3, i4 indices of the lower left, lower right, upper right, and upper left node
//     * \param width width of the element
//     * \param height height of the element
//     * \param[out] F1, F2, F3, F4 references to the load vector components
//     * \param[out] K11, K22, K33, K44, K12, K14, K24, K34 references to the stiffness matrix components
//     * \param F_function function returning load vector component
//     * \param Kmm_function function returning stiffness matrix diagonal component
//     * \param Kmn_function function returning stiffness matrix off-diagonal component
//     */
// template <typename ConditionT>
// static void setBoundaries(const BoundaryConditionsWithMesh<RectilinearMesh2D,ConditionT>& boundary_conditions,
//                           size_t i1, size_t i2, size_t i3, size_t i4, double width, double height,
//                           double& F1, double& F2, double& F3, double& F4,
//                           double& K11, double& K22, double& K33, double& K44,
//                           double& K12, double& K23, double& K34, double& K41,
//                           const std::function<double(double,ConditionT,ConditionT,size_t,size_t)>& F_function,
//                           const std::function<double(double,ConditionT,ConditionT,size_t,size_t)>& Kmm_function,
//                           const std::function<double(double,ConditionT,ConditionT,size_t,size_t)>& Kmn_function
//                          )
// {
//     auto val1 = boundary_conditions.getValue(i1);
//     auto val2 = boundary_conditions.getValue(i2);
//     auto val3 = boundary_conditions.getValue(i3);
//     auto val4 = boundary_conditions.getValue(i4);
//     if (val1 && val2) { // bottom
//         F1 += F_function(width, *val1, *val2, i1, i2); F2 += F_function(width, *val2, *val1, i2, i1);
//         K11 += Kmm_function(width, *val1, *val2, i1, i2); K22 += Kmm_function(width, *val2, *val1, i2, i1);
//         K12 += Kmn_function(width, *val1, *val2, i1, i2);
//     }
//     if (val2 && val3) { // right
//         F2 += F_function(height, *val2, *val3, i2, i3); F3 += F_function(height, *val3, *val2, i3, i2);
//         K22 += Kmm_function(height, *val2, *val3, i2, i3); K33 += Kmm_function(height, *val3, *val2, i3, i2);
//         K23 += Kmn_function(height, *val2, *val3, i2, i3);
//     }
//     if (val3 && val4) { // top
//         F3 += F_function(width, *val3, *val4, i3, i4); F4 += F_function(width, *val4, *val3, i4, i3);
//         K33 += Kmm_function(width, *val3, *val4, i3, i4); K44 += Kmm_function(width, *val4, *val3, i4, i3);
//         K34 += Kmn_function(width, *val3, *val4, i3, i4);
//     }
//     if (val4 && val1) { // left
//         F1 += F_function(height, *val1, *val4, i1, i4); F4 += F_function(height, *val4, *val1, i4, i1);
//         K11 += Kmm_function(height, *val1, *val4, i1, i4); K44 += Kmm_function(height, *val4, *val1, i4, i1);
//         K41 += Kmn_function(height, *val1, *val4, i1, i4);
//     }
// }

template <typename MatrixT>
void FiniteElementMethodThermal3DSolver::setMatrix(MatrixT& A, DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& btemperature,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& bheatflux,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,Convection>& bconvection,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,Radiation>& bradiation
                  )
{
//     this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, band=%2%)", mAOrder, mABand);
//
//     auto iMesh = (this->mesh)->getMidpointsMesh();
//     auto tHeatDensities = inHeatDensity(iMesh);
//
//     std::fill_n(oA.data, mABand*mAOrder, 0.); // zero the matrix
//     oLoad.fill(0.);
//
//     std::vector<Box2D> tVecBox = geometry->getLeafsBoundingBoxes();
//
//     // Set stiffness matrix and load vector
//     for (auto tE: this->mesh->elements)
//     {
//         // nodes numbers for the current element
//         size_t ll = tE.getLoLoIndex();
//         size_t lr = tE.getUpLoIndex();
//         size_t ul = tE.getLoUpIndex();
//         size_t ur = tE.getUpUpIndex();
//
//         // element size
//         double tElemWidth = tE.getUpper0() - tE.getLower0();
//         double tElemHeight = tE.getUpper1() - tE.getLower1();
//
//         // point and material in the middle of the element
//         Vec<2,double> tMidPoint = tE.getMidpoint();
//         auto tMaterial = geometry->getMaterial(tMidPoint);
//
//         // average temperature on the element
//         double tTemp = 0.25 * (temperatures[ll] + temperatures[lr] + temperatures[ul] + temperatures[ur]);
//
//         // thermal conductivity
//         double tKx, tKy;
//         auto tLeaf = dynamic_pointer_cast<const GeometryObjectD<2>>(geometry->getMatchingAt(tMidPoint, &GeometryObject::PredicateIsLeaf));
//         if (tLeaf)
//             std::tie(tKx,tKy) = std::tuple<double,double>(tMaterial->thermk(tTemp, tLeaf->getBoundingBox().height()));
//         else
//             std::tie(tKx,tKy) = std::tuple<double,double>(tMaterial->thermk(tTemp));
//
//         tKx *= tElemHeight; tKx /= tElemWidth;
//         tKy *= tElemWidth; tKy /= tElemHeight;
//
//         // load vector: heat densities
//         double tF = 0.25e-12 * tElemWidth * tElemHeight * tHeatDensities[tE.getIndex()]; // 1e-12 -> to transform µm² into m²
//
//         // set symmetric matrix components
//         double tK44, tK33, tK22, tK11, tK43, tK21, tK42, tK31, tK32, tK41;
//
//         tK44 = tK33 = tK22 = tK11 = (tKx + tKy) / 3.;
//         tK43 = tK21 = (-2. * tKx + tKy) / 6.;
//         tK42 = tK31 = - (tKx + tKy) / 6.;
//         tK32 = tK41 = (tKx - 2. * tKy) / 6.;
//
//         double tF1 = tF, tF2 = tF, tF3 = tF, tF4 = tF;
//
//         // boundary conditions: heat flux
//         setBoundaries<double>(iHFConst, ll, lr, ur, ul, tElemWidth, tElemHeight,
//                       tF1, tF2, tF3, tF4, tK11, tK22, tK33, tK44, tK21, tK32, tK43, tK41,
//                       [](double len, double val, double, size_t, size_t) { // F
//                           return - 0.5e-6 * len * val;
//                       },
//                       [](double,double,double,size_t,size_t){return 0.;}, // K diagonal
//                       [](double,double,double,size_t,size_t){return 0.;}  // K off-diagonal
//                      );
//
//         // boundary conditions: convection
//         setBoundaries<Convection>(iConvection, ll, lr, ur, ul, tElemWidth, tElemHeight,
//                       tF1, tF2, tF3, tF4, tK11, tK22, tK33, tK44, tK21, tK32, tK43, tK41,
//                       [](double len, Convection val, Convection, size_t, size_t) { // F
//                           return 0.5e-6 * len * val.mConvCoeff * val.mTAmb1;
//                       },
//                       [](double len, Convection val1, Convection val2, size_t, size_t) { // K diagonal
//                           return (val1.mConvCoeff + val2.mConvCoeff) * len / 3.;
//                       },
//                       [](double len, Convection val1, Convection val2, size_t, size_t) { // K off-diagonal
//                           return (val1.mConvCoeff + val2.mConvCoeff) * len / 6.;
//                       }
//                      );
//
//         // boundary conditions: radiation
//         setBoundaries<Radiation>(iRadiation, ll, lr, ur, ul, tElemWidth, tElemHeight,
//                       tF1, tF2, tF3, tF4, tK11, tK22, tK33, tK44, tK21, tK32, tK43, tK41,
//                       [this](double len, Radiation val, Radiation, size_t i, size_t) -> double { // F
//                           double a = val.mTAmb2; a = a*a;
//                           double T = this->temperatures[i]; T = T*T;
//                           return - 0.5e-6 * len * val.mSurfEmiss * phys::SB * (T*T - a*a);},
//                       [](double,Radiation,Radiation,size_t,size_t){return 0.;}, // K diagonal
//                       [](double,Radiation,Radiation,size_t,size_t){return 0.;}  // K off-diagonal
//                      );
//
//         // set stiffness matrix
//         oA(ll, ll) += tK11;
//         oA(lr, lr) += tK22;
//         oA(ur, ur) += tK33;
//         oA(ul, ul) += tK44;
//
//         oA(lr, ll) += tK21;
//         oA(ur, ll) += tK31;
//         oA(ul, ll) += tK41;
//         oA(ur, lr) += tK32;
//         oA(ul, lr) += tK42;
//         oA(ul, ur) += tK43;
//
//         // set load vector
//         oLoad[ll] += tF1;
//         oLoad[lr] += tF2;
//         oLoad[ur] += tF3;
//         oLoad[ul] += tF4;
//     }
//
//     // boundary conditions of the first kind
//     for (auto tCond: iTConst) {
//         for (auto tIndex: tCond.place) {
//             oA(tIndex, tIndex) += mBigNum;
//             oLoad[tIndex] += tCond.value * mBigNum;
//         }
//     }
//
// #ifndef NDEBUG
//     double* tAend = oA.data + oA.size * oA.band1;
//     for (double* pa = oA.data; pa != tAend; ++pa) {
//         if (isnan(*pa) || isinf(*pa))
//             throw ComputationError(this->getId(), "Error in stiffness matrix at position %1%", pa-oA.data);
//     }
// #endif
//
}

template <typename MatrixT>
double FiniteElementMethodThermal3DSolver::doCompute(int loops)
{
    this->initCalculation();

    fluxes.reset();

    // store boundary conditions for current mesh
    auto btemperature = temperature_boundary(this->mesh);
    auto bheatflux = heatflux_boundary(this->mesh);
    auto bconvection = convection_boundary(this->mesh);
    auto bradiation = radiation_boundary(this->mesh);

    this->writelog(LOG_INFO, "Running thermal calculations");

    int loop = 0;
    size_t size = mesh->size();

    MatrixT A(size, mesh->mediumAxis().size(), mesh->minorAxis().size());

    double max_abscorr = 0.,
           max_relcorr = 0.;

#   ifndef NDEBUG
        if (!temperatures.unique()) this->writelog(LOG_DEBUG, "Temperature data held by something else...");
#   endif
    temperatures = temperatures.claim();
    DataVector<double> T(size);

    do {
        setMatrix(A, T, btemperature, bheatflux, bconvection, bradiation);
        solveMatrix(A, T);

        saveTemperatures(T);

        if (abscorr > max_abscorr) max_abscorr = abscorr;
        if (relcorr > max_relcorr) max_relcorr = relcorr;

        ++loopno;
        ++loop;

        // show max correction
        this->writelog(LOG_RESULT, "Loop %d(%d): max(T)=%.3fK, update=%.3fK(%.3f%%)", loop, loopno, maxT, abscorr, relcorr);

    } while (((corrtype == CORRECTION_ABSOLUTE)? (abscorr > corrlim) : (relcorr > corrlim)) && (loops == 0 || loop < loops));

    outTemperature.fireChanged();
    outHeatFlux.fireChanged();

    // Make sure we store the maximum encountered values, not just the last ones
    // (so, this will indicate if the results changed since the last run, not since the last loop iteration)
    abscorr = max_abscorr;
    relcorr = max_relcorr;

    if (corrtype == CORRECTION_RELATIVE) return relcorr;
    else return abscorr;
}


double FiniteElementMethodThermal3DSolver::compute(int loops) {
    switch (algorithm) {
        case ALGORITHM_BLOCK: return doCompute<DpbMatrix>(loops);
        case ALGORITHM_ITERATIVE: return doCompute<SparseBandMatrix>(loops);
    }
}



void FiniteElementMethodThermal3DSolver::solveMatrix(DpbMatrix& A, DataVector<double>& B)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    int info = 0;

    // Factorize matrix
    dpbtrf(UPLO, A.size, A.band1, A.data, A.band1+1, info);
    if (info < 0)
        throw CriticalException("%1%: Argument %2% of dpbtrf has illegal value", this->getId(), -info);
    else if (info > 0)
        throw ComputationError(this->getId(), "Leading minor of order %1% of the stiffness matrix is not positive-definite", info);

    // Find solutions
    dpbtrs(UPLO, A.size, A.band1, 1, A.data, A.band1+1, B.data(), B.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", this->getId(), -info);

    // now A contains factorized matrix and B the solutions
}


void FiniteElementMethodThermal3DSolver::solveMatrix(SparseBandMatrix& A, DataVector<double>& B)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    Atimes atimes(A);
    MsolveJacobi msolve(A);
    DataVector<double> oX = temperatures.copy(); // We use previous temperatures as initial solution TODO use better error estimation
    double err;
    try {
        int iter = solveDCG(A.size, atimes, msolve, oX.data(), B.data(), err); //TODO add parameters for tolerance and maximum iterations
        this->writelog(LOG_DETAIL, "Conjugate gradient converged after %1% iterations.", iter);
    } catch (DCGError err) {
        throw ComputationError(this->getId(), "Conjugate gradient failed:, %1%", err.what());
    }
    B = oX;

    // now A contains factorized matrix and B the solutions
}


void FiniteElementMethodThermal3DSolver::saveTemperatures(DataVector<double>& T)
{
    abscorr = 0.;
    relcorr = 0.;

    maxT = 0.;

    for (auto oldT = temperatures.begin(), newT = T.begin(); newT != T.end(); ++oldT, ++newT)
    {
        double acor = std::abs(*newT - *oldT); // for boundary with constant temperature this will be zero anyway
        double rcor = acor / *newT;
        if (acor > abscorr) abscorr = acor;
        if (rcor > relcorr) relcorr = rcor;
        if (*newT > maxT) maxT = *newT;
    }
    relcorr *= 100.; // %
    std::swap(temperatures, T);
}


void FiniteElementMethodThermal3DSolver::saveHeatFluxes()
{
    this->writelog(LOG_DETAIL, "Computing heat fluxes");

    fluxes.reset(this->mesh->elements.size());

    for (auto el: this->mesh->elements)
    {
        Vec<3,double> midpoint = el.getMidpoint();
        auto tMaterial = this->geometry->getMaterial(midpoint);

        size_t lll = el.getLoLoLoIndex();
        size_t llu = el.getLoLoUpIndex();
        size_t lul = el.getLoUpLoIndex();
        size_t luu = el.getLoUpUpIndex();
        size_t ull = el.getUpLoLoIndex();
        size_t ulu = el.getUpLoUpIndex();
        size_t uul = el.getUpUpLoIndex();
        size_t uuu = el.getUpUpUpIndex();

        double temp = 0.125 * (temperatures[lll] + temperatures[llu] + temperatures[lul] + temperatures[luu] +
                               temperatures[ull] + temperatures[ulu] + temperatures[uul] + temperatures[uuu]);

        double kxy, kz;
        std::tie(kxy,kz) = std::tuple<double,double>(tMaterial->thermk(temp));
        //TODO add dependence on layer thickness

        fluxes[el.getIndex()] = vec(
            - 0.25e6 * kxy * (- temperatures[lll] - temperatures[llu] - temperatures[lul] - temperatures[luu]
                              + temperatures[ull] + temperatures[ulu] + temperatures[uul] + temperatures[uuu])
                / (el.getUpper0() - el.getLower0()), // 1e6 - from um to m
            - 0.25e6 * kxy * (- temperatures[lll] - temperatures[llu] + temperatures[lul] + temperatures[luu]
                              - temperatures[ull] - temperatures[ulu] + temperatures[uul] + temperatures[uuu])
                / (el.getUpper1() - el.getLower1()), // 1e6 - from um to m
            - 0.25e6 * kz  * (- temperatures[lll] + temperatures[llu] - temperatures[lul] + temperatures[luu]
                              - temperatures[ull] + temperatures[ulu] - temperatures[uul] + temperatures[uuu])
                / (el.getUpper2() - el.getLower2()) // 1e6 - from um to m
        );
    }
}


DataVector<const double> FiniteElementMethodThermal3DSolver::getTemperatures(const MeshD<3>& dst_mesh, InterpolationMethod method) const {
    this->writelog(LOG_DETAIL, "Getting temperatures");
    if (!temperatures) return DataVector<const double>(dst_mesh.size(), inittemp); // in case the receiver is connected and no temperature calculated yet
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    return interpolate(*(this->mesh), temperatures, WrappedMesh<3>(dst_mesh, this->geometry), method);
}


DataVector<const Vec<3> > FiniteElementMethodThermal3DSolver::getHeatFluxes(const MeshD<3>& dst_mesh, InterpolationMethod method) {
    this->writelog(LOG_DETAIL, "Getting heat fluxes");
    if (!temperatures) return DataVector<const Vec<3>>(dst_mesh.size(), Vec<3>(0.,0.,0.)); // in case the receiver is connected and no fluxes calculated yet
    if (!fluxes) saveHeatFluxes(); // we will compute fluxes only if they are needed
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    return interpolate(*((this->mesh)->getMidpointsMesh()), fluxes, WrappedMesh<3>(dst_mesh, this->geometry), method);
}

}}} // namespace plask::solvers::thermal
