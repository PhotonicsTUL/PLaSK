#include <type_traits>

#include "femT.h"

namespace plask { namespace solvers { namespace thermal3d {

FiniteElementMethodThermal3DSolver::FiniteElementMethodThermal3DSolver(const std::string& name) :
    SolverWithMesh<Geometry3D, RectilinearMesh3D>(name),
    algorithm(ALGORITHM_ITERATIVE),
    loopno(0),
    inittemp(300.),
    corrlim(0.05),
    corrtype(CORRECTION_ABSOLUTE),
    itererr(1e-8),
    iterlim(10000),
    logfreq(500),
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
            algorithm = source.enumAttribute<Algorithm>("algorithm")
                .value("cholesky", ALGORITHM_CHOLESKY)
                .value("iterative", ALGORITHM_ITERATIVE)
                .get(algorithm);
            itererr = source.getAttribute<double>("itererr", itererr);
            iterlim = source.getAttribute<size_t>("iterlim", iterlim);
            logfreq = source.getAttribute<size_t>("logfreq", logfreq);
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

/**
    * Helper function for applying boundary conditions of element edges to stiffness matrix.
    * Boundary conditions must be set for both nodes at the element edge.
    * \param boundary_conditions boundary conditions holder
    * \param idx indices of the element nodes
    * \param dx, dy, dz dimentions of the element
    * \param[out] F the load vector
    * \param[out] K stiffness matrix
    * \param F_function function returning load vector component
    * \param K_function function returning stiffness matrix component
    */
template <typename ConditionT>
static void setBoundaries(const BoundaryConditionsWithMesh<RectilinearMesh3D,ConditionT>& boundary_conditions,
                          const size_t (&idx)[8], double dx, double dy, double dz, double (&F)[8], double (&K)[8][8],
                          const std::function<double(double,ConditionT,size_t)>& F_function,
                          const std::function<double(double,ConditionT,ConditionT,size_t,size_t,bool)>& K_function
                         )
{
    boost::optional<ConditionT> values[8];
    for (int i = 0; i < 8; ++i) values[i] = boundary_conditions.getValue(idx[i]);

    constexpr int walls[6][4] = { {0,1,2,3}, {4,5,6,7}, {0,2,4,6}, {1,3,5,7}, {0,1,4,5}, {2,3,6,7} };
    const double areas[3] = { dx*dy, dy*dz, dz*dx };

    for (int side = 0; side < 6; ++side) {
        const auto& wall = walls[side];
        if (values[wall[0]] && values[wall[1]] && values[wall[2]] && values[wall[3]]) {
            double area = areas[side/2];
            for (int i = 0; i < 4; ++i) {
                F[i] += F_function(area, *values[wall[i]], wall[i]);
                for (int j = 0; j <= i; ++j) {
                    int ij = i ^ j; // numbers on the single edge differ by one bit only, so detect different bits
                    bool edge = (ij == 1 || ij == 2 || ij == 4);
                    K[i][j] += K_function(area, *values[wall[i]], *values[wall[j]], wall[i], wall[j], edge);
                }
            }
        }

    }
}

template <typename MatrixT>
void FiniteElementMethodThermal3DSolver::setMatrix(MatrixT& A, DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& btemperature,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& bheatflux,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,Convection>& bconvection,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,Radiation>& bradiation
                  )
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, bands=%2%{%3%})", A.size, A.kd+1, A.ld+1);

    auto heats = inHeatDensity(mesh->getMidpointsMesh());

    // zero the matrix and the load vector
    A.clear();
    B.fill(0.);

    // Set stiffness matrix and load vector
    for (auto elem: mesh->elements)
    {
        // nodes numbers for the current element
        size_t idx[8];
        idx[0] = elem.getLoLoLoIndex();          //   z y            6-----7
        idx[1] = elem.getUpLoLoIndex();          //   |/__x         /|    /|
        idx[2] = elem.getLoUpLoIndex();          //                4-----5 |
        idx[3] = elem.getUpUpLoIndex();          //                | 2---|-3
        idx[4] = elem.getLoLoUpIndex();          //                |/    |/
        idx[5] = elem.getUpLoUpIndex();          //                0-----1
        idx[6] = elem.getLoUpUpIndex();          //
        idx[7] = elem.getUpUpUpIndex();          //

        // element size
        double dx = elem.getUpper0() - elem.getLower0();
        double dy = elem.getUpper1() - elem.getLower1();
        double dz = elem.getUpper2() - elem.getLower2();

        // point and material in the middle of the element
        Vec<3> middle = elem.getMidpoint();
        auto material = geometry->getMaterial(middle);

        // average temperature on the element
        double temp = 0.; for (int i = 0; i < 8; ++i) temp += temperatures[idx[i]]; temp *= 0.125;

        // thermal conductivity
        double kx, ky, kz;
        auto leaf = dynamic_pointer_cast<const GeometryObjectD<3>>(geometry->getMatchingAt(middle, &GeometryObject::PredicateIsLeaf));
        if (leaf)
            std::tie(ky,kz) = std::tuple<double,double>(material->thermk(temp, leaf->getBoundingBox().height()));
        else
            std::tie(ky,kz) = std::tuple<double,double>(material->thermk(temp));

        ky *= 1e-6; kz *= 1e-6;                                         // W/m -> W/µm

        kx = ky/dx; kx = ky*dy; kx = ky*dz;
        ky *= dx;   ky /= dy;   ky *= dz;
        kz *= dx;   kz *= dy;   kz /= dz;

        // load vector: heat densities
        double f = 0.125e-18 * dx * dy * dz * heats[elem.getIndex()];   // 1e-18 -> to transform µm³ into m³

        // set symmetric matrix components
        double K[8][8];
        K[0][0] = K[1][1] = K[2][2] = K[3][3] = K[4][4] = K[5][5] = K[6][6] = K[7][7] = (kx + ky + kz) / 9.;

        K[1][0] = K[3][2] = K[5][4] = K[7][6] = (-2.*kx +    ky +    kz) / 18.;
        K[2][0] = K[3][1] = K[6][4] = K[7][5] = (    kx - 2.*ky +    kz) / 18.;
        K[4][0] = K[5][1] = K[6][2] = K[7][3] = (    kx +    ky - 2.*kz) / 18.;

        K[4][2] = K[5][3] = K[6][0] = K[7][1] = (    kx - 2.*ky - 2.*kz) / 36.;
        K[4][1] = K[5][0] = K[6][3] = K[7][2] = (-2.*kx +    ky - 2.*kz) / 36.;
        K[2][1] = K[3][0] = K[6][5] = K[7][4] = (-2.*kx - 2.*ky +    kz) / 36.;

        K[4][3] = K[5][2] = K[6][1] = K[7][0] = -(kx + ky + kz) / 36.;

        double F[8];
        std::fill_n(F, 8, f);

        // boundary conditions: heat flux
        setBoundaries<double>(bheatflux, idx, dx, dy, dz, F, K,
                              [](double area, double value, size_t) { // F
                                  return - 0.25e-12 * area * value;
                              },
                              [](double, double, double, size_t, size_t, bool) { return 0.; }  // K
                             );

        // boundary conditions: convection
        setBoundaries<Convection>(bconvection, idx, dx, dy, dz, F, K,
                                  [](double area, Convection value, size_t) { // F
                                      return 0.25e-12 * area * value.coeff * value.ambient;
                                  },
                                  [](double area, Convection value1, Convection value2, size_t i1, size_t i2, bool edge) -> double { // K
                                      double v = 0.125e-12 * area * (value1.coeff + value2.coeff);
                                      return v / (i2==i1? 9. : edge? 18. : 36.);
                                  }
                                 );

        // boundary conditions: radiation
        setBoundaries<Radiation>(bradiation, idx, dx, dy, dz, F, K,
                                 [this](double area, Radiation value, size_t i) -> double { // F
                                     double a = value.ambient; a = a*a;
                                     double T = this->temperatures[i]; T = T*T;
                                     return - 0.25e-12 * area * value.emissivity * phys::SB * (T*T - a*a);},
                                 [](double, Radiation, Radiation, size_t, size_t, bool) {return 0.;} // K
                                );

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j <= i; ++j) {
                A(idx[i],idx[j]) += K[i][j];
            }
            B[idx[i]] += F[i];
        }
    }

    applyBC(A, B, btemperature);
}

template <>
void FiniteElementMethodThermal3DSolver::applyBC<DpbMatrix>(DpbMatrix& A, DataVector<double>& B,
                                                            const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& btemperature) {
    // boundary conditions of the first kind
    for (auto cond: btemperature) {
        for (auto r: cond.place) {
            A(r,r) = 1.;
            register double val = B[r] = cond.value;
            size_t start = (r > A.kd)? r-A.kd : 0;
            size_t end = (r + A.kd < A.size)? r+A.kd+1 : A.size;
            for(size_t c = start; c < r; ++c) {
                B[c] -= A(r,c) * val;
                A(r,c) = 0.;
            }
            for(size_t c = r+1; c < end; ++c) {
                B[c] -= A(r,c) * val;
                A(r,c) = 0.;
            }
        }
    }
}

template <>
void FiniteElementMethodThermal3DSolver::applyBC<SparseBandMatrix>(SparseBandMatrix& A, DataVector<double>& B,
                                                                   const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& btemperature) {
    // boundary conditions of the first kind
    for (auto cond: btemperature) {
        for (auto r: cond.place) {
            double* rdata = A.data + LDA*r;
            *rdata = 1.;
            register double val = B[r] = cond.value;
            // below diagonal
            for (register ptrdiff_t i = 13; i > 0; --i) {
                register ptrdiff_t c = r - A.bno[i];
                if (c >= 0) {
                    B[c] -= A.data[LDA*c+i] * val;
                    A.data[LDA*c+i] = 0.;
                }
            }
            // above diagonal
            for (register ptrdiff_t i = 1; i < 14; ++i) {
                register ptrdiff_t c = r + A.bno[i];
                if (c < A.size) {
                    B[c] -= rdata[i] * val;
                    rdata[i] = 0.;
                }
            }
        }
    }
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

    MatrixT A(size, mesh->mediumAxis().size()*mesh->minorAxis().size(), mesh->minorAxis().size());

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
        case ALGORITHM_CHOLESKY: return doCompute<DpbMatrix>(loops);
        case ALGORITHM_ITERATIVE: return doCompute<SparseBandMatrix>(loops);
    }
    return 0.;
}



void FiniteElementMethodThermal3DSolver::solveMatrix(DpbMatrix& A, DataVector<double>& B)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    int info = 0;

    // Factorize matrix
    dpbtrf(UPLO, A.size, A.kd, A.data, A.ld+1, info);
    if (info < 0)
        throw CriticalException("%1%: Argument %2% of dpbtrf has illegal value", this->getId(), -info);
    if (info > 0)
        throw ComputationError(this->getId(), "Leading minor of order %1% of the stiffness matrix is not positive-definite", info);

    // Find solutions
    dpbtrs(UPLO, A.size, A.kd, 1, A.data, A.ld+1, B.data(), B.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", this->getId(), -info);

    // now A contains factorized matrix and B the solutions
}


void FiniteElementMethodThermal3DSolver::solveMatrix(SparseBandMatrix& A, DataVector<double>& B)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    PrecondJacobi precond(A);

    DataVector<double> X = temperatures.copy(); // We use previous temperatures as initial solution
    double err;
    try {
        int iter = solveDCG(A, precond, X.data(), B.data(), err, iterlim, itererr, logfreq, this->getId());
        this->writelog(LOG_DETAIL, "Conjugate gradient converged after %1% iterations.", iter);
    } catch (DCGError err) {
        throw ComputationError(this->getId(), "Conjugate gradient failed:, %1%", err.what());
    }

    B = X;

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
        auto material = this->geometry->getMaterial(midpoint);

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
        auto leaf = dynamic_pointer_cast<const GeometryObjectD<3>>(geometry->getMatchingAt(midpoint, &GeometryObject::PredicateIsLeaf));
        if (leaf)
            std::tie(kxy,kz) = std::tuple<double,double>(material->thermk(temp, leaf->getBoundingBox().height()));
        else
            std::tie(kxy,kz) = std::tuple<double,double>(material->thermk(temp));

        fluxes[el.getIndex()] = vec(
            - 0.25e6 * kxy * (- temperatures[lll] - temperatures[llu] - temperatures[lul] - temperatures[luu]
                              + temperatures[ull] + temperatures[ulu] + temperatures[uul] + temperatures[uuu])
                / (el.getUpper0() - el.getLower0()), // 1e6 - from µm to m
            - 0.25e6 * kxy * (- temperatures[lll] - temperatures[llu] + temperatures[lul] + temperatures[luu]
                              - temperatures[ull] - temperatures[ulu] + temperatures[uul] + temperatures[uuu])
                / (el.getUpper1() - el.getLower1()), // 1e6 - from µm to m
            - 0.25e6 * kz  * (- temperatures[lll] + temperatures[llu] - temperatures[lul] + temperatures[luu]
                              - temperatures[ull] + temperatures[ulu] - temperatures[uul] + temperatures[uuu])
                / (el.getUpper2() - el.getLower2()) // 1e6 - from µm to m
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
