#include "femT3d.h"

namespace plask { namespace solvers { namespace thermal {

const double BIG = 1e16;

FiniteElementMethodDynamicThermal3DSolver::FiniteElementMethodDynamicThermal3DSolver(const std::string& name) :
    SolverWithMesh<Geometry3D, RectangularMesh<3>>(name),
    loopno(0),
    outTemperature(this, &FiniteElementMethodDynamicThermal3DSolver::getTemperatures),
    outHeatFlux(this, &FiniteElementMethodDynamicThermal3DSolver::getHeatFluxes),
    outThermalConductivity(this, &FiniteElementMethodDynamicThermal3DSolver::getThermalConductivity),
    algorithm(ALGORITHM_CHOLESKY),
    inittemp(300.),    
    methodparam(0.5),
    timestep(0.1),
    lumping(true),
    rebuildfreq(1),
    logfreq(25)
{
    temperatures.reset();
    mHeatFluxes.reset();
    inHeat = 0.;
}


FiniteElementMethodDynamicThermal3DSolver::~FiniteElementMethodDynamicThermal3DSolver() {
}


void FiniteElementMethodDynamicThermal3DSolver::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "temperature")
            this->readBoundaryConditions(manager, source, temperature_boundary);

        else if (param == "loop") {
            methodparam = source.getAttribute<double>("methodparam", methodparam);
            lumping = source.getAttribute<bool>("lumping", lumping);
            inittemp = source.getAttribute<double>("inittemp", inittemp);
            timestep = source.getAttribute<double>("timestep", timestep);
            rebuildfreq = source.getAttribute<size_t>("rebuildfreq", rebuildfreq);
            logfreq = source.getAttribute<size_t>("logfreq", logfreq);
            source.requireTagEnd();
        }

        else if (param == "matrix") {
            algorithm = source.enumAttribute<Algorithm>("algorithm")
                .value("cholesky", ALGORITHM_CHOLESKY)
                .value("gauss", ALGORITHM_GAUSS)
                .get(algorithm);
            source.requireTagEnd();
        } else
            this->parseStandardConfiguration(source, manager);
    }
}


void FiniteElementMethodDynamicThermal3DSolver::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    loopno = 0;
    size = this->mesh->size();
    temperatures.reset(size, inittemp);
}


void FiniteElementMethodDynamicThermal3DSolver::onInvalidate() {
    temperatures.reset();
    mHeatFluxes.reset();
}


template<typename MatrixT>
void FiniteElementMethodDynamicThermal3DSolver::setMatrix(MatrixT& A, MatrixT& B, DataVector<double>& F,
        const BoundaryConditionsWithMesh<RectangularMesh<3>,double>& btemperature)
{
    auto iMesh = (this->mesh)->getMidpointsMesh();
    auto heats = inHeat(iMesh);

    // zero the matrices A, B and the load vector F
    std::fill_n(A.data, A.size*(A.ld+1), 0.);
    std::fill_n(B.data, B.size*(B.ld+1), 0.);
    F.fill(0.);

    // Set stiffness matrix and load vector
    for (auto elem: this->mesh->elements)
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
        kx = ky;

        kx /= dx; kx *= dy; kx *= dz;
        ky *= dx; ky /= dy; ky *= dz;
        kz *= dx; kz *= dy; kz /= dz;

        // element of heat capacity matrix
        double c = material->cp(temp) * material->dens(temp) * 0.125 * 1E-12 * dx * dy * dz / timestep / 1E-9;

        // load vector: heat densities
        double f = 0.125e-18 * dx * dy * dz * heats[elem.getIndex()];   // 1e-18 -> to transform µm³ into m³

        // set components of symmetric matrix K
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

        // updating A, B matrices with K elements
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j <= i; ++j) {
                A(idx[i],idx[j]) += methodparam*K[i][j];
                B(idx[i],idx[j]) += -(1-methodparam)*K[i][j];
            }
        }

        // updating A, B matrices with C elements
        // Wheter lumping the mass matrces A, B?
        if (lumping)
        {
            for (int i = 0; i < 8; ++i) {
                A(idx[i],idx[i]) += c;
                B(idx[i],idx[i]) += c;
            }
        }
        else
        {
            //TODO
        }

    }

    //boundary conditions of the first kind
    for (auto cond: btemperature) {
        for (auto r: cond.place) {
            A(r,r) += BIG;
            F[r] += BIG * cond.value;
        }
    }

    // macierz A -> L L^T
    prepareMatrix(A);

#ifndef NDEBUG
    double* aend = A.data + A.size * A.kd;
    for (double* pa = A.data; pa != aend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(this->getId(), "Error in stiffness matrix at position %1%", pa-A.data);
    }
#endif

}


double FiniteElementMethodDynamicThermal3DSolver::compute(double time) {
    switch (algorithm) {
        case ALGORITHM_CHOLESKY: return doCompute<DpbMatrix>(time);
        case ALGORITHM_GAUSS: return doCompute<DgbMatrix>(time);
    }
    return 0.;
}


template<typename MatrixT>
double FiniteElementMethodDynamicThermal3DSolver::doCompute(double time)
{
    this->initCalculation();

    mHeatFluxes.reset();

    // store boundary conditions for current mesh
    auto btemperature = temperature_boundary(this->mesh, this->geometry);

    MatrixT A(size, this->mesh->minorAxis()->size());
    MatrixT B(size, this->mesh->minorAxis()->size());
    this->writelog(LOG_INFO, "Running thermal calculations");
    this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, bands=%2%{%3%})", A.size, A.kd+1, A.ld+1);
    maxT = *std::max_element(temperatures.begin(), temperatures.end());
    this->writelog(LOG_RESULT, "Time %d: max(T) = %d K", 0., maxT);

#   ifndef NDEBUG
        if (!temperatures.unique()) this->writelog(LOG_DEBUG, "Temperature data held by something else...");
#   endif
    temperatures = temperatures.claim();
    DataVector<double> F(size), T(size);

    setMatrix(A, B, F, btemperature);

    size_t r = rebuildfreq - 1,
           l = logfreq - 1;

    for (double t = timestep; t < time + timestep; t += timestep) {

        if (rebuildfreq && r == 0)
        {
            setMatrix(A, B, F, btemperature);
            r = rebuildfreq;
        }

        B.mult(temperatures, T);
        for (std::size_t i = 0; i < T.size(); ++i) T[i] += F[i];

        solveMatrix(A, T);

        std::swap(temperatures, T);

        if (logfreq && l == 0)
        {
            maxT = *std::max_element(temperatures.begin(), temperatures.end());
            this->writelog(LOG_RESULT, "Time %d: max(T) = %d K", t, maxT);
            l = logfreq;
        }

        r--;
        l--;
    }

    outTemperature.fireChanged();
    outHeatFlux.fireChanged();

    return toterr;
}


void FiniteElementMethodDynamicThermal3DSolver::prepareMatrix(DpbMatrix& A)
{
    int info = 0;

    // Factorize matrix TODO bez tego
    dpbtrf(UPLO, A.size, A.kd, A.data, A.ld+1, info);
    if (info < 0)
        throw CriticalException("%1%: Argument %2% of dpbtrf has illegal value", this->getId(), -info);
    else if (info > 0)
        throw ComputationError(this->getId(), "Leading minor of order %1% of the stiffness matrix is not positive-definite", info);

    // now A contains factorized matrix
}

void FiniteElementMethodDynamicThermal3DSolver::solveMatrix(DpbMatrix& A, DataVector<double>& B)
{
    int info = 0;

    // Find solutions
    dpbtrs(UPLO, A.size, A.kd, 1, A.data, A.ld+1, B.data(), B.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", this->getId(), -info);

    // now B contains solutions
}

void FiniteElementMethodDynamicThermal3DSolver::prepareMatrix(DgbMatrix& A)
{
    int info = 0;
    A.ipiv.reset(aligned_malloc<int>(A.size));

    A.mirror();

    // Factorize matrix
    dgbtrf(A.size, A.size, A.kd, A.kd, A.data, A.ld+1, A.ipiv.get(), info);
    if (info < 0) {
        throw CriticalException("%1%: Argument %2% of dgbtrf has illegal value", this->getId(), -info);
    } else if (info > 0) {
        throw ComputationError(this->getId(), "Matrix is singlar (at %1%)", info);
    }

    // now A contains factorized matrix
}

void FiniteElementMethodDynamicThermal3DSolver::solveMatrix(DgbMatrix& A, DataVector<double>& B)
{
    int info = 0;

    // Find solutions
    dgbtrs('N', A.size, A.kd, A.kd, 1, A.data, A.ld+1, A.ipiv.get(), B.data(), B.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dgbtrs has illegal value", this->getId(), -info);

    // now A contains factorized matrix and B the solutions
}

void FiniteElementMethodDynamicThermal3DSolver::saveHeatFluxes()
{
    this->writelog(LOG_DETAIL, "Computing heat fluxes");

    mHeatFluxes.reset(this->mesh->elements.size());

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

        mHeatFluxes[el.getIndex()] = vec(
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


const LazyData<double> FiniteElementMethodDynamicThermal3DSolver::getTemperatures(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) const {
    this->writelog(LOG_DETAIL, "Getting temperatures");
    if (!temperatures) return DataVector<const double>(dst_mesh->size(), inittemp); // in case the receiver is connected and no temperature calculated yet
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh, temperatures, make_shared<const WrappedMesh<3>>(dst_mesh, this->geometry), method);
}


const LazyData<Vec<3>> FiniteElementMethodDynamicThermal3DSolver::getHeatFluxes(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) {
    this->writelog(LOG_DETAIL, "Getting heat fluxes");
    if (!temperatures) return DataVector<const Vec<3>>(dst_mesh->size(), Vec<3>(0.,0.,0.)); // in case the receiver is connected and no fluxes calculated yet
    if (!mHeatFluxes) saveHeatFluxes(); // we will compute fluxes only if they are needed
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh->getMidpointsMesh(), mHeatFluxes, make_shared<const WrappedMesh<3>>(dst_mesh, this->geometry), method);
}


FiniteElementMethodDynamicThermal3DSolver::
ThermalConductivityData::ThermalConductivityData(const FiniteElementMethodDynamicThermal3DSolver* solver, const shared_ptr<const MeshD<3>>& dst_mesh):
solver(solver), element_mesh(solver->mesh->getMidpointsMesh()), target_mesh(dst_mesh, solver->geometry)
{
    if (solver->temperatures) temps = interpolate(solver->mesh, solver->temperatures, element_mesh, INTERPOLATION_LINEAR);
    else temps = LazyData<double>(element_mesh->size(), solver->inittemp);
}
Tensor2<double> FiniteElementMethodDynamicThermal3DSolver::ThermalConductivityData::at(std::size_t i) const {
    auto point = target_mesh[i];
    size_t x = std::upper_bound(solver->mesh->axis0->begin(), solver->mesh->axis0->end(), point[0]) - solver->mesh->axis0->begin();
    size_t y = std::upper_bound(solver->mesh->axis1->begin(), solver->mesh->axis1->end(), point[1]) - solver->mesh->axis1->begin();
    size_t z = std::upper_bound(solver->mesh->axis2->begin(), solver->mesh->axis2->end(), point[2]) - solver->mesh->axis2->begin();
    if (x == 0 || y == 0 || z == 0 || x == solver->mesh->axis0->size() || y == solver->mesh->axis1->size() || z == solver->mesh->axis2->size())
        return Tensor2<double>(NAN);
    else {
        size_t idx = element_mesh->index(x-1, y-1, z-1);
        auto point = element_mesh->at(idx);
        auto material = solver->geometry->getMaterial(point);
        Tensor2<double> result;
        if (auto leaf = dynamic_pointer_cast<const GeometryObjectD<2>>(solver->geometry->getMatchingAt(point, &GeometryObject::PredicateIsLeaf)))
            result = material->thermk(temps[idx], leaf->getBoundingBox().height());
        else
            result = material->thermk(temps[idx]);
        return result;
    }
}
std::size_t FiniteElementMethodDynamicThermal3DSolver::ThermalConductivityData::size() const { return target_mesh.size(); }

const LazyData<Tensor2<double>> FiniteElementMethodDynamicThermal3DSolver::getThermalConductivity(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) const {
    this->writelog(LOG_DETAIL, "Getting thermal conductivities");
    return LazyData<Tensor2<double>>(new FiniteElementMethodDynamicThermal3DSolver::ThermalConductivityData(this, dst_mesh));
}


}}} // namespace plask::solvers::thermal
