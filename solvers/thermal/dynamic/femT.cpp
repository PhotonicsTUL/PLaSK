#include "femT.h"

namespace plask { namespace solvers { namespace thermal {

const double BIG = 1e16;

template<typename Geometry2DType>
FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::FiniteElementMethodDynamicThermal2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DType, RectangularMesh<2>>(name),
    outTemperature(this, &FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::getTemperatures),
    outHeatFlux(this, &FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::getHeatFluxes),
    outThermalConductivity(this, &FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::getThermalConductivity),
    algorithm(ALGORITHM_CHOLESKY),
    inittemp(300.),    
    methodparam(0.5),
    timestep(0.1),
    elapstime(0.),
    lumping(true),
    rebuildfreq(0),
    logfreq(500)
{
    temperatures.reset();
    mHeatFluxes.reset();
    inHeat = 0.;
}


template<typename Geometry2DType>
FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::~FiniteElementMethodDynamicThermal2DSolver() {
}


template<typename Geometry2DType>
void FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "temperature")
            this->readBoundaryConditions(manager, source, temperature_boundary);

        else if (param == "loop") {
            inittemp = source.getAttribute<double>("inittemp", inittemp);
            timestep = source.getAttribute<double>("timestep", timestep);
            rebuildfreq = source.getAttribute<size_t>("rebuildfreq", rebuildfreq);
            logfreq = source.getAttribute<size_t>("logfreq", logfreq);
            source.requireTagEnd();
        }

        else if (param == "matrix") {
            methodparam = source.getAttribute<double>("methodparam", methodparam);
            lumping = source.getAttribute<bool>("lumping", lumping);
            algorithm = source.enumAttribute<Algorithm>("algorithm")
                .value("cholesky", ALGORITHM_CHOLESKY)
                .value("gauss", ALGORITHM_GAUSS)
                .get(algorithm);
            source.requireTagEnd();
        } else
            this->parseStandardConfiguration(source, manager);
    }
}


template<typename Geometry2DType>
void FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    elapstime = 0.;
    size = this->mesh->size();
    temperatures.reset(size, inittemp);
}


template<typename Geometry2DType> void FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::onInvalidate() {
    temperatures.reset();
    mHeatFluxes.reset();
}


template<> template<typename MatrixT>
void FiniteElementMethodDynamicThermal2DSolver<Geometry2DCartesian>::setMatrix(
        MatrixT& A, MatrixT& B, DataVector<double>& F,
        const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& btemperature)
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, bands=%2%{%3%})", A.size, A.kd+1, A.ld+1);

    auto iMesh = (this->mesh)->getMidpointsMesh();
    auto heatdensities = inHeat(iMesh);

    // zero the matrices A, B and the load vector F
    std::fill_n(A.data, A.size*(A.ld+1), 0.);
    std::fill_n(B.data, B.size*(B.ld+1), 0.);
    F.fill(0.);

    // Set stiffness matrix and load vector
    for (auto e: this->mesh->elements)
    {
        // nodes numbers for the current element
        size_t loleftno = e.getLoLoIndex();
        size_t lorghtno = e.getUpLoIndex();
        size_t upleftno = e.getLoUpIndex();
        size_t uprghtno = e.getUpUpIndex();

        // element size
        double elemwidth = e.getUpper0() - e.getLower0();
        double elemheight = e.getUpper1() - e.getLower1();

        // point and material in the middle of the element
        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        // average temperature on the element
        double temp = 0.25 * (temperatures[loleftno] + temperatures[lorghtno] + temperatures[upleftno] + temperatures[uprghtno]);

        // thermal conductivity
        double kx, ky;
        auto leaf = dynamic_pointer_cast<const GeometryObjectD<2>>(this->geometry->getMatchingAt(midpoint, &GeometryObject::PredicateIsLeaf));
        if (leaf)
            std::tie(kx,ky) = std::tuple<double,double>(material->thermk(temp, leaf->getBoundingBox().height()));
        else
            std::tie(kx,ky) = std::tuple<double,double>(material->thermk(temp));

        // element of heat capacity matrix
        double c = material->cp(temp) * material->dens(temp) * 0.25 * 1E-12 * elemheight * elemwidth / timestep / 1E-9;

        kx *= elemheight; kx /= elemwidth;
        ky *= elemwidth; ky /= elemheight;

        // load vector: heat densities
        double f = 0.25e-12 * elemwidth * elemheight * heatdensities[e.getIndex()]; // 1e-12 -> to transform µm² into m²

        // set symmetric matrix components in thermal conductivity matrix
        double k44, k33, k22, k11, k43, k21, k42, k31, k32, k41;

        k44 = k33 = k22 = k11 = (kx + ky) / 3.;
        k43 = k21 = (-2. * kx + ky) / 6.;
        k42 = k31 = - (kx + ky) / 6.;
        k32 = k41 = (kx - 2. * ky) / 6.;

        double f1 = f, f2 = f, f3 = f, f4 = f;

        //Wheter lumping the mass matrces A, B?
        if (lumping)
        {
            A(loleftno, loleftno) += methodparam*k11 + c;
            A(lorghtno, lorghtno) += methodparam*k22 + c;
            A(uprghtno, uprghtno) += methodparam*k33 + c;
            A(upleftno, upleftno) += methodparam*k44 + c;

            A(lorghtno, loleftno) += methodparam*k21;
            A(uprghtno, loleftno) += methodparam*k31;
            A(upleftno, loleftno) += methodparam*k41;
            A(uprghtno, lorghtno) += methodparam*k32;
            A(upleftno, lorghtno) += methodparam*k42;
            A(upleftno, uprghtno) += methodparam*k43;

            B(loleftno, loleftno) += -(1-methodparam)*k11 + c;
            B(lorghtno, lorghtno) += -(1-methodparam)*k22 + c;
            B(uprghtno, uprghtno) += -(1-methodparam)*k33 + c;
            B(upleftno, upleftno) += -(1-methodparam)*k44 + c;

            B(lorghtno, loleftno) += -(1-methodparam)*k21;
            B(uprghtno, loleftno) += -(1-methodparam)*k31;
            B(upleftno, loleftno) += -(1-methodparam)*k41;
            B(uprghtno, lorghtno) += -(1-methodparam)*k32;
            B(upleftno, lorghtno) += -(1-methodparam)*k42;
            B(upleftno, uprghtno) += -(1-methodparam)*k43;
        }
        else
        {
            A(loleftno, loleftno) += methodparam*k11 + 4./9.*c;
            A(lorghtno, lorghtno) += methodparam*k22 + 4./9.*c;
            A(uprghtno, uprghtno) += methodparam*k33 + 4./9.*c;
            A(upleftno, upleftno) += methodparam*k44 + 4./9.*c;

            A(lorghtno, loleftno) += methodparam*k21 + 2./9.*c;
            A(uprghtno, loleftno) += methodparam*k31 + 1./9.*c;
            A(upleftno, loleftno) += methodparam*k41 + 2./9.*c;
            A(uprghtno, lorghtno) += methodparam*k32 + 2./9.*c;
            A(upleftno, lorghtno) += methodparam*k42 + 1./9.*c;
            A(upleftno, uprghtno) += methodparam*k43 + 2./9.*c;

            B(loleftno, loleftno) += -(1-methodparam)*k11 + 4./9.*c;
            B(lorghtno, lorghtno) += -(1-methodparam)*k22 + 4./9.*c;
            B(uprghtno, uprghtno) += -(1-methodparam)*k33 + 4./9.*c;
            B(upleftno, upleftno) += -(1-methodparam)*k44 + 4./9.*c;

            B(lorghtno, loleftno) += -(1-methodparam)*k21 + 2./9.*c;
            B(uprghtno, loleftno) += -(1-methodparam)*k31 + 1./9.*c;
            B(upleftno, loleftno) += -(1-methodparam)*k41 + 2./9.*c;
            B(uprghtno, lorghtno) += -(1-methodparam)*k32 + 2./9.*c;
            B(upleftno, lorghtno) += -(1-methodparam)*k42 + 1./9.*c;
            B(upleftno, uprghtno) += -(1-methodparam)*k43 + 2./9.*c;
        }
        // set load vector
        F[loleftno] += f1;
        F[lorghtno] += f2;
        F[uprghtno] += f3;
        F[upleftno] += f4;
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


template<typename Geometry2DType>
double FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::compute(double time) {
    switch (algorithm) {
        case ALGORITHM_CHOLESKY: return doCompute<DpbMatrix>(time);
        case ALGORITHM_GAUSS: return doCompute<DgbMatrix>(time);
    }
    return 0.;
}


template<typename Geometry2DType> template<typename MatrixT>
double FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::doCompute(double time)
{
    this->initCalculation();

    mHeatFluxes.reset();

    // store boundary conditions for current mesh
    auto btemperature = temperature_boundary(this->mesh, this->geometry);

    MatrixT A(size, this->mesh->minorAxis()->size());
    MatrixT B(size, this->mesh->minorAxis()->size());
    this->writelog(LOG_INFO, "Running thermal calculations");
    maxT = *std::max_element(temperatures.begin(), temperatures.end());

#   ifndef NDEBUG
        if (!temperatures.unique()) this->writelog(LOG_DEBUG, "Temperature data held by something else...");
#   endif
    temperatures = temperatures.claim();
    DataVector<double> F(size), T(size);

    setMatrix(A, B, F, btemperature);

    size_t r = rebuildfreq,
           l = logfreq;

    for (double t = 0.; t < time; t += timestep) {

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
            this->writelog(LOG_RESULT, "Time %.4f us: max(T) = %.3f K", elapstime/1e3, maxT);
            l = logfreq;
        }

        r--;
        l--;
        elapstime += timestep;
    }

    elapstime -= timestep;
    outTemperature.fireChanged();
    outHeatFlux.fireChanged();

    return 0.;
}


template<typename Geometry2DType>
void FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::prepareMatrix(DpbMatrix& A)
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

template<typename Geometry2DType>
void FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::solveMatrix(DpbMatrix& A, DataVector<double>& B)
{
    int info = 0;

    // Find solutions
    dpbtrs(UPLO, A.size, A.kd, 1, A.data, A.ld+1, B.data(), B.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", this->getId(), -info);

    // now B contains solutions
}

template<typename Geometry2DType>
void FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::prepareMatrix(DgbMatrix& A)
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

template<typename Geometry2DType>
void FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::solveMatrix(DgbMatrix& A, DataVector<double>& B)
{
    int info = 0;

    // Find solutions
    dgbtrs('N', A.size, A.kd, A.kd, 1, A.data, A.ld+1, A.ipiv.get(), B.data(), B.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dgbtrs has illegal value", this->getId(), -info);

    // now A contains factorized matrix and B the solutions
}

template<typename Geometry2DType>
void FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::saveHeatFluxes()
{
    this->writelog(LOG_DETAIL, "Computing heat fluxes");

    mHeatFluxes.reset(this->mesh->elements.size());

    for (auto e: this->mesh->elements)
    {
        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        size_t loleftno = e.getLoLoIndex();
        size_t lorghtno = e.getUpLoIndex();
        size_t upleftno = e.getLoUpIndex();
        size_t uprghtno = e.getUpUpIndex();

        double temp = 0.25 * (temperatures[loleftno] + temperatures[lorghtno] +
                               temperatures[upleftno] + temperatures[uprghtno]);

        double kx, ky;
        auto leaf = dynamic_pointer_cast<const GeometryObjectD<2>>(
                        this->geometry->getMatchingAt(midpoint, &GeometryObject::PredicateIsLeaf)
                     );
        if (leaf)
            std::tie(kx,ky) = std::tuple<double,double>(material->thermk(temp, leaf->getBoundingBox().height()));
        else
            std::tie(kx,ky) = std::tuple<double,double>(material->thermk(temp));


        mHeatFluxes[e.getIndex()] = vec(
            - 0.5e6 * kx * (- temperatures[loleftno] + temperatures[lorghtno]
                             - temperatures[upleftno] + temperatures[uprghtno]) / (e.getUpper0() - e.getLower0()), // 1e6 - from um to m
            - 0.5e6 * ky * (- temperatures[loleftno] - temperatures[lorghtno]
                             + temperatures[upleftno] + temperatures[uprghtno]) / (e.getUpper1() - e.getLower1())); // 1e6 - from um to m
    }
}


template<typename Geometry2DType>
const LazyData<double> FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::getTemperatures(const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method) const {
    this->writelog(LOG_DETAIL, "Getting temperatures");
    if (!temperatures) return DataVector<const double>(dst_mesh->size(), inittemp); // in case the receiver is connected and no temperature calculated yet
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh, temperatures, make_shared<const WrappedMesh<2>>(dst_mesh, this->geometry), method);
}


template<typename Geometry2DType>
const LazyData<Vec<2>> FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::getHeatFluxes(const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method) {
    this->writelog(LOG_DETAIL, "Getting heat fluxes");
    if (!temperatures) return DataVector<const Vec<2>>(dst_mesh->size(), Vec<2>(0.,0.)); // in case the receiver is connected and no fluxes calculated yet
    if (!mHeatFluxes) saveHeatFluxes(); // we will compute fluxes only if they are needed
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh->getMidpointsMesh(), mHeatFluxes,make_shared<const WrappedMesh<2>>(dst_mesh, this->geometry), method);
}


template<typename Geometry2DType> FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::
ThermalConductivityData::ThermalConductivityData(const FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>* solver, const shared_ptr<const MeshD<2>>& dst_mesh):
    solver(solver), element_mesh(solver->mesh->getMidpointsMesh()), target_mesh(dst_mesh, solver->geometry)
{
    if (solver->temperatures) temps = interpolate(solver->mesh, solver->temperatures, element_mesh, INTERPOLATION_LINEAR);
    else temps = LazyData<double>(element_mesh->size(), solver->inittemp);
}
template<typename Geometry2DType> Tensor2<double> FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::
ThermalConductivityData::at(std::size_t i) const {
    auto point = target_mesh[i];
    size_t x = std::upper_bound(solver->mesh->axis0->begin(), solver->mesh->axis0->end(), point[0]) - solver->mesh->axis0->begin();
    size_t y = std::upper_bound(solver->mesh->axis1->begin(), solver->mesh->axis1->end(), point[1]) - solver->mesh->axis1->begin();
    if (x == 0 || y == 0 || x == solver->mesh->axis0->size() || y == solver->mesh->axis1->size())
        return Tensor2<double>(NAN);
    else {
        size_t idx = element_mesh->index(x-1, y-1);
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
template<typename Geometry2DType> std::size_t FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::
ThermalConductivityData::size() const { return target_mesh.size(); }

template<typename Geometry2DType>
const LazyData<Tensor2<double>> FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::getThermalConductivity(const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod) const {
    this->writelog(LOG_DETAIL, "Getting thermal conductivities");
    return LazyData<Tensor2<double>>(new
        FiniteElementMethodDynamicThermal2DSolver<Geometry2DType>::ThermalConductivityData(this, dst_mesh)
    );
}


template<> std::string FiniteElementMethodDynamicThermal2DSolver<Geometry2DCartesian>::getClassName() const { return "thermal.Dynamic2D"; }

template struct PLASK_SOLVER_API FiniteElementMethodDynamicThermal2DSolver<Geometry2DCartesian>;

}}} // namespace plask::solvers::thermal
