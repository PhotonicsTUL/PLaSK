#include <type_traits>

#include "femT3d.h"

namespace plask { namespace thermal { namespace dynamic {

const double BIG = 1e16;

FiniteElementMethodDynamicThermal3DSolver::FiniteElementMethodDynamicThermal3DSolver(const std::string& name) :
    SolverWithMesh<Geometry3D, RectangularMesh<3>>(name),
    outTemperature(this, &FiniteElementMethodDynamicThermal3DSolver::getTemperatures),
    outHeatFlux(this, &FiniteElementMethodDynamicThermal3DSolver::getHeatFluxes),
    outThermalConductivity(this, &FiniteElementMethodDynamicThermal3DSolver::getThermalConductivity),
    algorithm(ALGORITHM_CHOLESKY),
    inittemp(300.),
    methodparam(0.5),
    timestep(0.1),
    elapstime(0.),
    lumping(true),
    rebuildfreq(0),
    logfreq(500),
    use_full_mesh(false)
{
    temperatures.reset();
    fluxes.reset();
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
        }

        else {
            if (param == "mesh") {
                use_full_mesh = source.getAttribute<bool>("include-empty", use_full_mesh);
            }
            this->parseStandardConfiguration(source, manager);
        }
    }
}


void FiniteElementMethodDynamicThermal3DSolver::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    elapstime = 0.;
    band = 0;

    if (use_full_mesh)
        maskedMesh->selectAll(*this->mesh);
    else
        maskedMesh->reset(*this->mesh, *this->geometry, ~plask::Material::EMPTY);

    temperatures.reset(this->maskedMesh->size(), inittemp);

    thickness.reset(this->maskedMesh->getElementsCount(), NAN);
    // Set stiffness matrix and load vector
    for (auto elem: this->maskedMesh->elements())
    {
        if (!isnan(thickness[elem.getIndex()])) continue;
        auto material = this->geometry->getMaterial(elem.getMidpoint());
        double top = elem.getUpper2(), bottom = elem.getLower2();
        size_t row = elem.getIndex2();
        size_t itop = row+1, ibottom = row;
        for (size_t r = row; r > 0; r--) {
            auto e = this->mesh->element(elem.getIndex0(), elem.getIndex1(), r-1);
            auto m = this->geometry->getMaterial(e.getMidpoint());
            if (m == material) {                            //TODO ignore doping
                bottom = e.getLower2();
                ibottom = r-1;
            }
            else break;
        }
        for (size_t r = elem.getIndex2()+1; r < this->mesh->axis[2]->size()-1; r++) {
            auto e = this->mesh->element(elem.getIndex0(), elem.getIndex1(), r);
            auto m = this->geometry->getMaterial(e.getMidpoint());
            if (m == material) {                            //TODO ignore doping
                top = e.getUpper2();
                itop = r+1;
            }
            else break;
        }
        double h = top - bottom;
        for (size_t r = ibottom; r != itop; ++r) {
            size_t idx = this->maskedMesh->element(elem.getIndex0(), elem.getIndex1(), r).getIndex();
            if (idx != RectangularMaskedMesh3D::Element::UNKNOWN_ELEMENT_INDEX)
                thickness[idx] = h;
        }
    }
}


void FiniteElementMethodDynamicThermal3DSolver::onInvalidate() {
    temperatures.reset();
    fluxes.reset();
    thickness.reset();
}


template<typename MatrixT>
void FiniteElementMethodDynamicThermal3DSolver::setMatrix(MatrixT& A, MatrixT& B, DataVector<double>& F,
        const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,double>& btemperature)
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size={0}, bands={1}({2}))", A.size, A.kd+1, A.ld+1);

    auto heats = inHeat(maskedMesh->getElementMesh()/*, INTERPOLATION_NEAREST*/);

    // zero the matrices A, B and the load vector F
    std::fill_n(A.data, A.size*(A.ld+1), 0.);
    std::fill_n(B.data, B.size*(B.ld+1), 0.);
    F.fill(0.);

    // Set stiffness matrix and load vector
    for (auto elem: this->maskedMesh->elements())
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
        std::tie(ky,kz) = std::tuple<double,double>(material->thermk(temp, thickness[elem.getIndex()]));

        ky *= 1e-6; kz *= 1e-6;                                         // W/m -> W/µm
        kx = ky;

        kx /= dx; kx *= dy; kx *= dz;
        ky *= dx; ky /= dy; ky *= dz;
        kz *= dx; kz *= dy; kz /= dz;

        // element of heat capacity matrix
        double c = material->cp(temp) * material->dens(temp) * 0.125e-9 * dx * dy * dz / timestep;  //0.125e-9 = 0.5*0.5*0.5*1e-18/1E-9

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

        // updating A, B matrices with K elements and F load vector
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j <= i; ++j) {
                A(idx[i],idx[j]) += methodparam*K[i][j];
                B(idx[i],idx[j]) += -(1-methodparam)*K[i][j];
            }
            F[idx[i]] += f;
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
            // set components of symmetric matrix K
            double C[8][8];
            C[0][0] = C[1][1] = C[2][2] = C[3][3] = C[4][4] = C[5][5] = C[6][6] = C[7][7] = c * 8 / 27.;
            C[1][0] = C[3][0] = C[4][0] = C[2][1] = C[5][1] = C[3][2] = C[6][2] = C[7][3] = C[5][4] = C[7][4] = C[6][5] = C[7][6] = c * 4 / 27.;
            C[2][0] = C[5][0] = C[7][0] = C[3][1] = C[4][1] = C[6][1] = C[5][2] = C[7][2] = C[4][3] = C[6][3] = C[6][4] = C[7][5] = c * 2 / 27.;
            C[6][0] = C[7][1] = C[4][2] = C[5][3] = c / 27.;

            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j <= i; ++j) {
                    A(idx[i],idx[j]) += C[i][j];
                    B(idx[i],idx[j]) += C[i][j];
                }
            }
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
            throw ComputationError(this->getId(), "Error in stiffness matrix at position {0}", pa-A.data);
    }
#endif

}


template <typename MatrixT>
MatrixT FiniteElementMethodDynamicThermal3DSolver::makeMatrix() {
    if (band == 0) {
        if (use_full_mesh) {
            band = this->mesh->minorAxis()->size() + 1;
        } else {
            for (auto element: this->maskedMesh->elements()) {
                size_t span = element.getUpUpUpIndex() - element.getLoLoLoIndex();
                if (span > band) band = span;
            }
        }
    }
    return MatrixT(this->maskedMesh->size(), band);
}

// template <>
// SparseBandMatrix3D FiniteElementMethodDynamicThermal3DSolver::makeMatrix<SparseBandMatrix3D>() {
//     if (!use_full_mesh)
//         throw NotImplemented(this->getId(), "Iterative algorithm with empty materials not included");
//     return SparseBandMatrix3D(this->maskedMesh->size(), mesh->mediumAxis()->size()*mesh->minorAxis()->size(), mesh->minorAxis()->size());
// }


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

    fluxes.reset();

    // store boundary conditions for current mesh
    auto btemperature = temperature_boundary(this->maskedMesh, this->geometry);

    size_t size = this->maskedMesh->size();
    MatrixT A = makeMatrix<MatrixT>();
    MatrixT B = makeMatrix<MatrixT>();
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

    time += timestep/2.;
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
            this->writelog(LOG_RESULT, "Time {:.2f} ns: max(T) = {:.3f} K", elapstime, maxT);
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


void FiniteElementMethodDynamicThermal3DSolver::prepareMatrix(DpbMatrix& A)
{
    int info = 0;

    // Factorize matrix TODO bez tego
    dpbtrf(UPLO, int(A.size), int(A.kd), A.data, int(A.ld+1), info);
    if (info < 0)
        throw CriticalException("{0}: Argument {1} of dpbtrf has illegal value", this->getId(), -info);
    else if (info > 0)
        throw ComputationError(this->getId(), "Leading minor of order {0} of the stiffness matrix is not positive-definite", info);

    // now A contains factorized matrix
}

void FiniteElementMethodDynamicThermal3DSolver::solveMatrix(DpbMatrix& A, DataVector<double>& B)
{
    int info = 0;

    // Find solutions
    dpbtrs(UPLO, int(A.size), int(A.kd), 1, A.data, int(A.ld+1), B.data(), int(B.size()), info);
    if (info < 0) throw CriticalException("{0}: Argument {1} of dpbtrs has illegal value", this->getId(), -info);

    // now B contains solutions
}

void FiniteElementMethodDynamicThermal3DSolver::prepareMatrix(DgbMatrix& A)
{
    int info = 0;
    A.ipiv.reset(aligned_malloc<int>(A.size));

    A.mirror();

    // Factorize matrix
    dgbtrf(int(A.size), int(A.size), int(A.kd), int(A.kd), A.data, int(A.ld+1), A.ipiv.get(), info);
    if (info < 0) {
        throw CriticalException("{0}: Argument {1} of dgbtrf has illegal value", this->getId(), -info);
    } else if (info > 0) {
        throw ComputationError(this->getId(), "Matrix is singlar (at {0})", info);
    }

    // now A contains factorized matrix
}

void FiniteElementMethodDynamicThermal3DSolver::solveMatrix(DgbMatrix& A, DataVector<double>& B)
{
    int info = 0;

    // Find solutions
    dgbtrs('N', int(A.size), int(A.kd), int(A.kd), 1, A.data, int(A.ld+1), A.ipiv.get(), B.data(), int(B.size()), info);
    if (info < 0) throw CriticalException("{0}: Argument {1} of dgbtrs has illegal value", this->getId(), -info);

    // now A contains factorized matrix and B the solutions
}

void FiniteElementMethodDynamicThermal3DSolver::saveHeatFluxes()
{
    this->writelog(LOG_DETAIL, "Computing heat fluxes");

    fluxes.reset(this->maskedMesh->getElementsCount());

    for (auto el: this->maskedMesh->elements())
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


const LazyData<double> FiniteElementMethodDynamicThermal3DSolver::getTemperatures(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) const {
    this->writelog(LOG_DEBUG, "Getting temperatures");
    if (!temperatures) return LazyData<double>(dst_mesh->size(), inittemp); // in case the receiver is connected and no temperature calculated yet
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    if (use_full_mesh)
        return SafeData<double>(interpolate(this->mesh, temperatures, dst_mesh, method, this->geometry), 300.);
    else
        return SafeData<double>(interpolate(this->maskedMesh, temperatures, dst_mesh, method, this->geometry), 300.);
}


const LazyData<Vec<3>> FiniteElementMethodDynamicThermal3DSolver::getHeatFluxes(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) {
    this->writelog(LOG_DEBUG, "Getting heat fluxes");
    if (!temperatures) return LazyData<Vec<3>>(dst_mesh->size(), Vec<3>(0.,0.,0.)); // in case the receiver is connected and no fluxes calculated yet
    if (!fluxes) saveHeatFluxes(); // we will compute fluxes only if they are needed
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    if (use_full_mesh)
        return SafeData<Vec<3>>(interpolate(this->mesh->getElementMesh(), fluxes, dst_mesh, method,
                                            InterpolationFlags(this->geometry, InterpolationFlags::Symmetry::NPP, InterpolationFlags::Symmetry::PNP, InterpolationFlags::Symmetry::PPN)),
                                Zero<Vec<3>>());
    else
        return SafeData<Vec<3>>(interpolate(this->maskedMesh->getElementMesh(), fluxes, dst_mesh, method,
                                            InterpolationFlags(this->geometry,InterpolationFlags::Symmetry::NPP, InterpolationFlags::Symmetry::PNP, InterpolationFlags::Symmetry::PPN)),
                                Zero<Vec<3>>());
}


FiniteElementMethodDynamicThermal3DSolver::
ThermalConductivityData::ThermalConductivityData(const FiniteElementMethodDynamicThermal3DSolver* solver, const shared_ptr<const MeshD<3>>& dst_mesh):
    solver(solver), dest_mesh(dst_mesh), flags(solver->geometry)
{
    if (solver->temperatures) temps = interpolate(solver->maskedMesh, solver->temperatures, solver->maskedMesh->getElementMesh(), INTERPOLATION_LINEAR);
    else temps = LazyData<double>(solver->mesh->getElementsCount(), solver->inittemp);
}
Tensor2<double> FiniteElementMethodDynamicThermal3DSolver::ThermalConductivityData::at(std::size_t i) const {
    auto point = flags.wrap(dest_mesh->at(i));
    std::size_t x = solver->mesh->axis[0]->findUpIndex(point[0]),
                y = solver->mesh->axis[1]->findUpIndex(point[1]),
                z = solver->mesh->axis[2]->findUpIndex(point[2]);
    if (x == 0 || y == 0 || z == 0 || x == solver->mesh->axis[0]->size() || y == solver->mesh->axis[1]->size() || z == solver->mesh->axis[2]->size())
        return Tensor2<double>(NAN);
    else {
        auto elem = solver->maskedMesh->element(x-1, y-1, z-1);
        auto material = solver->geometry->getMaterial(elem.getMidpoint());
        size_t idx = elem.getIndex();
        if (idx == RectangularMaskedMesh3D::Element::UNKNOWN_ELEMENT_INDEX) return Tensor2<double>(NAN);
        return material->thermk(temps[idx], solver->thickness[idx]);
    }
}
std::size_t FiniteElementMethodDynamicThermal3DSolver::ThermalConductivityData::size() const { return dest_mesh->size(); }

const LazyData<Tensor2<double>> FiniteElementMethodDynamicThermal3DSolver::getThermalConductivity(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod /*method*/) {
    this->initCalculation();
    this->writelog(LOG_DEBUG, "Getting thermal conductivities");
    return LazyData<Tensor2<double>>(new FiniteElementMethodDynamicThermal3DSolver::ThermalConductivityData(this, dst_mesh));
}


}}} // namespace plask::thermal::thermal
