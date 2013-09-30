#include <type_traits>

#include "electr3d.h"

namespace plask { namespace solvers { namespace electrical3d {

FiniteElementMethodElectrical3DSolver::FiniteElementMethodElectrical3DSolver(const std::string& name) :
    SolverWithMesh<Geometry3D, RectilinearMesh3D>(name),
    algorithm(ALGORITHM_ITERATIVE),
    loopno(0),
    default_junction_conductivity(5.),
    maxerr(0.05),
    itererr(1e-8),
    iterlim(10000),
    logfreq(500),
    outPotential(this, &FiniteElementMethodElectrical3DSolver::getPotential),
    outCurrentDensity(this, &FiniteElementMethodElectrical3DSolver::getCurrentDensity),
    outHeat(this, &FiniteElementMethodElectrical3DSolver::getHeatDensity)
{
    potential.reset();
    current.reset();
    inTemperature = 300.;
    junction_conductivity.reset(1, default_junction_conductivity);
}


FiniteElementMethodElectrical3DSolver::~FiniteElementMethodElectrical3DSolver() {
}


void FiniteElementMethodElectrical3DSolver::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "voltage")
            readBoundaryConditions(manager, source, voltage_boundary);

        else if (param == "loop") {
            maxerr = source.getAttribute<double>("maxerr", maxerr);
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
        }

        else if (param == "junction") {
            js = source.getAttribute<double>("js", js);
            beta = source.getAttribute<double>("beta", beta);
            auto condjunc = source.getAttribute<double>("pnjcond");
            if (condjunc) setCondJunc(*condjunc);
            auto wavelength = source.getAttribute<double>("wavelength");
            if (wavelength) inWavelength = *wavelength;
            heatmet = source.enumAttribute<HeatMethod>("heat")
                .value("joules", HEAT_JOULES)
                .value("wavelength", HEAT_BANDGAP)
                .get(heatmet);
            source.requireTagEnd();
        }

        else if (param == "contacts") {
            pcond = source.getAttribute<double>("pcond", pcond);
            ncond = source.getAttribute<double>("ncond", ncond);
            source.requireTagEnd();
        }

        else
            parseStandardConfiguration(source, manager);
    }
}


void FiniteElementMethodElectrical3DSolver::setActiveRegions()
{
    if (!geometry || !mesh) {
        if (junction_conductivity.size() != 1) {
            double condy = 0.;
            for (auto cond: junction_conductivity) condy += cond;
            junction_conductivity.reset(1, condy / junction_conductivity.size());
        }
        return;
    }

    actlo.clear();
    acthi.clear();
    actd.clear();

    shared_ptr<RectilinearMesh3D> points = mesh->getMidpointsMesh();

    size_t ileft = 0, iright = points->axis0.size();
    size_t iback = 0, ifront = points->axis1.size();
    bool in_active = false;

    for (size_t ver = 0; ver < points->axis2.size(); ++ver) {
        bool had_active_tra = false;
        for (size_t tra = 0; tra < points->axis1.size(); ++tra) {
            bool had_active_lon = false;
            for (size_t lon = 0; lon < points->axis0.size(); ++lon) {
                auto point = points->at(lon, tra, ver);
                bool active = isActive(point);

                // Here we are inside potential active region
                if (tra < ileft) {
                    if (active) throw Exception("%1%: Left edge of the active region not aligned.", getId());
                } else if (tra >= iright) {
                    if (active) throw Exception("%1%: Right edge of the active region not aligned.", getId());
                } else if (lon < iback) {
                    if (active) throw Exception("%1%: Back edge of the active region not aligned.", getId());
                } else if (lon >= ifront) {
                    if (active) throw Exception("%1%: Front edge of the active region not aligned.", getId());
                } else if (active) {
                    if (!had_active_lon && !had_active_tra) {
                        if (!in_active) { // active region is starting set-up new region info
                            ileft = tra;
                            iback = lon;
                            actlo.push_back(ver);
                        }
                    }
                    had_active_lon = true;
                } else if (had_active_lon) {
                    if (!in_active && !had_active_tra) ifront = lon;
                    else throw Exception("%1%: Front edge of the active region not aligned.", getId());
                } else if (had_active_tra) {
                    if (!in_active) iright = tra;
                    else throw Exception("%1%: Right edge of the active region not aligned.", getId());
                }
            }
            had_active_tra |= had_active_lon;
        }
        in_active = had_active_tra;

        // Test if the active region has finished
        if (!in_active && actlo.size() != acthi.size()) {
            acthi.push_back(ver);
            actd.push_back(mesh->axis2[acthi.back()] - mesh->axis2[actlo.back()]);
            this->writelog(LOG_DETAIL, "Detected active layer %2% thickness = %1%nm", 1e3 * actd.back(), actd.size()-1);
        }
    }

    if (actlo.size() != acthi.size()) {
        acthi.push_back(points->axis2.size());
        actd.push_back(mesh->axis2[acthi.back()] - mesh->axis2[actlo.back()]);
        this->writelog(LOG_DETAIL, "Detected active layer %2% thickness = %1%nm", 1e3 * actd.back(), actd.size()-1);
    }

    assert(acthi.size() == actlo.size());

    size_t condsize = max(actlo.size() * (mesh->axis0.size()-1) * (mesh->axis1.size()-1), size_t(1));

    if (junction_conductivity.size() != condsize) {
        double condy = 0.;
        for (auto cond: junction_conductivity) condy += cond;
        junction_conductivity.reset(condsize, condy / junction_conductivity.size());
    }
}


void FiniteElementMethodElectrical3DSolver::onInitialize() {
    if (!geometry) throw NoGeometryException(getId());
    if (!mesh) throw NoMeshException(getId());
    loopno = 0;
    potential.reset(mesh->size(), 0.);
    current.reset(this->mesh->elements.size(), vec(0.,0.,0.));
    conds.reset(this->mesh->elements.size());
    if (junction_conductivity.size() == 1) {
        size_t condsize = max(actlo.size() * (this->mesh->axis1.size()-1) * (this->mesh->axis0.size()-1), size_t(1));
        junction_conductivity.reset(condsize, junction_conductivity[0]);
    }
}


void FiniteElementMethodElectrical3DSolver::onInvalidate() {
    conds.reset();
    potential.reset();
    current.reset();
    heat.reset();
    junction_conductivity.reset(1, default_junction_conductivity);
}


void FiniteElementMethodElectrical3DSolver::loadConductivity()
{
    auto midmesh = (mesh)->getMidpointsMesh();
    auto temperature = inTemperature(midmesh);

    for (auto e: mesh->elements)
    {
        size_t i = e.getIndex();
        Vec<3,double> midpoint = e.getMidpoint();

        auto roles = geometry->getRolesAt(midpoint);
        if (roles.find("active") != roles.end() || roles.find("junction") != roles.end()) {
            size_t n = std::upper_bound(acthi.begin(), acthi.end(), mesh->index2(i)) - acthi.begin();
            assert(n < acthi.size());
            conds[i] = Tensor2<double>(0., junction_conductivity[(n * (mesh->axis1.size()-1) + e.getIndex1()) * (mesh->axis0.size()-1) + e.getIndex0()]);
        } else if (roles.find("p-contact") != roles.end()) {
            conds[i] = Tensor2<double>(pcond, pcond);
        } else if (roles.find("n-contact") != roles.end()) {
            conds[i] = Tensor2<double>(ncond, ncond);
        } else
            conds[i] = geometry->getMaterial(midpoint)->cond(temperature[i]);
    }
}

void FiniteElementMethodElectrical3DSolver::saveConductivity()
{
    for (size_t n = 0; n < getActNo(); ++n) {
        size_t z = (actlo[n]+acthi[n])/2;
        for (size_t y = 0; y != mesh->axis1.size()-1; ++y)
            for (size_t x = 0; x != mesh->axis0.size()-1; ++x)
                junction_conductivity[(n * (mesh->axis1.size()-1) + y) * (mesh->axis0.size()-1) + x] = conds[mesh->elements(x, y, z).getIndex()].c11;
    }
}


template <typename MatrixT>
void FiniteElementMethodElectrical3DSolver::setMatrix(MatrixT& A, DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& bvoltage)
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, bands=%2%{%3%})", A.size, A.kd+1, A.ld+1);

    // Uupdate junction conductivities
    if (loopno != 0) {
        for (auto elem: mesh->elements) {
            if (isActive(elem)) {
                size_t index = elem.getIndex(), lll = elem.getLoLoLoIndex(), uuu = elem.getUpUpUpIndex();
                size_t back = mesh->index0(lll),
                       front = mesh->index0(uuu),
                       left = mesh->index1(lll),
                       right = mesh->index1(uuu);
                size_t nact = std::upper_bound(acthi.begin(), acthi.end(), this->mesh->index2(lll)) - acthi.begin();
                assert(nact < acthi.size());
                double jy = - 0.25e6 * conds[index].c11  *
                    (- potential[mesh->index(back,left,actlo[nact])] - potential[mesh->index(front,left,actlo[nact])]
                     - potential[mesh->index(back,right,actlo[nact])] - potential[mesh->index(front,right,actlo[nact])]
                     + potential[mesh->index(back,left,acthi[nact])] + potential[mesh->index(front,left,acthi[nact])]
                     + potential[mesh->index(back,right,acthi[nact])] + potential[mesh->index(front,right,acthi[nact])])
                    / actd[nact]; // [j] = A/m²
                conds[index] = Tensor2<double>(0., 1e-6 * beta * jy * actd[nact] / log(jy / js + 1.));
            }
        }
    }

    // Zero the matrix and the load vector
    A.clear();
    B.fill(0.);

    // Set stiffness matrix and load vector
    for (auto elem: mesh->elements) {

        size_t index = elem.getIndex();

        // nodes numbers for the current element
        size_t idx[8];
        idx[0] = elem.getLoLoLoIndex();          //   z              4-----6
        idx[1] = elem.getUpLoLoIndex();          //   |__y          /|    /|
        idx[2] = elem.getLoUpLoIndex();          //  x/            5-----7 |
        idx[3] = elem.getUpUpLoIndex();          //                | 0---|-2
        idx[4] = elem.getLoLoUpIndex();          //                |/    |/
        idx[5] = elem.getUpLoUpIndex();          //                1-----3
        idx[6] = elem.getLoUpUpIndex();          //
        idx[7] = elem.getUpUpUpIndex();          //

        // element size
        double dx = elem.getUpper0() - elem.getLower0();
        double dy = elem.getUpper1() - elem.getLower1();
        double dz = elem.getUpper2() - elem.getLower2();

        // point and material in the middle of the element
        Vec<3> middle = elem.getMidpoint();
        auto material = geometry->getMaterial(middle);

        // average voltage on the element
        double temp = 0.; for (int i = 0; i < 8; ++i) temp += potential[idx[i]]; temp *= 0.125;

        // electrical conductivity
        double kx, ky = conds[index].c00, kz = conds[index].c11;

        ky *= 1e-6; kz *= 1e-6;                                         // 1/m -> 1/µm

        kx = ky/dx; kx = ky*dy; kx = ky*dz;
        ky *= dx;   ky /= dy;   ky *= dz;
        kz *= dx;   kz *= dy;   kz /= dz;

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

        for (int i = 0; i < 8; ++i)
            for (int j = 0; j <= i; ++j)
                A(idx[i],idx[j]) += K[i][j];
    }

    applyBC(A, B, bvoltage);

#ifndef NDEBUG
    double* aend = A.data + A.size * A.kd;
    for (double* pa = A.data; pa != aend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(getId(), "Error in stiffness matrix at position %1% (%2%)", pa-A.data, isnan(*pa)?"nan":"inf");
    }
#endif

}

template <>
void FiniteElementMethodElectrical3DSolver::applyBC<DpbMatrix>(DpbMatrix& A, DataVector<double>& B,
                                                            const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& bvoltage) {
    // boundary conditions of the first kind
    for (auto cond: bvoltage) {
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
void FiniteElementMethodElectrical3DSolver::applyBC<SparseBandMatrix>(SparseBandMatrix& A, DataVector<double>& B,
                                                                   const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& bvoltage) {
    // boundary conditions of the first kind
    for (auto cond: bvoltage) {
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
double FiniteElementMethodElectrical3DSolver::doCompute(unsigned loops)
{
    initCalculation();

    // store boundary conditions for current mesh
    auto bvoltage = voltage_boundary(mesh, geometry);

    this->writelog(LOG_INFO, "Running electrical calculations");

    unsigned loop = 0;
    size_t size = mesh->size();

    MatrixT A(size, mesh->mediumAxis().size()*mesh->minorAxis().size(), mesh->minorAxis().size());

    double err = 0.;
    toterr = 0.;

#   ifndef NDEBUG
        if (!potential.unique()) this->writelog(LOG_DEBUG, "Potentials data held by something else...");
#   endif
    potential = potential.claim();

    loadConductivity();

    bool noactive = (actd.size() == 0);
    double minj = 100e-7 * js; // assume no significant heating below this current

    do {
        setMatrix(A, potential, bvoltage);   // corr holds RHS now
        solveMatrix(A, potential);

        err = 0.;
        double mcur = 0.;
        for (auto el: mesh->elements) {
            size_t i = el.getIndex();
            size_t lll = el.getLoLoLoIndex();
            size_t llu = el.getLoLoUpIndex();
            size_t lul = el.getLoUpLoIndex();
            size_t luu = el.getLoUpUpIndex();
            size_t ull = el.getUpLoLoIndex();
            size_t ulu = el.getUpLoUpIndex();
            size_t uul = el.getUpUpLoIndex();
            size_t uuu = el.getUpUpUpIndex();
            auto cur = vec(
                - 0.025 * conds[i].c00 * (- potential[lll] - potential[llu] - potential[lul] - potential[luu]
                                          + potential[ull] + potential[ulu] + potential[uul] + potential[uuu])
                    / (el.getUpper0() - el.getLower0()), // [j] = kA/cm²
                - 0.025 * conds[i].c00 * (- potential[lll] - potential[llu] + potential[lul] + potential[luu]
                                          - potential[ull] - potential[ulu] + potential[uul] + potential[uuu])
                    / (el.getUpper1() - el.getLower1()), // [j] = kA/cm²
                - 0.025 * conds[i].c11  * (- potential[lll] + potential[llu] - potential[lul] + potential[luu]
                                           - potential[ull] + potential[ulu] - potential[uul] + potential[uuu])
                    / (el.getUpper2() - el.getLower2()) // [j] = kA/cm²
            );
            if (noactive || isActive(el)) {
                double acur = abs2(cur);
                if (acur > mcur) { mcur = acur; maxcur = cur; }
            }
            double delta = abs2(current[i] - cur);
            if (delta > err) err = delta;
            current[i] = cur;
        }
        mcur = sqrt(mcur);
        err = 100. * sqrt(err) / mcur;
        
        if ((loop != 0 || mcur >= minj) && err > toterr) toterr = err;

        ++loopno;
        ++loop;

        this->writelog(LOG_RESULT, "Loop %d(%d): max(j%s) = %g kA/cm2, error = %g %%",
                       loop, loopno, noactive?"":"@junc", mcur, err);

    } while (err > maxerr && (loops == 0 || loop < loops));

    saveConductivity();

    outPotential.fireChanged();
    outCurrentDensity.fireChanged();
    outHeat.fireChanged();

    return toterr;
}


double FiniteElementMethodElectrical3DSolver::compute(unsigned loops) {
    switch (algorithm) {
        case ALGORITHM_CHOLESKY: return doCompute<DpbMatrix>(loops);
        case ALGORITHM_ITERATIVE: return doCompute<SparseBandMatrix>(loops);
    }
    return 0.;
}



void FiniteElementMethodElectrical3DSolver::solveMatrix(DpbMatrix& A, DataVector<double>& B)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    int info = 0;

    // Factorize matrix
    dpbtrf(UPLO, A.size, A.kd, A.data, A.ld+1, info);
    if (info < 0)
        throw CriticalException("%1%: Argument %2% of dpbtrf has illegal value", getId(), -info);
    if (info > 0)
        throw ComputationError(getId(), "Leading minor of order %1% of the stiffness matrix is not positive-definite", info);

    // Find solutions
    dpbtrs(UPLO, A.size, A.kd, 1, A.data, A.ld+1, B.data(), B.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", getId(), -info);

    // now A contains factorized matrix and B the solutions
}


void FiniteElementMethodElectrical3DSolver::solveMatrix(SparseBandMatrix& A, DataVector<double>& B)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    PrecondJacobi precond(A);

    DataVector<double> X = potential.copy(); // We use previous potential as initial solution
    double err;
    try {
        int iter = solveDCG(A, precond, X.data(), B.data(), err, iterlim, itererr, logfreq, getId());
        this->writelog(LOG_DETAIL, "Conjugate gradient converged after %1% iterations.", iter);
    } catch (DCGError err) {
        throw ComputationError(getId(), "Conjugate gradient failed:, %1%", err.what());
    }

    B = X;

    // now A contains factorized matrix and B the solutions
}


void FiniteElementMethodElectrical3DSolver::saveHeatDensity()
{
    this->writelog(LOG_DETAIL, "Computing heat densities");

    heat.reset(mesh->elements.size());

    if (heatmet == HEAT_JOULES) {
        for (auto el: mesh->elements) {
            size_t i = el.getIndex();
            size_t lll = el.getLoLoLoIndex();
            size_t llu = el.getLoLoUpIndex();
            size_t lul = el.getLoUpLoIndex();
            size_t luu = el.getLoUpUpIndex();
            size_t ull = el.getUpLoLoIndex();
            size_t ulu = el.getUpLoUpIndex();
            size_t uul = el.getUpUpLoIndex();
            size_t uuu = el.getUpUpUpIndex();
            double dvx = - 0.25e6 * (- potential[lll] - potential[llu] - potential[lul] - potential[luu]
                                     + potential[ull] + potential[ulu] + potential[uul] + potential[uuu])
                                / (el.getUpper0() - el.getLower0()); // 1e6 - from µm to m
            double dvy = - 0.25e6 * (- potential[lll] - potential[llu] + potential[lul] + potential[luu]
                                     - potential[ull] - potential[ulu] + potential[uul] + potential[uuu])
                                / (el.getUpper1() - el.getLower1()); // 1e6 - from µm to m
            double dvz = - 0.25e6 * (- potential[lll] + potential[llu] - potential[lul] + potential[luu]
                                     - potential[ull] + potential[ulu] - potential[uul] + potential[uuu])
                                / (el.getUpper2() - el.getLower2()); // 1e6 - from µm to m
            auto midpoint = el.getMidpoint();
            if (geometry->getMaterial(midpoint)->kind() == Material::NONE || geometry->hasRoleAt("noheat", midpoint))
                heat[i] = 0.;
            else {
                heat[i] = conds[i].c00 * dvx*dvx + conds[i].c00 * dvy*dvy + conds[i].c11 * dvz*dvz;
            }
        }
    } else {
        for (auto el: mesh->elements) {
            size_t i = el.getIndex();
            size_t lll = el.getLoLoLoIndex();
            size_t llu = el.getLoLoUpIndex();
            size_t lul = el.getLoUpLoIndex();
            size_t luu = el.getLoUpUpIndex();
            size_t ull = el.getUpLoLoIndex();
            size_t ulu = el.getUpLoUpIndex();
            size_t uul = el.getUpUpLoIndex();
            size_t uuu = el.getUpUpUpIndex();
            double dvx = - 0.25e6 * (- potential[lll] - potential[llu] - potential[lul] - potential[luu]
                                     + potential[ull] + potential[ulu] + potential[uul] + potential[uuu])
                                / (el.getUpper0() - el.getLower0()); // 1e6 - from µm to m
            double dvy = - 0.25e6 * (- potential[lll] - potential[llu] + potential[lul] + potential[luu]
                                     - potential[ull] - potential[ulu] + potential[uul] + potential[uuu])
                                / (el.getUpper1() - el.getLower1()); // 1e6 - from µm to m
            double dvz = - 0.25e6 * (- potential[lll] + potential[llu] - potential[lul] + potential[luu]
                                     - potential[ull] + potential[ulu] - potential[uul] + potential[uuu])
                                / (el.getUpper2() - el.getLower2()); // 1e6 - from µm to m
            auto midpoint = el.getMidpoint();
            auto roles = this->geometry->getRolesAt(midpoint);
            if (roles.find("active") != roles.end() || roles.find("junction") != roles.end()) {
                size_t nact = std::upper_bound(acthi.begin(), acthi.end(), mesh->index2(i)) - acthi.begin();
                assert(nact < acthi.size());
                double heatfact = 1e15 * phys::h_J * phys::c / (phys::qe * real(inWavelength(0)) * actd[nact]);
                double jz = conds[i].c11 * fabs(dvz); // [j] = A/m²
                heat[i] = heatfact * jz ;
            } else if (geometry->getMaterial(midpoint)->kind() == Material::NONE || roles.find("noheat") != roles.end())
                heat[i] = 0.;
            else
                heat[i] = conds[i].c00 * dvx*dvx + conds[i].c00 * dvy*dvy + conds[i].c11 * dvz*dvz;
        }
    }
}


double FiniteElementMethodElectrical3DSolver::integrateCurrent(size_t vindex)
{
    if (!potential) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis0.size()-1; ++i) {
        for (size_t j = 0; j < mesh->axis1.size()-1; ++j) {
            auto element = mesh->elements(i, j, vindex);
            result += current[element.getIndex()].c2 * element.getSize0() * element.getSize1();
        }
    }
    if (geometry->isSymmetric(Geometry::DIRECTION_LONG)) result *= 2.;
    if (geometry->isSymmetric(Geometry::DIRECTION_TRAN)) result *= 2.;
    return result * 0.01; // kA/cm² µm² -->  mA
}

double FiniteElementMethodElectrical3DSolver::getTotalCurrent(size_t nact)
{
    if (nact >= actlo.size()) throw BadInput(this->getId(), "Wrong active region number");
    // Find the average of the active region
    size_t level = (actlo[nact] + acthi[nact]) / 2;
    return integrateCurrent(level);
}


DataVector<const double> FiniteElementMethodElectrical3DSolver::getPotential(const MeshD<3>& dst_mesh, InterpolationMethod method) const {
    if (!potential) throw NoValue("Potential");
    this->writelog(LOG_DETAIL, "Getting potential");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    return interpolate(*(mesh), potential, WrappedMesh<3>(dst_mesh, geometry), method);
}


DataVector<const Vec<3> > FiniteElementMethodElectrical3DSolver::getCurrentDensity(const MeshD<3>& dst_mesh, InterpolationMethod method) {
    if (!potential) throw NoValue("Current density");
    this->writelog(LOG_DETAIL, "Getting current density");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    auto dest_mesh = WrappedMesh<3>(dst_mesh, geometry);
    auto result = interpolate(*(mesh->getMidpointsMesh()), current, dest_mesh, method);
    constexpr Vec<3> zero(0.,0.,0.);
    for (size_t i = 0; i < result.size(); ++i)
        if (!geometry->getChildBoundingBox().contains(dest_mesh[i])) result[i] = zero;
    return result;
}


DataVector<const double> FiniteElementMethodElectrical3DSolver::getHeatDensity(const MeshD<3>& dst_mesh, InterpolationMethod method) {
    if (!potential) throw NoValue("Heat density");
    this->writelog(LOG_DETAIL, "Getting heat density");
    if (!heat) saveHeatDensity(); // we will compute heats only if they are needed
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    auto dest_mesh = WrappedMesh<3>(dst_mesh, geometry);
    auto result = interpolate(*(mesh->getMidpointsMesh()), heat, dest_mesh, method);
    for (size_t i = 0; i < result.size(); ++i)
        if (!geometry->getChildBoundingBox().contains(dest_mesh[i])) result[i] = 0.;
    return result;
}

}}} // namespace plask::solvers::electrical
