#include "femV.h"

namespace plask { namespace solvers { namespace electrical {

template<typename Geometry2DType> FiniteElementMethodElectrical2DSolver<Geometry2DType>::FiniteElementMethodElectrical2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DType, RectilinearMesh2D>(name),
    js(1.),
    beta(20.),
    pcond(5.),
    ncond(50.),
    loopno(0),
    default_junction_conductivity(5.),
    maxerr(0.05),
    heatmet(HEAT_JOULES),
    outPotential(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getPotentials),
    outCurrentDensity(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getCurrentDensities),
    outHeat(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getHeatDensities),
    algorithm(ALGORITHM_CHOLESKY),
    itererr(1e-8),
    iterlim(10000),
    logfreq(500)
{
    onInvalidate();
    inTemperature = 300.;
    junction_conductivity.reset(1, default_junction_conductivity);
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "voltage" || param == "potential")
            this->readBoundaryConditions(manager, source, voltage_boundary);

        else if (param == "loop") {
            maxerr = source.getAttribute<double>("maxerr", maxerr);
            source.requireTagEnd();
        }

        else if (param == "matrix") {
            algorithm = source.enumAttribute<Algorithm>("algorithm")
                .value("cholesky", ALGORITHM_CHOLESKY)
                .value("gauss", ALGORITHM_GAUSS)
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
            this->parseStandardConfiguration(source, manager);
    }
}


template<typename Geometry2DType> FiniteElementMethodElectrical2DSolver<Geometry2DType>::~FiniteElementMethodElectrical2DSolver() {
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::setActiveRegions()
{
    if (!this->geometry || !this->mesh) {
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

    shared_ptr<RectilinearMesh2D> points = this->mesh->getMidpointsMesh();

    size_t ileft = 0, iright = points->axis0.size();
    bool in_active = false;

    for (size_t r = 0; r < points->axis1.size(); ++r) {
        bool had_active = false;
        for (size_t c = 0; c < points->axis0.size(); ++c) { // In the (possible) active region
            auto point = points->at(c,r);
            bool active = isActive(point);

            if (c < ileft) {
                if (active) throw Exception("%1%: Left edge of the active region not aligned.", this->getId());
            } else if (c >= iright) {
                if (active) throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
            } else {
                // Here we are inside potential active region
                if (active) {
                    if (!had_active) {
                        if (!in_active) { // active region is starting set-up new region info
                            ileft = c;
                            actlo.push_back(r);
                        }
                    }
                } else if (had_active) {
                    if (!in_active) iright = c;
                    else throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
                }
                had_active |= active;
            }
        }
        in_active = had_active;

        // Test if the active region has finished
        if (!in_active && actlo.size() != acthi.size()) {
            acthi.push_back(r);
            actd.push_back(this->mesh->axis1[acthi.back()] - this->mesh->axis1[actlo.back()]);
            this->writelog(LOG_DETAIL, "Detected active layer %2% thickness = %1%nm", 1e3 * actd.back(), actd.size()-1);
        }
    }

    // Test if the active region has finished
    if (actlo.size() != acthi.size()) {
        acthi.push_back(points->axis1.size());
        actd.push_back(this->mesh->axis1[acthi.back()] - this->mesh->axis1[actlo.back()]);
        this->writelog(LOG_DETAIL, "Detected active layer %2% thickness = %1%nm", 1e3 * actd.back(), actd.size()-1);
    }

    assert(acthi.size() == actlo.size());

    size_t condsize = max(actlo.size() * (this->mesh->axis0.size()-1), size_t(1));

    if (junction_conductivity.size() != condsize) {
        double condy = 0.;
        for (auto cond: junction_conductivity) condy += cond;
        junction_conductivity.reset(condsize, condy / junction_conductivity.size());
    }
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    loopno = 0;
    size = this->mesh->size();
    potentials.reset(size, 0.);
    currents.reset(this->mesh->elements.size(), vec(0.,0.));
    conds.reset(this->mesh->elements.size());
    if (junction_conductivity.size() == 1) {
        size_t condsize = max(actlo.size() * (this->mesh->axis0.size()-1), size_t(1));
        junction_conductivity.reset(condsize, junction_conductivity[0]);
    }
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::onInvalidate() {
    conds.reset();
    potentials.reset();
    currents.reset();
    heats.reset();
    junction_conductivity.reset(1, default_junction_conductivity);
}


template<>
inline void FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::setLocalMatrix(double&, double&, double&, double&,
                            double&, double&, double&, double&, double&, double&,
                            double, double, const Vec<2,double>&) {
    return;
}

template<>
inline void FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::setLocalMatrix(double& k44, double& k33, double& k22, double& k11,
                               double& k43, double& k21, double& k42, double& k31, double& k32, double& k41,
                               double ky, double width, const Vec<2,double>& midpoint) {
        double r = midpoint.rad_r();
        double dkr = ky * width / 12.;
        k44 = r * k44 - dkr;
        k33 = r * k33 + dkr;
        k22 = r * k22 + dkr;
        k11 = r * k11 - dkr;
        k43 = r * k43;
        k21 = r * k21;
        k42 = r * k42;
        k31 = r * k31;
        k32 = r * k32 - dkr;
        k41 = r * k41 + dkr;
}

template<typename Geometry2DType> template <typename MatrixT>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::applyBC(MatrixT& A, DataVector<double>& B,
                                                                    const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& bvoltage) {
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

template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::applyBC(SparseBandMatrix& A, DataVector<double>& B,
                                                                    const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& bvoltage) {
    // boundary conditions of the first kind
    for (auto cond: bvoltage) {
        for (auto r: cond.place) {
            double* rdata = A.data + LDA*r;
            *rdata = 1.;
            register double val = B[r] = cond.value;
            // below diagonal
            for (register ptrdiff_t i = 4; i > 0; --i) {
                register ptrdiff_t c = r - A.bno[i];
                if (c >= 0) {
                    B[c] -= A.data[LDA*c+i] * val;
                    A.data[LDA*c+i] = 0.;
                }
            }
            // above diagonal
            for (register ptrdiff_t i = 1; i < 5; ++i) {
                register ptrdiff_t c = r + A.bno[i];
                if (c < A.size) {
                    B[c] -= rdata[i] * val;
                    rdata[i] = 0.;
                }
            }
        }
    }
}

/// Set stiffness matrix + load vector
template<typename Geometry2DType> template <typename MatrixT>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::setMatrix(MatrixT& A, DataVector<double>& B,
                                                                      const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& bvoltage)
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, bands=%2%{%3%})", A.size, A.kd+1, A.ld+1);

    // Update junction conductivities
    if (loopno != 0) {
        for (auto e: this->mesh->elements) {
            if (isActive(e)) {
                size_t i = e.getIndex();
                size_t left = this->mesh->index0(e.getLoLoIndex());
                size_t right = this->mesh->index0(e.getUpLoIndex());
                size_t nact = std::upper_bound(acthi.begin(), acthi.end(), this->mesh->index1(e.getLoLoIndex())) - acthi.begin();
                assert(nact < acthi.size());
                double jy = 0.5e6 * conds[i].c11 *
                    abs( - potentials[this->mesh->index(left, actlo[nact])] - potentials[this->mesh->index(right, actlo[nact])]
                         + potentials[this->mesh->index(left, acthi[nact])] + potentials[this->mesh->index(right, acthi[nact])]
                    ) / actd[nact]; // [j] = A/m²
                conds[i] = Tensor2<double>(0., 1e-6 * beta * jy * actd[nact] / log(jy / js + 1.));
            }
        }
    }

    std::fill_n(A.data, A.size*(A.ld+1), 0.); // zero the matrix
    B.fill(0.);

    // Set stiffness matrix and load vector
    for (auto e: this->mesh->elements) {

        size_t i = e.getIndex();

        // nodes numbers for the current element
        size_t loleftno = e.getLoLoIndex();
        size_t lorghtno = e.getUpLoIndex();
        size_t upleftno = e.getLoUpIndex();
        size_t uprghtno = e.getUpUpIndex();

        // element size
        double elemwidth = e.getUpper0() - e.getLower0();
        double elemheight = e.getUpper1() - e.getLower1();

        Vec<2,double> midpoint = e.getMidpoint();

        double kx = conds[i].c00;
        double ky = conds[i].c11;

        kx *= elemheight; kx /= elemwidth;
        ky *= elemwidth; ky /= elemheight;

        // set symmetric matrix components
        double k44, k33, k22, k11, k43, k21, k42, k31, k32, k41;

        k44 = k33 = k22 = k11 = (kx + ky) / 3.;
        k43 = k21 = (-2. * kx + ky) / 6.;
        k42 = k31 = - (kx + ky) / 6.;
        k32 = k41 = (kx - 2. * ky) / 6.;

        // set stiffness matrix
        setLocalMatrix(k44, k33, k22, k11, k43, k21, k42, k31, k32, k41, ky, elemwidth, midpoint);

        A(loleftno, loleftno) += k11;
        A(lorghtno, lorghtno) += k22;
        A(uprghtno, uprghtno) += k33;
        A(upleftno, upleftno) += k44;

        A(lorghtno, loleftno) += k21;
        A(uprghtno, loleftno) += k31;
        A(upleftno, loleftno) += k41;
        A(uprghtno, lorghtno) += k32;
        A(upleftno, lorghtno) += k42;
        A(upleftno, uprghtno) += k43;
    }

    // boundary conditions of the first kind
    applyBC(A, B, bvoltage);

#ifndef NDEBUG
    double* aend = A.data + A.size * A.kd;
    for (double* pa = A.data; pa != aend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(this->getId(), "Error in stiffness matrix at position %1% (%2%)", pa-A.data, isnan(*pa)?"nan":"inf");
    }
#endif

}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::loadConductivities()
{
    auto midmesh = (this->mesh)->getMidpointsMesh();
    auto temperature = inTemperature(midmesh);

    for (auto e: this->mesh->elements)
    {
        size_t i = e.getIndex();
        Vec<2,double> midpoint = e.getMidpoint();

        auto roles = this->geometry->getRolesAt(midpoint);
        if (roles.find("active") != roles.end() || roles.find("junction") != roles.end()) {
            size_t n = std::upper_bound(acthi.begin(), acthi.end(), this->mesh->index1(i)) - acthi.begin();
            assert(n < acthi.size());
            conds[i] = Tensor2<double>(0., junction_conductivity[n * (this->mesh->axis0.size()-1) + e.getIndex0()]);
        } else if (roles.find("p-contact") != roles.end()) {
            conds[i] = Tensor2<double>(pcond, pcond);
        } else if (roles.find("n-contact") != roles.end()) {
            conds[i] = Tensor2<double>(ncond, ncond);
        } else
            conds[i] = this->geometry->getMaterial(midpoint)->cond(temperature[i]);
    }
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::saveConductivities()
{
    for (size_t n = 0; n < getActNo(); ++n)
        for (size_t i = 0, j = (actlo[n]+acthi[n])/2; i != this->mesh->axis0.size()-1; ++i)
            junction_conductivity[n * (this->mesh->axis0.size()-1) + i] = conds[this->mesh->elements(i,j).getIndex()].c11;
}


template<typename Geometry2DType> double FiniteElementMethodElectrical2DSolver<Geometry2DType>::compute(unsigned loops) {
    switch (algorithm) {
        case ALGORITHM_CHOLESKY: return doCompute<DpbMatrix>(loops);
        case ALGORITHM_GAUSS: return doCompute<DgbMatrix>(loops);
        case ALGORITHM_ITERATIVE: return doCompute<SparseBandMatrix>(loops);
    }
    return 0.;
}

template<typename Geometry2DType> template <typename MatrixT>
double FiniteElementMethodElectrical2DSolver<Geometry2DType>::doCompute(unsigned loops)
{
    this->initCalculation();

    heats.reset();

    // Store boundary conditions for current mesh
    auto vconst = voltage_boundary(this->mesh, this->geometry);

    this->writelog(LOG_INFO, "Running electrical calculations");

    unsigned loop = 0;

    MatrixT A(size, this->mesh->minorAxis().size());

    double err = 0.;
    toterr = 0.;

#   ifndef NDEBUG
        if (!potentials.unique()) this->writelog(LOG_DEBUG, "Potential data held by something else...");
#   endif
    potentials = potentials.claim();

    loadConductivities();

    bool noactive = (actd.size() == 0);

    do {
        setMatrix(A, potentials, vconst);    // corr holds RHS now
        solveMatrix(A, potentials);

        err = 0.;
        double mcur = 0.;
        for (auto el: this->mesh->elements) {
            size_t i = el.getIndex();
            size_t loleftno = el.getLoLoIndex();
            size_t lorghtno = el.getUpLoIndex();
            size_t upleftno = el.getLoUpIndex();
            size_t uprghtno = el.getUpUpIndex();
            double dvx = - 0.05 * (- potentials[loleftno] + potentials[lorghtno] - potentials[upleftno] + potentials[uprghtno])
                                 / (el.getUpper0() - el.getLower0()); // [j] = kA/cm²
            double dvy = - 0.05 * (- potentials[loleftno] - potentials[lorghtno] + potentials[upleftno] + potentials[uprghtno])
                                 / (el.getUpper1() - el.getLower1()); // [j] = kA/cm²
            auto cur = vec(conds[i].c00 * dvx, conds[i].c11 * dvy);
            if (noactive || isActive(el)) {
                double acur = abs2(cur);
                if (acur > mcur) { mcur = acur; maxcur = cur; }
            }
            double delta = abs2(currents[i] - cur);
            if (delta > err) err = delta;
            currents[i] = cur;
        }
        err = 100. * sqrt(err / max(mcur,1e-8*js));
        if (err > toterr) toterr = err;

        ++loopno;
        ++loop;

        this->writelog(LOG_RESULT, "Loop %d(%d): max(j%s) = %g kA/cm2, error = %g %%",
                       loop, loopno, noactive?"":"@junc", sqrt(mcur), err);

    } while (err > maxerr && (loops == 0 || loop < loops));

    saveConductivities();

    outPotential.fireChanged();
    outCurrentDensity.fireChanged();
    outHeat.fireChanged();

    return toterr;
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(DpbMatrix& A, DataVector<double>& B)
{
    int info = 0;

    this->writelog(LOG_DETAIL, "Solving matrix system");

    // Factorize matrix
    dpbtrf(UPLO, A.size, A.kd, A.data, A.ld+1, info);
    if (info < 0)
        throw CriticalException("%1%: Argument %2% of dpbtrf has illegal value", this->getId(), -info);
    else if (info > 0)
        throw ComputationError(this->getId(), "Leading minor of order %1% of the stiffness matrix is not positive-definite", info);

    // Find solutions
    dpbtrs(UPLO, A.size, A.kd, 1, A.data, A.ld+1, B.data(), B.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", this->getId(), -info);

    // now A contains factorized matrix and B the solutions
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(DgbMatrix& A, DataVector<double>& B)
{
    int info = 0;
    this->writelog(LOG_DETAIL, "Solving matrix system");
    int* ipiv = aligned_malloc<int>(A.size);

    A.mirror();

    // Factorize matrix
    dgbtrf(A.size, A.size, A.kd, A.kd, A.data, A.ld+1, ipiv, info);
    if (info < 0) {
        aligned_free(ipiv);
        throw CriticalException("%1%: Argument %2% of dgbtrf has illegal value", this->getId(), -info);
    } else if (info > 0) {
        aligned_free(ipiv);
        throw ComputationError(this->getId(), "Matrix is singlar (at %1%)", info);
    }

    // Find solutions
    dgbtrs('N', A.size, A.kd, A.kd, 1, A.data, A.ld+1, ipiv, B.data(), B.size(), info);
    aligned_free(ipiv);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dgbtrs has illegal value", this->getId(), -info);

    // now A contains factorized matrix and B the solutions
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(SparseBandMatrix& ioA, DataVector<double>& B)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    PrecondJacobi precond(ioA);

    DataVector<double> x = potentials.copy(); // We use previous potentials as initial solution
    double err;
    try {
        int iter = solveDCG(ioA, precond, x.data(), B.data(), err, iterlim, itererr, logfreq, this->getId());
        this->writelog(LOG_DETAIL, "Conjugate gradient converged after %1% iterations.", iter);
    } catch (DCGError exc) {
        throw ComputationError(this->getId(), "Conjugate gradient failed:, %1%", exc.what());
    }

    B = x;

    // now A contains factorized matrix and B the solutions
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::saveHeatDensities()
{
    this->writelog(LOG_DETAIL, "Computing heat densities");

    heats.reset(this->mesh->elements.size());

    if (heatmet == HEAT_JOULES) {
        for (auto e: this->mesh->elements) {
            size_t i = e.getIndex();
            size_t loleftno = e.getLoLoIndex();
            size_t lorghtno = e.getUpLoIndex();
            size_t upleftno = e.getLoUpIndex();
            size_t uprghtno = e.getUpUpIndex();
            auto midpoint = e.getMidpoint();
            if (this->geometry->getMaterial(midpoint)->kind() == Material::NONE || this->geometry->hasRoleAt("noheat", midpoint))
                heats[i] = 0.;
            else {
                double dvx = 0.5e6 * (- potentials[loleftno] + potentials[lorghtno] - potentials[upleftno] + potentials[uprghtno])
                                    / (e.getUpper0() - e.getLower0()); // [grad(dV)] = V/m
                double dvy = 0.5e6 * (- potentials[loleftno] - potentials[lorghtno] + potentials[upleftno] + potentials[uprghtno])
                                    / (e.getUpper1() - e.getLower1()); // [grad(dV)] = V/m
                heats[i] = conds[i].c00 * dvx*dvx + conds[i].c11 * dvy*dvy;
            }
        }
    } else {
        for (auto e: this->mesh->elements) {
            size_t i = e.getIndex();
            size_t loleftno = e.getLoLoIndex();
            size_t lorghtno = e.getUpLoIndex();
            size_t upleftno = e.getLoUpIndex();
            size_t uprghtno = e.getUpUpIndex();
            double dvx = 0.5e6 * (- potentials[loleftno] + potentials[lorghtno] - potentials[upleftno] + potentials[uprghtno])
                                / (e.getUpper0() - e.getLower0()); // [grad(dV)] = V/m
            double dvy = 0.5e6 * (- potentials[loleftno] - potentials[lorghtno] + potentials[upleftno] + potentials[uprghtno])
                                / (e.getUpper1() - e.getLower1()); // [grad(dV)] = V/m
            auto midpoint = e.getMidpoint();
            auto roles = this->geometry->getRolesAt(midpoint);
            if (roles.find("active") != roles.end() || roles.find("junction") != roles.end()) {
                size_t nact = std::upper_bound(acthi.begin(), acthi.end(), this->mesh->index1(i)) - acthi.begin();
                assert(nact < acthi.size());
                double heatfact = 1e15 * phys::h_J * phys::c / (phys::qe * real(inWavelength(0)) * actd[nact]);
                double jy = conds[i].c11 * fabs(dvy); // [j] = A/m²
                heats[i] = heatfact * jy ;
            } else if (this->geometry->getMaterial(midpoint)->kind() == Material::NONE || roles.find("noheat") != roles.end())
                heats[i] = 0.;
            else
                heats[i] = conds[i].c00 * dvx*dvx + conds[i].c11 * dvy*dvy;
        }
    }
}


template<> double FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::integrateCurrent(size_t vindex)
{
    if (!potentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis0.size()-1; ++i) {
        auto element = mesh->elements(i, vindex);
        result += currents[element.getIndex()].c1 * element.getSize0();
    }
    if (this->getGeometry()->isSymmetric(Geometry::DIRECTION_TRAN)) result *= 2.;
    return result * geometry->getExtrusion()->getLength() * 0.01; // kA/cm² µm² -->  mA;
}


template<> double FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::integrateCurrent(size_t vindex)
{
    if (!potentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis0.size()-1; ++i) {
        auto element = mesh->elements(i, vindex);
        double rin = element.getLower0(), rout = element.getUpper0();
        result += currents[element.getIndex()].c1 * (rout*rout - rin*rin);
    }
    return result * M_PI * 0.01; // kA/cm² µm² -->  mA
}


template<typename Geometry2DType> double FiniteElementMethodElectrical2DSolver<Geometry2DType>::getTotalCurrent(size_t nact)
{
    if (nact >= actlo.size()) throw BadInput(this->getId(), "Wrong active region number");
    // Find the average of the active region
    size_t level = (actlo[nact] + acthi[nact]) / 2;
    return integrateCurrent(level);
}


template<typename Geometry2DType> DataVector<const double> FiniteElementMethodElectrical2DSolver<Geometry2DType>::getPotentials(const MeshD<2>& dst_mesh, InterpolationMethod method) const
{
    if (!potentials) throw NoValue("Potential");
    this->writelog(LOG_DETAIL, "Getting potentials");
    if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
    return interpolate(*(this->mesh), potentials, WrappedMesh<2>(dst_mesh, this->geometry), method);
}


template<typename Geometry2DType> DataVector<const Vec<2> > FiniteElementMethodElectrical2DSolver<Geometry2DType>::getCurrentDensities(const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    if (!potentials) throw NoValue("Current density");
    this->writelog(LOG_DETAIL, "Getting current densities");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    auto dest_mesh = WrappedMesh<2>(dst_mesh, this->geometry);
    auto result = interpolate(*(this->mesh->getMidpointsMesh()), currents, dest_mesh, method);
    constexpr Vec<2> zero(0.,0.);
    for (size_t i = 0; i < result.size(); ++i)
        if (!this->geometry->getChildBoundingBox().contains(dest_mesh[i])) result[i] = zero;
    return result;
}

template<typename Geometry2DType> DataVector<const double> FiniteElementMethodElectrical2DSolver<Geometry2DType>::getHeatDensities(const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    if (!potentials) throw NoValue("Heat density");
    this->writelog(LOG_DETAIL, "Getting heat density");
    if (!heats) saveHeatDensities(); // we will compute heats only if they are needed
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    auto dest_mesh = WrappedMesh<2>(dst_mesh, this->geometry);
    auto result = interpolate(*(this->mesh->getMidpointsMesh()), heats, dest_mesh, method);
    for (size_t i = 0; i < result.size(); ++i)
        if (!this->geometry->getChildBoundingBox().contains(dest_mesh[i])) result[i] = 0.;
    return result;
}



template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::getClassName() const { return "electrical.Shockley2D"; }
template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::getClassName() const { return "electrical.ShockleyCyl"; }

template struct FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>;
template struct FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>;

}}} // namespace plask::solvers::thermal
