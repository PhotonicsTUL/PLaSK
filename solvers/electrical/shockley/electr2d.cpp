#include "electr2d.h"

namespace plask { namespace electrical { namespace shockley {

template<typename Geometry2DType>
FiniteElementMethodElectrical2DSolver<Geometry2DType>::FiniteElementMethodElectrical2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DType, RectangularMesh<2>>(name),
    pcond(5.),
    ncond(50.),
    loopno(0),
    default_junction_conductivity(5.),
    maxerr(0.05),
    heatmet(HEAT_JOULES),
    outVoltage(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getVoltage),
    outCurrentDensity(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getCurrentDensities),
    outHeat(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getHeatDensities),
    outConductivity(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getConductivity),
    algorithm(ALGORITHM_CHOLESKY),
    itererr(1e-8),
    iterlim(10000),
    logfreq(500)
{
    js.assign(1, 1.),
    beta.assign(1, NAN),
    onInvalidate();
    inTemperature = 300.;
    junction_conductivity.reset(1, default_junction_conductivity);
}

template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "potential")
            source.throwException("<potential> boundary conditions have been permanently renamed to <voltage>");

        if (param == "voltage")
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
            js[0] = source.getAttribute<double>("js", js[0]);
            beta[0] = source.getAttribute<double>("beta", beta[0]);
            auto condjunc = source.getAttribute<double>("pnjcond");
            if (condjunc) setCondJunc(*condjunc);
            auto wavelength = source.getAttribute<double>("wavelength");
            if (wavelength) inWavelength = *wavelength;
            heatmet = source.enumAttribute<HeatMethod>("heat")
                .value("joules", HEAT_JOULES)
                .value("wavelength", HEAT_BANDGAP)
                .get(heatmet);
            for (auto attr: source.getAttributes()) {
                if (attr.first == "beta" || attr.first == "Vt" || attr.first == "js" || attr.first == "pnjcond" || attr.first == "wavelength" || attr.first == "heat") continue;
                if (attr.first.substr(0,4) == "beta") {
                    size_t no;
                    try { no = boost::lexical_cast<size_t>(attr.first.substr(4)); }
                    catch (boost::bad_lexical_cast) { throw XMLUnexpectedAttrException(source, attr.first); }
                    setBeta(no, source.requireAttribute<double>(attr.first));
                }
                else if (attr.first.substr(0,2) == "js") {
                    size_t no;
                    try { no = boost::lexical_cast<size_t>(attr.first.substr(2)); }
                    catch (boost::bad_lexical_cast) { throw XMLUnexpectedAttrException(source, attr.first); }
                    setJs(no, source.requireAttribute<double>(attr.first));
                }
                else
                    throw XMLUnexpectedAttrException(source, attr.first);
            }
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


template<typename Geometry2DType>
FiniteElementMethodElectrical2DSolver<Geometry2DType>::~FiniteElementMethodElectrical2DSolver() {
}

template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::setActiveRegions()
{
    if (!this->geometry || !this->mesh) {
        if (junction_conductivity.size() != 1) {
            double condy = 0.;
            for (auto cond: junction_conductivity) condy += cond;
            junction_conductivity.reset(1, condy / double(junction_conductivity.size()));
        }
        return;
    }

    shared_ptr<RectangularMesh<2>> points = this->mesh->getMidpointsMesh();

    std::vector<typename Active::Region> regions;

    for (size_t r = 0; r < points->axis[1]->size(); ++r) {
        size_t prev = 0;
        shared_ptr<Material> material;
        for (size_t c = 0; c < points->axis[0]->size(); ++c) { // In the (possible) active region
            auto point = points->at(c,r);
            size_t num = isActive(point);

            if (num) { // here we are inside the active region
                if (regions.size() >= num && regions[num-1].warn) {
                    if (!material) material = this->geometry->getMaterial(points->at(c,r));
                    else if (*material != *this->geometry->getMaterial(points->at(c,r))) {
                        writelog(LOG_WARNING, "Junction {} is laterally non-uniform", num-1);
                        regions[num-1].warn = false;
                    }
                }
                regions.resize(max(regions.size(), num));
                auto& reg = regions[num-1];
                if (prev != num) { // this region starts in the current row
                    if (reg.top < r) {
                        throw Exception("{0}: Junction {1} is disjoint", this->getId(), num-1);
                    }
                    if (reg.bottom >= r) reg.bottom = r; // first row
                    else if (reg.rowr <= c) throw Exception("{0}: Active region {1} is disjoint", this->getId(), num-1);
                    reg.top = r + 1;
                    reg.rowl = c; if (reg.left > reg.rowl) reg.left = reg.rowl;
                }
            }
            if (prev && prev != num) { // previous region ended
                auto& reg = regions[prev-1];
                if (reg.bottom < r && reg.rowl >= c) throw Exception("{0}: Junction {1} is disjoint", this->getId(), prev-1);
                reg.rowr = c; if (reg.right < reg.rowr) reg.right = reg.rowr;
            }
            prev = num;
        }
        if (prev) // junction reached the edge
            regions[prev-1].rowr = regions[prev-1].right = points->axis[0]->size();
    }

    size_t condsize = 0;
    active.clear();
    active.reserve(regions.size());
    size_t i = 0;
    for (auto& reg: regions) {
        if (reg.bottom == size_t(-1)) reg.bottom = reg.top = 0;
        active.emplace_back(condsize, reg.left, reg.right, reg.bottom, reg.top, this->mesh->axis[1]->at(reg.top) - this->mesh->axis[1]->at(reg.bottom));
        condsize += reg.right - reg.left;
        this->writelog(LOG_DETAIL, "Detected junction {0} thickness = {1}nm", i++, 1e3 * active.back().height);
        this->writelog(LOG_DEBUG, "Junction {0} span: [{1},{3}]-[{2},{4}]", i-1, reg.left, reg.right, reg.bottom, reg.top);
    }

    if (junction_conductivity.size() != condsize) {
        double condy = 0.;
        for (auto cond: junction_conductivity) condy += cond;
        junction_conductivity.reset(max(condsize, size_t(1)), condy / double(junction_conductivity.size()));
    }
}


template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    loopno = 0;
    size = this->mesh->size();
    potentials.reset(size, 0.);
    currents.reset(this->mesh->getElementsCount(), vec(0.,0.));
    conds.reset(this->mesh->getElementsCount());
    if (junction_conductivity.size() == 1) {
        size_t condsize = 0;
        for (const auto& act: active) condsize += act.right - act.left;
        condsize = max(condsize, size_t(1));
        junction_conductivity.reset(condsize, junction_conductivity[0]);
    }
}


template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::onInvalidate() {
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
                               double, double, const Vec<2,double>& midpoint) {
        double r = midpoint.rad_r();
        k44 = r * k44;
        k33 = r * k33;
        k22 = r * k22;
        k11 = r * k11;
        k43 = r * k43;
        k21 = r * k21;
        k42 = r * k42;
        k31 = r * k31;
        k32 = r * k32;
        k41 = r * k41;
}

template<typename Geometry2DType>
template <typename MatrixT>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::applyBC(MatrixT& A, DataVector<double>& B,
                                                                    const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& bvoltage) {
    // boundary conditions of the first kind
    for (auto cond: bvoltage) {
        for (auto r: cond.place) {
            A(r,r) = 1.;
            double val = B[r] = cond.value;
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
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::applyBC(SparseBandMatrix2D& A, DataVector<double>& B,
                                                                    const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double> &bvoltage) {
    // boundary conditions of the first kind
    for (auto cond: bvoltage) {
        for (auto r: cond.place) {
            double* rdata = A.data + LDA*r;
            *rdata = 1.;
            double val = B[r] = cond.value;
            // below diagonal
            for (ptrdiff_t i = 4; i > 0; --i) {
                ptrdiff_t c = r - A.bno[i];
                if (c >= 0) {
                    B[c] -= A.data[LDA*c+i] * val;
                    A.data[LDA*c+i] = 0.;
                }
            }
            // above diagonal
            for (ptrdiff_t i = 1; i < 5; ++i) {
                ptrdiff_t c = r + A.bno[i];
                if (c < A.size) {
                    B[c] -= rdata[i] * val;
                    rdata[i] = 0.;
                }
            }
        }
    }
}

/// Set stiffness matrix + load vector
template<typename Geometry2DType>
template <typename MatrixT>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::setMatrix(MatrixT& A, DataVector<double>& B,
                                                                      const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& bvoltage)
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size={0}, bands={1}({2}))", A.size, A.kd+1, A.ld+1);

    // Update junction conductivities
    if (loopno != 0) {
        for (auto e: this->mesh->elements()) {
            if (size_t nact = isActive(e)) {
                size_t i = e.getIndex();
                size_t left = this->mesh->index0(e.getLoLoIndex());
                size_t right = this->mesh->index0(e.getUpLoIndex());
                const Active& act = active[nact-1];
                double jy = 0.5e6 * conds[i].c11 *
                    abs( - potentials[this->mesh->index(left, act.bottom)] - potentials[this->mesh->index(right, act.bottom)]
                         + potentials[this->mesh->index(left, act.top)] + potentials[this->mesh->index(right, act.top)]
                    ) / act.height; // [j] = A/m²
                conds[i] = Tensor2<double>(0., 1e-6 * getBeta(nact-1) * jy * act.height / log(jy / getJs(nact-1) + 1.));
                if (isnan(conds[i].c11) || abs(conds[i].c11) < 1e-16) conds[i].c11 = 1e-16;
            }
        }
    }

    std::fill_n(A.data, A.size*(A.ld+1), 0.); // zero the matrix
    B.fill(0.);

    // Set stiffness matrix and load vector
    for (auto e: this->mesh->elements()) {

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
            throw ComputationError(this->getId(), "Error in stiffness matrix at position {0} ({1})", pa-A.data, isnan(*pa)?"nan":"inf");
    }
#endif

}


template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::loadConductivities()
{
    auto midmesh = (this->mesh)->getMidpointsMesh();
    auto temperature = inTemperature(midmesh);

    for (auto e: this->mesh->elements())
    {
        size_t i = e.getIndex();
        Vec<2,double> midpoint = e.getMidpoint();

        auto roles = this->geometry->getRolesAt(midpoint);
        if (size_t actn = isActive(midpoint)) {
            const auto& act = active[actn-1];
            conds[i] = Tensor2<double>(0., junction_conductivity[act.offset + e.getIndex0()]);
            if (isnan(conds[i].c11) || abs(conds[i].c11) < 1e-16) conds[i].c11 = 1e-16;
        } else if (roles.find("p-contact") != roles.end()) {
            conds[i] = Tensor2<double>(pcond, pcond);
        } else if (roles.find("n-contact") != roles.end()) {
            conds[i] = Tensor2<double>(ncond, ncond);
        } else
            conds[i] = this->geometry->getMaterial(midpoint)->cond(temperature[i]);
    }
}

template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::saveConductivities()
{
    for (size_t n = 0; n < active.size(); ++n) {
        const auto& act = active[n];
        for (size_t i = act.left, r = (act.top + act.bottom)/2; i != act.right; ++i)
            junction_conductivity[act.offset + i] = conds[this->mesh->element(i,r).getIndex()].c11;
    }
}


template<typename Geometry2DType>
double FiniteElementMethodElectrical2DSolver<Geometry2DType>::compute(unsigned loops) {
    switch (algorithm) {
        case ALGORITHM_CHOLESKY: return doCompute<DpbMatrix>(loops);
        case ALGORITHM_GAUSS: return doCompute<DgbMatrix>(loops);
        case ALGORITHM_ITERATIVE: return doCompute<SparseBandMatrix2D>(loops);
    }
    return 0.;
}

template<typename Geometry2DType>
template <typename MatrixT>
double FiniteElementMethodElectrical2DSolver<Geometry2DType>::doCompute(unsigned loops)
{
    this->initCalculation();

    heats.reset();

    // Store boundary conditions for current mesh
    auto vconst = voltage_boundary(this->mesh, this->geometry);

    this->writelog(LOG_INFO, "Running electrical calculations");

    unsigned loop = 0;

    MatrixT A(size, this->mesh->minorAxis()->size());

    double err = 0.;
    toterr = 0.;

#   ifndef NDEBUG
        if (!potentials.unique()) this->writelog(LOG_DEBUG, "Voltage data held by something else...");
#   endif
    potentials = potentials.claim();

    loadConductivities();

    bool noactive = (active.size() == 0);
    double minj = js[0]; // assume no significant heating below this current
    for (auto j: js) if (j < minj) minj = j;
    minj *= 100e-7;

    do {
        setMatrix(A, potentials, vconst);    // corr holds RHS now
        solveMatrix(A, potentials);

        err = 0.;
        double mcur = 0.;
        for (auto el: this->mesh->elements()) {
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
        mcur = sqrt(mcur);
        err = 100. * sqrt(err) / max(mcur, minj);
        if ((loop != 0 || mcur >= minj) && err > toterr) toterr = err;

        ++loopno;
        ++loop;

        this->writelog(LOG_RESULT, "Loop {:d}({:d}): max(j{}) = {:g} kA/cm2, error = {:g}%",
                       loop, loopno, noactive?"":"@junc", mcur, err);

    } while (err > maxerr && (loops == 0 || loop < loops));

    saveConductivities();

    outVoltage.fireChanged();
    outCurrentDensity.fireChanged();
    outHeat.fireChanged();

    return toterr;
}


template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(DpbMatrix& A, DataVector<double>& B)
{
    int info = 0;

    this->writelog(LOG_DETAIL, "Solving matrix system");

    // Factorize matrix
    dpbtrf(UPLO, int(A.size), int(A.kd), A.data, int(A.ld)+1, info);
    if (info < 0)
        throw CriticalException("{0}: Argument {1} of dpbtrf has illegal value", this->getId(), -info);
    else if (info > 0)
        throw ComputationError(this->getId(), "Leading minor of order {0} of the stiffness matrix is not positive-definite", info);

    // Find solutions
    dpbtrs(UPLO, int(A.size), int(A.kd), 1, A.data, int(A.ld)+1, B.data(), int(B.size()), info);
    if (info < 0) throw CriticalException("{0}: Argument {1} of dpbtrs has illegal value", this->getId(), -info);

    // now A contains factorized matrix and B the solutions
}

template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(DgbMatrix& A, DataVector<double>& B)
{
    int info = 0;
    this->writelog(LOG_DETAIL, "Solving matrix system");
    aligned_unique_ptr<int> ipiv(aligned_malloc<int>(A.size));

    A.mirror();

    // Factorize matrix
    dgbtrf(int(A.size), int(A.size), int(A.kd), int(A.kd), A.data, int(A.ld+1), ipiv.get(), info);
    if (info < 0) {
        throw CriticalException("{0}: Argument {1} of dgbtrf has illegal value", this->getId(), -info);
    } else if (info > 0) {
        throw ComputationError(this->getId(), "Matrix is singlar (at {0})", info);
    }

    // Find solutions
    dgbtrs('N', int(A.size), int(A.kd), int(A.kd), 1, A.data, int(A.ld+1), ipiv.get(), B.data(), int(B.size()), info);
    if (info < 0) throw CriticalException("{0}: Argument {1} of dgbtrs has illegal value", this->getId(), -info);

    // now A contains factorized matrix and B the solutions
}

template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(SparseBandMatrix2D& A, DataVector<double>& B)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    PrecondJacobi2D precond(A);

    DataVector<double> x = potentials.copy(); // We use previous potentials as initial solution
    double err;
    try {
        std::size_t iter = solveDCG(A, precond, x.data(), B.data(), err, iterlim, itererr, logfreq, this->getId());
        this->writelog(LOG_DETAIL, "Conjugate gradient converged after {0} iterations.", iter);
    } catch (DCGError exc) {
        throw ComputationError(this->getId(), "Conjugate gradient failed:, {0}", exc.what());
    }

    B = x;

    // now A contains factorized matrix and B the solutions
}


template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::saveHeatDensities()
{
    this->writelog(LOG_DETAIL, "Computing heat densities");

    heats.reset(this->mesh->getElementsCount());

    if (heatmet == HEAT_JOULES) {
        for (auto e: this->mesh->elements()) {
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
        for (auto e: this->mesh->elements()) {
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
            if (size_t nact = isActive(midpoint)) {
                const auto& act = active[nact-1];
                double heatfact = 1e15 * phys::h_J * phys::c / (phys::qe * real(inWavelength(0)) * act.height);
                double jy = conds[i].c11 * fabs(dvy); // [j] = A/m²
                heats[i] = heatfact * jy ;
            } else if (this->geometry->getMaterial(midpoint)->kind() == Material::NONE || this->geometry->hasRoleAt("noheat", midpoint))
                heats[i] = 0.;
            else
                heats[i] = conds[i].c00 * dvx*dvx + conds[i].c11 * dvy*dvy;
        }
    }
}


template<> double FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::integrateCurrent(size_t vindex, bool onlyactive)
{
    if (!potentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis[0]->size()-1; ++i) {
        auto element = mesh->element(i, vindex);
        if (!onlyactive || isActive(element.getMidpoint()))
            result += currents[element.getIndex()].c1 * element.getSize0();
    }
    if (this->getGeometry()->isSymmetric(Geometry::DIRECTION_TRAN)) result *= 2.;
    return result * geometry->getExtrusion()->getLength() * 0.01; // kA/cm² µm² -->  mA;
}


template<> double FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::integrateCurrent(size_t vindex, bool onlyactive)
{
    if (!potentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis[0]->size()-1; ++i) {
        auto element = mesh->element(i, vindex);
        if (!onlyactive || isActive(element.getMidpoint())) {
            double rin = element.getLower0(), rout = element.getUpper0();
            result += currents[element.getIndex()].c1 * (rout*rout - rin*rin);
        }
    }
    return result * plask::PI * 0.01; // kA/cm² µm² -->  mA
}


template<typename Geometry2DType>
double FiniteElementMethodElectrical2DSolver<Geometry2DType>::getTotalCurrent(size_t nact)
{
    if (nact >= active.size()) throw BadInput(this->getId(), "Wrong active region number");
    const auto& act = active[nact];
    // Find the average of the active region
    size_t level = (act.bottom + act.top) / 2;
    return integrateCurrent(level, true);
}


template<typename Geometry2DType>
const LazyData<double> FiniteElementMethodElectrical2DSolver<Geometry2DType>::getVoltage(shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) const
{
    if (!potentials) throw NoValue("Voltage");
    this->writelog(LOG_DEBUG, "Getting voltage");
    if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh, potentials, dst_mesh, method, this->geometry);
}


template<typename Geometry2DType>
const LazyData<Vec<2>> FiniteElementMethodElectrical2DSolver<Geometry2DType>::getCurrentDensities(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method)
{
    if (!potentials) throw NoValue("Current density");
    this->writelog(LOG_DEBUG, "Getting current densities");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    InterpolationFlags flags(this->geometry, InterpolationFlags::Symmetry::NP, InterpolationFlags::Symmetry::PN);
    auto result = interpolate(this->mesh->getMidpointsMesh(), currents, dest_mesh, method, flags);
    return LazyData<Vec<2>>(result.size(),
        [this, dest_mesh, result, flags](size_t i) {
            return this->geometry->getChildBoundingBox().contains(flags.wrap(dest_mesh->at(i)))? result[i] : Vec<2>(0.,0.);
        }
    );
}


template<typename Geometry2DType>
const LazyData<double> FiniteElementMethodElectrical2DSolver<Geometry2DType>::getHeatDensities(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method)
{
    if (!potentials) throw NoValue("Heat density");
    this->writelog(LOG_DEBUG, "Getting heat density");
    if (!heats) saveHeatDensities(); // we will compute heats only if they are needed
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    InterpolationFlags flags(this->geometry);
    auto result = interpolate(this->mesh->getMidpointsMesh(), heats, dest_mesh, method, flags);
    return LazyData<double>(result.size(),
        [this, dest_mesh, result, flags](size_t i) {
            return this->geometry->getChildBoundingBox().contains(flags.wrap(dest_mesh->at(i)))? result[i] : 0.;
        }
    );
}


template<typename Geometry2DType>
const LazyData<Tensor2<double>> FiniteElementMethodElectrical2DSolver<Geometry2DType>::getConductivity(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod) {
    this->initCalculation();
    this->writelog(LOG_DEBUG, "Getting conductivities");
    loadConductivities();
    InterpolationFlags flags(this->geometry);
    return LazyData<Tensor2<double>>(new LazyDataDelegateImpl<Tensor2<double>>(dest_mesh->size(),
        [this, dest_mesh, flags](size_t i) -> Tensor2<double> {
            auto point = flags.wrap(dest_mesh->at(i));
            size_t x = this->mesh->axis[0]->findUpIndex(point[0]),
                   y = this->mesh->axis[1]->findUpIndex(point[1]);
            if (x == 0 || y == 0 || x == this->mesh->axis[0]->size() || y == this->mesh->axis[1]->size())
                return Tensor2<double>(NAN);
            else
                return this->conds[this->mesh->element(x-1, y-1).getIndex()];
        }
    ));
}


template <>
double FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::getTotalEnergy() {
    double W = 0.;
    auto T = inTemperature(this->mesh->getMidpointsMesh());
    for (auto e: this->mesh->elements()) {
        size_t ll = e.getLoLoIndex();
        size_t lu = e.getUpLoIndex();
        size_t ul = e.getLoUpIndex();
        size_t uu = e.getUpUpIndex();
        double dvx = 0.5e6 * (- potentials[ll] + potentials[lu] - potentials[ul] + potentials[uu])
                            / (e.getUpper0() - e.getLower0()); // [grad(dV)] = V/m
        double dvy = 0.5e6 * (- potentials[ll] - potentials[lu] + potentials[ul] + potentials[uu])
                            / (e.getUpper1() - e.getLower1()); // [grad(dV)] = V/m
        double w = this->geometry->getMaterial(e.getMidpoint())->eps(T[e.getIndex()]) * (dvx*dvx + dvy*dvy);
        double width = e.getUpper0() - e.getLower0();
        double height = e.getUpper1() - e.getLower1();
        W += width * height * w;
    }
    //TODO add outsides of comptational areas
    return geometry->getExtrusion()->getLength() * 0.5e-18 * phys::epsilon0 * W; // 1e-18 µm³ -> m³
}

template <>
double FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::getTotalEnergy() {
    double W = 0.;
    auto T = inTemperature(this->mesh->getMidpointsMesh());
    for (auto e: this->mesh->elements()) {
        size_t ll = e.getLoLoIndex();
        size_t lu = e.getUpLoIndex();
        size_t ul = e.getLoUpIndex();
        size_t uu = e.getUpUpIndex();
        auto midpoint = e.getMidpoint();
        double dvx = 0.5e6 * (- potentials[ll] + potentials[lu] - potentials[ul] + potentials[uu])
                            / (e.getUpper0() - e.getLower0()); // [grad(dV)] = V/m
        double dvy = 0.5e6 * (- potentials[ll] - potentials[lu] + potentials[ul] + potentials[uu])
                            / (e.getUpper1() - e.getLower1()); // [grad(dV)] = V/m
        double w = this->geometry->getMaterial(midpoint)->eps(T[e.getIndex()]) * (dvx*dvx + dvy*dvy);
        double width = e.getUpper0() - e.getLower0();
        double height = e.getUpper1() - e.getLower1();
        W += width * height * midpoint.rad_r() * w;
    }
    //TODO add outsides of computational area
    return 2.*plask::PI * 0.5e-18 * phys::epsilon0 * W; // 1e-18 µm³ -> m³
}


template<typename Geometry2DType>
double FiniteElementMethodElectrical2DSolver<Geometry2DType>::getCapacitance() {

    if (this->voltage_boundary.size() != 2) {
        throw BadInput(this->getId(), "Cannot estimate applied voltage (exactly 2 voltage boundary conditions required)");
    }

    double U = voltage_boundary[0].value - voltage_boundary[1].value;

    return 2e12 * getTotalEnergy() / (U*U); // 1e12 F -> pF
}


template <>
double FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::getTotalHeat() {
    double W = 0.;
    if (!heats) saveHeatDensities(); // we will compute heats only if they are needed
    for (auto e: this->mesh->elements()) {
        double width = e.getUpper0() - e.getLower0();
        double height = e.getUpper1() - e.getLower1();
        W += width * height * heats[e.getIndex()];
    }
    return geometry->getExtrusion()->getLength() * 1e-15 * W; // 1e-15 µm³ -> m³, W -> mW
}

template <>
double FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::getTotalHeat() {
    double W = 0.;
    if (!heats) saveHeatDensities(); // we will compute heats only if they are needed
    for (auto e: this->mesh->elements()) {
        double width = e.getUpper0() - e.getLower0();
        double height = e.getUpper1() - e.getLower1();
        double r = e.getMidpoint().rad_r();
        W += width * height * r * heats[e.getIndex()];
    }
    return 2e-15*plask::PI * W; // 1e-15 µm³ -> m³, W -> mW
}


template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::getClassName() const { return "electrical.Shockley2D"; }
template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::getClassName() const { return "electrical.ShockleyCyl"; }

template struct PLASK_SOLVER_API FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>;

}}} // namespaces
