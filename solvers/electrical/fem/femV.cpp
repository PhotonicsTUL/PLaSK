#include "femV.h"

namespace plask { namespace solvers { namespace electrical {

template<typename Geometry2DType> FiniteElementMethodElectrical2DSolver<Geometry2DType>::FiniteElementMethodElectrical2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DType, RectilinearMesh2D>(name),
    mJs(1.),
    mBeta(20.),
    mCondPcontact(5.),
    mCondNcontact(50.),
    mVCorrLim(1e-3),
    mLoopNo(0),
    mActLo(0),
    mActHi(0),
    mDact(0.),
    mCorrType(CORRECTION_ABSOLUTE),
    mHeatMethod(HEAT_JOULES),
    outPotential(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getPotentials),
    outCurrentDensity(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getCurrentDensities),
    outHeatDensity(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getHeatDensities),
    mAlgorithm(ALGORITHM_CHOLESKY),
    mIterErr(1e-8),
    mIterLim(10000),
    mLogFreq(500)
{
    mCond.reset();
    mPotentials.reset();
    mCurrentDensities.reset();
    mHeatDensities.reset();
    mCondJunc.reset(1, 5.);

    inTemperature = 300.;
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "voltage" || param == "potential")
            this->readBoundaryConditions(manager, source, mVConst);

        else if (param == "loop") {
            mVCorrLim = source.getAttribute<double>("corrlim", mVCorrLim);
            mCorrType = source.enumAttribute<CorrectionType>("corrtype")
                .value("absolute", CORRECTION_ABSOLUTE, 3)
                .value("relative", CORRECTION_RELATIVE, 3)
                .get(mCorrType);
            source.requireTagEnd();
        }

        else if (param == "matrix") {
            mAlgorithm = source.enumAttribute<Algorithm>("algorithm")
                .value("cholesky", ALGORITHM_CHOLESKY)
                .value("gauss", ALGORITHM_GAUSS)
                .value("iterative", ALGORITHM_ITERATIVE)
                .get(mAlgorithm);
            source.requireTagEnd();
        }

        else if (param == "junction") {
            mJs = source.getAttribute<double>("js", mJs);
            mBeta = source.getAttribute<double>("beta", mBeta);
            auto tCondJunc = source.getAttribute<double>("pnjcond");
            if (tCondJunc) setCondJunc(*tCondJunc);
            auto tWavelength = source.getAttribute<double>("wavelength");
            if (tWavelength) inWavelength = *tWavelength;
            mHeatMethod = source.enumAttribute<HeatMethod>("heat")
                .value("joules", HEAT_JOULES)
                .value("wavelength", HEAT_BANDGAP)
                .get(mHeatMethod);
            source.requireTagEnd();
        }

        else if (param == "contacts") {
            mCondPcontact = source.getAttribute<double>("pcond", mCondPcontact);
            mCondNcontact = source.getAttribute<double>("ncond", mCondNcontact);
            source.requireTagEnd();
        }

        else
            this->parseStandardConfiguration(source, manager);
    }
}


template<typename Geometry2DType> FiniteElementMethodElectrical2DSolver<Geometry2DType>::~FiniteElementMethodElectrical2DSolver() {
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::setActiveRegion()
{
    if (!this->geometry || !this->mesh) return;

    mActLo = mActHi = 0;
    mDact = 0.;

    // Scan for active region
    bool tHadActive = false;
    auto tMidMesh = this->mesh->getMidpointsMesh();
    for (size_t tX = 0; tX != tMidMesh->axis0.size(); ++tX) {
        bool tActive = false;
        for (size_t tY = 0; tY != tMidMesh->axis1.size(); ++tY) {
            if (this->geometry->hasRoleAt("active", tMidMesh->at(tX, tY))) {
                if (!tActive) {
                    if (tHadActive && mActLo != tY)
                        throw BadInput(this->getId(), "Only single flat active region allowed");
                    mActLo = tY;
                    tActive = true;
                }
            } else {
                if (tActive) {
                    if (tHadActive && mActHi != tY)
                        throw BadInput(this->getId(), "Only single flat active region allowed");
                    mActHi = tY;
                    tActive = false;
                    tHadActive = true;
                }
            }
        }
        if (tActive) {
            if (tHadActive && mActHi != tMidMesh->axis1.size())
                throw BadInput(this->getId(), "Only single flat active region allowed");
            mActHi = tMidMesh->axis1.size();
            tHadActive = true;
        }
    }
    if (tHadActive) {
        mDact = this->mesh->axis1[mActHi] - this->mesh->axis1[mActLo];
#ifndef NDEBUG
        this->writelog(LOG_DEBUG, "Active layer thickness = %1%nm", 1e3 * mDact);
#endif
    }

    if (mCondJunc.size() != this->mesh->axis0.size()-1) {
        double tCondY = 0.;
        for (auto cond: mCondJunc) tCondY += cond;
        mCondJunc.reset(this->mesh->axis0.size()-1, tCondY / mCondJunc.size());
    }
}



template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    mLoopNo = 0;
    mAsize = this->mesh->size();
    mPotentials.reset(mAsize, 0.);
    mCond.reset(this->mesh->elements.size());
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::onInvalidate() {
    mCond.reset();
    mPotentials.reset();
    mCurrentDensities.reset();
    mHeatDensities.reset();
}


template<>
inline void FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::setLocalMatrix(double&, double&, double&, double&,
                            double&, double&, double&, double&, double&, double&,
                            double, double, const Vec<2,double>&) {
    return;
}

template<>
inline void FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::setLocalMatrix(double& ioK44, double& ioK33, double& ioK22, double& ioK11,
                               double& ioK43, double& ioK21, double& ioK42, double& ioK31, double& ioK32, double& ioK41,
                               double iKy, double iElemWidth, const Vec<2,double>& iMidPoint) {
        double r = iMidPoint.rad_r();
        double tKr = iKy * iElemWidth / 12.;
        ioK44 = r * ioK44 - tKr;
        ioK33 = r * ioK33 + tKr;
        ioK22 = r * ioK22 + tKr;
        ioK11 = r * ioK11 - tKr;
        ioK43 = r * ioK43;
        ioK21 = r * ioK21;
        ioK42 = r * ioK42;
        ioK31 = r * ioK31;
        ioK32 = r * ioK32 - tKr;
        ioK41 = r * ioK41 + tKr;
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::loadConductivities()
{
    auto iMesh = (this->mesh)->getMidpointsMesh();
    auto tTemperature = inTemperature(iMesh);

    for (auto tE: this->mesh->elements)
    {
        size_t i = tE.getIndex();
        Vec<2,double> tMidPoint = tE.getMidpoint();

        auto tRoles = this->geometry->getRolesAt(tMidPoint);
        if (tRoles.find("active") != tRoles.end()) {
            mCond[i] = Tensor2<double>(0., mCondJunc[tE.getIndex0()]);
        } else if (tRoles.find("p-contact") != tRoles.end()) {
            mCond[i] = Tensor2<double>(mCondPcontact, mCondPcontact);
        } else if (tRoles.find("n-contact") != tRoles.end()) {
            mCond[i] = Tensor2<double>(mCondNcontact, mCondNcontact);
        } else
            mCond[i] = this->geometry->getMaterial(tMidPoint)->cond(tTemperature[i]);
    }
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::saveConductivities()
{
    for (size_t i = 0, j = (mActLo+mActHi)/2; i != this->mesh->axis0.size()-1; ++i)
        mCondJunc[i] = mCond[this->mesh->elements(i,j).getIndex()].c11;
}


template<typename Geometry2DType> double FiniteElementMethodElectrical2DSolver<Geometry2DType>::compute(int loops) {
    switch (mAlgorithm) {
        case ALGORITHM_CHOLESKY: return doCompute<DpbMatrix>(loops);
        case ALGORITHM_GAUSS: return doCompute<DgbMatrix>(loops);
        case ALGORITHM_ITERATIVE: return doCompute<SparseBandMatrix>(loops);
    }
    return 0.;
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(DpbMatrix& iA, DataVector<double>& ioB)
{
    int info = 0;

    this->writelog(LOG_DETAIL, "Solving matrix system");

    // Factorize matrix
    dpbtrf(UPLO, iA.size, iA.kd, iA.data, iA.ld+1, info);
    if (info < 0)
        throw CriticalException("%1%: Argument %2% of dpbtrf has illegal value", this->getId(), -info);
    else if (info > 0)
        throw ComputationError(this->getId(), "Leading minor of order %1% of the stiffness matrix is not positive-definite", info);

    // Find solutions
    dpbtrs(UPLO, iA.size, iA.kd, 1, iA.data, iA.ld+1, ioB.data(), ioB.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", this->getId(), -info);

    // now iA contains factorized matrix and ioB the solutions
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(DgbMatrix& iA, DataVector<double>& ioB)
{
    int info = 0;
    this->writelog(LOG_DETAIL, "Solving matrix system");
    int* ipiv = aligned_malloc<int>(iA.size);

    iA.mirror();

    // Factorize matrix
    dgbtrf(iA.size, iA.size, iA.kd, iA.kd, iA.data, iA.ld+1, ipiv, info);
    if (info < 0) {
        aligned_free(ipiv);
        throw CriticalException("%1%: Argument %2% of dgbtrf has illegal value", this->getId(), -info);
    } else if (info > 0) {
        aligned_free(ipiv);
        throw ComputationError(this->getId(), "Matrix is singlar (at %1%)", info);
    }

    // Find solutions
    dgbtrs('N', iA.size, iA.kd, iA.kd, 1, iA.data, iA.ld+1, ipiv, ioB.data(), ioB.size(), info);
    aligned_free(ipiv);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dgbtrs has illegal value", this->getId(), -info);

    // now iA contains factorized matrix and ioB the solutions
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(SparseBandMatrix& ioA, DataVector<double>& ioB)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    PrecondJacobi precond(ioA);

    DataVector<double> tX = mPotentials.copy(); // We use previous potentials as initial solution
    double tErr;
    try {
        int iter = solveDCG(ioA, precond, tX.data(), ioB.data(), tErr, mIterLim, mIterErr, mLogFreq, this->getId());
        this->writelog(LOG_DETAIL, "Conjugate gradient converged after %1% iterations.", iter);
    } catch (DCGError tExc) {
        throw ComputationError(this->getId(), "Conjugate gradient failed:, %1%", tExc.what());
    }

    ioB = tX;

    // now A contains factorized matrix and B the solutions
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::savePotentials(DataVector<double>& iV)
{
    mMaxAbsVCorr = 0.;
    mMaxRelVCorr = 0.;

    double tMaxV = 0., tMinV = INFINITY;

    for (auto ttPot = mPotentials.begin(), ttV = iV.begin(); ttV != iV.end(); ++ttPot, ++ttV)
    {
        double tAbsCorr = std::abs(*ttV - *ttPot); // for boundary with constant temperature this will be zero anyway
        double tRelCorr = tAbsCorr / *ttV;
        if (tAbsCorr > mMaxAbsVCorr) mMaxAbsVCorr = tAbsCorr;
        if (tRelCorr > mMaxRelVCorr) mMaxRelVCorr = tRelCorr;
        if (*ttV > tMaxV) tMaxV = *ttV;
        if (*ttV < tMinV) tMinV = *ttV;
    }
    mDV = tMaxV - tMinV;
    mMaxRelVCorr *= 100.; // %
    if (mLoopNo == 0) mMaxRelVCorr = 1.;
    std::swap(mPotentials, iV);
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::saveCurrentDensities()
{
    this->writelog(LOG_DETAIL, "Computing current densities");

    mCurrentDensities.reset(this->mesh->elements.size());

    for (auto tE: this->mesh->elements) {
        size_t i = tE.getIndex();
        size_t tLoLeftNo = tE.getLoLoIndex();
        size_t tLoRghtNo = tE.getUpLoIndex();
        size_t tUpLeftNo = tE.getLoUpIndex();
        size_t tUpRghtNo = tE.getUpUpIndex();
        double tDVx = - 0.05 * (- mPotentials[tLoLeftNo] + mPotentials[tLoRghtNo] - mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                             / (tE.getUpper0() - tE.getLower0()); // [j] = kA/cm²
        double tDVy = - 0.05 * (- mPotentials[tLoLeftNo] - mPotentials[tLoRghtNo] + mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                             / (tE.getUpper1() - tE.getLower1()); // [j] = kA/cm²
        mCurrentDensities[i] = vec(mCond[i].c00 * tDVx, mCond[i].c11 * tDVy);
    }
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::saveHeatDensities()
{
    this->writelog(LOG_DETAIL, "Computing heat densities");

    mHeatDensities.reset(this->mesh->elements.size());

    if (mHeatMethod == HEAT_JOULES) {
        for (auto tE: this->mesh->elements) {
            size_t i = tE.getIndex();
            size_t tLoLeftNo = tE.getLoLoIndex();
            size_t tLoRghtNo = tE.getUpLoIndex();
            size_t tUpLeftNo = tE.getLoUpIndex();
            size_t tUpRghtNo = tE.getUpUpIndex();
            auto tMidPoint = tE.getMidpoint();
            if (this->geometry->getMaterial(tMidPoint)->kind() == Material::NONE || this->geometry->hasRoleAt("noheat", tMidPoint))
                mHeatDensities[i] = 0.;
            else {
                double tDVx = 0.5e6 * (- mPotentials[tLoLeftNo] + mPotentials[tLoRghtNo] - mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                                    / (tE.getUpper0() - tE.getLower0()); // [grad(dV)] = V/m
                double tDVy = 0.5e6 * (- mPotentials[tLoLeftNo] - mPotentials[tLoRghtNo] + mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                                    / (tE.getUpper1() - tE.getLower1()); // [grad(dV)] = V/m
                mHeatDensities[i] = mCond[i].c00 * tDVx*tDVx + mCond[i].c11 * tDVy*tDVy;
            }
        }
    } else {
        double tHeatFact = 1e15 * phys::h_J * phys::c / (phys::qe * real(inWavelength()) * mDact);
        for (auto tE: this->mesh->elements) {
            size_t i = tE.getIndex();
            size_t tLoLeftNo = tE.getLoLoIndex();
            size_t tLoRghtNo = tE.getUpLoIndex();
            size_t tUpLeftNo = tE.getLoUpIndex();
            size_t tUpRghtNo = tE.getUpUpIndex();
            double tDVx = 0.5e6 * (- mPotentials[tLoLeftNo] + mPotentials[tLoRghtNo] - mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                                / (tE.getUpper0() - tE.getLower0()); // [grad(dV)] = V/m
            double tDVy = 0.5e6 * (- mPotentials[tLoLeftNo] - mPotentials[tLoRghtNo] + mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                                / (tE.getUpper1() - tE.getLower1()); // [grad(dV)] = V/m
            auto tMidPoint = tE.getMidpoint();
            auto tRoles = this->geometry->getRolesAt(tMidPoint);
            if (tRoles.find("active") != tRoles.end()) {
                double tJy = mCond[i].c11 * fabs(tDVy); // [j] = A/m²
                mHeatDensities[i] = tHeatFact * tJy ;
            } else if (this->geometry->getMaterial(tMidPoint)->kind() == Material::NONE || tRoles.find("noheat") != tRoles.end())
                mHeatDensities[i] = 0.;
            else
                mHeatDensities[i] = mCond[i].c00 * tDVx*tDVx + mCond[i].c11 * tDVy*tDVy;
        }
    }
}


template<> double FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::integrateCurrent(size_t iVertIndex)
{
    if (!mPotentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    if (!mCurrentDensities) saveCurrentDensities();
    double result = 0.;
    for (size_t i = 0; i < mesh->axis0.size()-1; ++i) {
        auto element = mesh->elements(i, iVertIndex);
        result += mCurrentDensities[element.getIndex()].c1 * element.getSize0();
    }
    result *= geometry->getExtrusion()->getLength() * 0.01; // kA/cm² µm² -->  mA
    if (this->getGeometry()->isSymmetric(Geometry::DIRECTION_TRAN)) return 2.*result;
    return result;
}

template<> double FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::integrateCurrent(size_t iVertIndex)
{
    if (!mPotentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    if (!mCurrentDensities) saveCurrentDensities();
    double result = 0.;
    for (size_t i = 0; i < mesh->axis0.size()-1; ++i) {
        auto element = mesh->elements(i, iVertIndex);
        double rin = element.getLower0(), rout = element.getUpper0();
        result += mCurrentDensities[element.getIndex()].c1 * (rout*rout - rin*rin);
    }
    return result * M_PI * 0.01; // kA/cm² µm² -->  mA
}

template<typename Geometry2DType> double FiniteElementMethodElectrical2DSolver<Geometry2DType>::getTotalCurrent()
{
    // Find the average of the active region
    size_t level = (mActLo + mActHi) / 2;
    return integrateCurrent(level);
}


template<typename Geometry2DType> DataVector<const double> FiniteElementMethodElectrical2DSolver<Geometry2DType>::getPotentials(const MeshD<2>& dst_mesh, InterpolationMethod method) const
{
    if (!mPotentials) throw NoValue("Potentials");
    this->writelog(LOG_DETAIL, "Getting potentials");
    if (method == DEFAULT_INTERPOLATION)  method = INTERPOLATION_LINEAR;
    return interpolate(*(this->mesh), mPotentials, WrappedMesh<2>(dst_mesh, this->geometry), method);
}

template<typename Geometry2DType> DataVector<const Vec<2> > FiniteElementMethodElectrical2DSolver<Geometry2DType>::getCurrentDensities(const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    if (!mPotentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Getting current densities");
    if (!mCurrentDensities) saveCurrentDensities();
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    auto dest_mesh = WrappedMesh<2>(dst_mesh, this->geometry);
    auto result = interpolate(*(this->mesh->getMidpointsMesh()), mCurrentDensities, dest_mesh, method);
    constexpr Vec<2> zero(0.,0.);
    for (size_t i = 0; i < result.size(); ++i)
        if (!this->geometry->getChildBoundingBox().includes(dest_mesh[i])) result[i] = zero;
    return result;
}

template<typename Geometry2DType> DataVector<const double> FiniteElementMethodElectrical2DSolver<Geometry2DType>::getHeatDensities(const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    if (!mPotentials) throw NoValue("Heat densities");
    this->writelog(LOG_DETAIL, "Getting heat densities");
    if (!mHeatDensities) saveHeatDensities(); // we will compute fluxes only if they are needed
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    auto dest_mesh = WrappedMesh<2>(dst_mesh, this->geometry);
    auto result = interpolate(*(this->mesh->getMidpointsMesh()), mHeatDensities, dest_mesh, method);
    for (size_t i = 0; i < result.size(); ++i)
        if (!this->geometry->getChildBoundingBox().includes(dest_mesh[i])) result[i] = 0.;
    return result;
}



template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::getClassName() const { return "electrical.Beta2D"; }
template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::getClassName() const { return "electrical.BetaCyl"; }

template struct FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>;
template struct FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>;

}}} // namespace plask::solvers::thermal
