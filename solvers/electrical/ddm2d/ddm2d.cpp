#include <boost/version.hpp>

#if (BOOST_VERSION >= 105000)
    #include <boost/algorithm/clamp.hpp>
    using boost::algorithm::clamp;
#else
    template <typename T>
    const T& clamp(const T& v, const T& min, const T& max) {
        if (v < min) return min;
        if (v > max) return max;
        return v;
    }
#endif

#include "ddm2d.h"

namespace plask { namespace electrical { namespace drift_diffusion {

/** Compute effective density of states
 * \param M carrier effective mass
 * \param T temperature
 */
static inline double Neff(Tensor2<double> M, double T) {
    constexpr double fact = phys::me * phys::kB_eV / (2.*plask::PI * phys::hb_eV * phys::hb_J);
    double m = pow(M.c00 * M.c00 * M.c11, 0.3333333333333333);
    return 2e-6 * pow(fact * m * T, 1.5);
}

/** Compute intrinsic carrier concentration
 * \param M carrier effective mass
 * \param T temperature
 */
static inline double Ni(double Nc, double Nv, double Eg, double T) {
    return sqrt(Nc*Nv) * exp(-Eg/(2*phys::kB_eV*T));
}

template <typename Geometry2DType>
DriftDiffusionModel2DSolver<Geometry2DType>::DriftDiffusionModel2DSolver(const std::string& name) : SolverWithMesh <Geometry2DType, RectangularMesh<2>>(name),
    mTx(300.),
    mEx(phys::kB_eV*mTx),
    mNx(1e18),
    mEpsRx(12.9),
    mXx(sqrt((phys::epsilon0*phys::kB_J*mTx*mEpsRx)/(phys::qe*phys::qe*mNx))*1e3),
    //mKx(100.),
    mMix(1000.),
    mRx(((phys::kB_J*mTx*mMix*mNx)/(phys::qe*mXx*mXx))*1e8),
    mJx(((phys::kB_J*mNx)*mTx*mMix/mXx)*10.),
    //mtx(mNx/mRx),
    mAx(mRx/mNx),
    mBx(mRx/(mNx*mNx)),
    mCx(mRx/(mNx*mNx*mNx)),
    //mHx(((mKx*mTx)/(mXx*mXx))*1e12),
    mPx((mXx*phys::qe*mNx)*1e-4), // polarization (C/m^2)
    dU(0.002),
    maxDelPsi0(2.),
    maxDelPsi(0.1*dU),
    maxDelFn(1e20),
    maxDelFp(1e20),
    stat(STAT_MB),
    conttype(OHMIC),
    needPsi0(true),
    //loopno(0),
    //maxerr(0.05),
    outPotential(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getPotentials),
    outFermiLevels(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getFermiLevels),
    outBandEdges(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getBandEdges),
    outCurrentDensityForElectrons(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getCurrentDensitiesForElectrons),
    outCurrentDensityForHoles(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getCurrentDensitiesForHoles),
    outCarriersConcentration(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getCarriersConcentration),
    outHeat(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getHeatDensities),
    //outConductivity(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getConductivity),
    algorithm(ALGORITHM_CHOLESKY),
    mRsrh(false),
    mRrad(false),
    mRaug(false),
    mPol(false),
    mFullIon(true),
    mSchottkyP(0.),
    mSchottkyN(0.),
    maxerrPsiI(1e-6),
    maxerrPsi0(1e-6),
    maxerrPsi(1e-6),
    maxerrFn(1e-4),
    maxerrFp(1e-4),
    loopsPsiI(10000),
    loopsPsi0(200),
    loopsPsi(3),
    loopsFn(3),
    loopsFp(3),
    itererr(1e-8),
    iterlim(10000),
    logfreq(500),
	T(300.), // TODO T=300 for kp method tests
	T0(300.)
{
    onInvalidate();
    inTemperature = 300.;
    inTemperature.changedConnectMethod(this, &DriftDiffusionModel2DSolver<Geometry2DType>::onInputChange);
}

template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "voltage")
            this->readBoundaryConditions(manager, source, voltage_boundary);
        else if (param == "loop") {
            stat = source.enumAttribute<Stat>("stat")
                .value("MB", STAT_MB)
                .value("FD", STAT_FD)
                .value("Maxwell-Boltzmann", STAT_MB)
                .value("Fermi-Dirac", STAT_FD)
                .get(stat);
            conttype = source.enumAttribute<ContType>("conttype")
                .value("ohmic", OHMIC)
                .value("Schottky", SCHOTTKY)
                .get(conttype);
            mSchottkyP = source.getAttribute<double>("SchottkyP", mSchottkyP);
            mSchottkyN = source.getAttribute<double>("SchottkyN", mSchottkyN);
            mRsrh = source.getAttribute<bool>("Rsrh", mRsrh);
            mRrad = source.getAttribute<bool>("Rrad", mRrad);
            mPol = source.getAttribute<bool>("Pol", mPol);
            mRaug = source.getAttribute<bool>("Raug", mRaug);
            mFullIon = source.getAttribute<bool>("FullIon", mFullIon);
            maxerrPsiI = source.getAttribute<double>("maxerrVi", maxerrPsiI);
            maxerrPsi0 = source.getAttribute<double>("maxerrV0", maxerrPsi0);
            maxerrPsi = source.getAttribute<double>("maxerrV", maxerrPsi);
            maxerrFn = source.getAttribute<double>("maxerrFn", maxerrFn);
            maxerrFp = source.getAttribute<double>("maxerrFp", maxerrFp);
            loopsPsiI = source.getAttribute<size_t>("loopsVi", loopsPsiI);
            loopsPsi0 = source.getAttribute<size_t>("loopsV0", loopsPsi0);
            loopsPsi = source.getAttribute<size_t>("loopsV", loopsPsi);
            loopsFn = source.getAttribute<size_t>("loopsFn", loopsFn);
            loopsFp = source.getAttribute<size_t>("loopsFp", loopsFp);
            source.requireTagEnd();
        } else if (param == "matrix") {
            algorithm = source.enumAttribute<Algorithm>("algorithm")
                .value("cholesky", ALGORITHM_CHOLESKY)
                .value("gauss", ALGORITHM_GAUSS)
                .value("iterative", ALGORITHM_ITERATIVE)
                .get(algorithm);
            itererr = source.getAttribute<double>("itererr", itererr);
            iterlim = source.getAttribute<size_t>("iterlim", iterlim);
            logfreq = source.getAttribute<size_t>("logfreq", logfreq);
            source.requireTagEnd();
        } else if (param == "config") {
			T0 = source.getAttribute<double>("T0", T0);
			strained = source.getAttribute<bool>("strained", strained);
			//             quick_levels = reader.getAttribute<bool>("quick-levels", quick_levels);
			source.requireTagEnd();
		}
		else
            this->parseStandardConfiguration(source, manager);
    }
}


template <typename Geometry2DType>
DriftDiffusionModel2DSolver<Geometry2DType>::~DriftDiffusionModel2DSolver() {
}


template<typename Geometry2DType>
size_t DriftDiffusionModel2DSolver<Geometry2DType>::getActiveRegionMeshIndex(size_t actnum) const
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());

    size_t actlo, acthi, lon = 0, hin = 0;

    shared_ptr<RectangularMesh<2>> points = this->mesh->getMidpointsMesh();
    size_t ileft = 0, iright = points->axis[0]->size();
    bool in_active = false;
    for (size_t r = 0; r < points->axis[1]->size(); ++r) {
        bool had_active = false;
        for (size_t c = 0; c < points->axis[0]->size(); ++c) { // In the (possible) active region
            auto point = points->at(c,r);
            bool active = isActive(point);
            if (c >= ileft && c <= iright) {
                // Here we are inside potential active region
                if (active) {
                    if (!had_active) {
                        if (!in_active) { // active region is starting set-up new region info
                            ileft = c;
                            actlo = r;
                            lon++;
                        }
                    }
                } else if (had_active) {
                    if (!in_active) iright = c;
                    else throw Exception("{}: Right edge of the active region not aligned.", this->getId());
                }
                had_active |= active;
            }
        }
        in_active = had_active;
        // Test if the active region has finished
        if (!in_active && lon != hin) {
            acthi = r;
            if(hin++ == actnum) return (actlo + acthi) / 2;
        }
    }
    // Test if the active region has finished
    if (lon != hin) {
        acthi = points->axis[1]->size();
        if(hin++ == actnum) return (actlo + acthi) / 2;
    }
    throw BadInput(this->getId(), "Wrong active region number {}", actnum);
}



template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
	detectActiveRegions();

    size = this->mesh->size();

    dvnPsi0.reset(size);
    dvnFnEta.reset(size, 1.);
    dvnFpKsi.reset(size, 1.);

    dvePsi.reset(this->mesh->getElementsCount());
    dveFnEta.reset(this->mesh->getElementsCount(), 1.);
    dveFpKsi.reset(this->mesh->getElementsCount(), 1.);
    dveN.reset(this->mesh->getElementsCount());
    dveP.reset(this->mesh->getElementsCount());

    currentsN.reset(this->mesh->getElementsCount());
    currentsP.reset(this->mesh->getElementsCount());

    needPsi0 = true;
}


template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::onInvalidate() {
    dvnPsi0.reset();
    dvnPsi.reset();
    dvnFnEta.reset();
    dvnFpKsi.reset();
    dvePsi.reset();
    dveFnEta.reset();
    dveFpKsi.reset();
    dveN.reset();
    dveP.reset();
    currentsN.reset();
    currentsP.reset();
    heats.reset();
    regions.clear();
    materialSubstrate.reset();
}


template <typename Geometry2DType>
template <typename MatrixT> // add deltaPsi = 0 on p- and n-contacts
void DriftDiffusionModel2DSolver<Geometry2DType>::applyBC(MatrixT& A, DataVector<double>& B,
                                                          const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double> & bvoltage) {
    // boundary conditions of the first kind
    for (auto cond: bvoltage) {
        for (auto r: cond.place) {
            A(r,r) = 1.;
            B[r] = 0.;
            size_t start = (r > A.kd)? r-A.kd : 0;
            size_t end = (r + A.kd < A.size)? r+A.kd+1 : A.size;
            for(size_t c = start; c < r; ++c) A(r,c) = 0.;
            for(size_t c = r+1; c < end; ++c) A(r,c) = 0.;
        }
    }
}

template <typename Geometry2DType> // add deltaPsi = 0 on p- and n-contacts
void DriftDiffusionModel2DSolver<Geometry2DType>::applyBC(SparseBandMatrix& A, DataVector<double>& B,
                                                          const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double> &bvoltage) {
    // boundary conditions of the first kind
    for (auto cond: bvoltage) {
        for (auto r: cond.place) {
            double* rdata = A.data + LDA*r;
            *rdata = 1.;
            B[r] = 0.;
            // below diagonal
            for (ptrdiff_t i = 4; i > 0; --i) {
                ptrdiff_t c = r - A.bno[i];
                if (c >= 0) A.data[LDA*c+i] = 0.;
            }
            // above diagonal
            for (ptrdiff_t i = 1; i < 5; ++i) {
                ptrdiff_t c = r + A.bno[i];
                if (c < A.size) rdata[i] = 0.;
            }
        }
    }
}

template <>
inline void DriftDiffusionModel2DSolver<Geometry2DCartesian>::addCurvature(double&, double&, double&, double&,
                             double&, double&, double&, double&, double&, double&,
                             double, double, const Vec<2,double>&)
{
}

template <> // TODO czy to bedzie OK?
inline void DriftDiffusionModel2DSolver<Geometry2DCylindrical>::addCurvature(double& k44, double& k33, double& k22, double& k11,
                                double& k43, double& k21, double& k42, double& k31, double& k32, double& k41,
                                double, double, const Vec<2,double>& midpoint)
{
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

template <typename Geometry2DType>
template <CalcType calctype, typename MatrixT>
void DriftDiffusionModel2DSolver<Geometry2DType>::setMatrix(MatrixT& A, DataVector<double>& B,
                                                            const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double> &bvoltage)
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size={0}, bands={1}({2}))", A.size, A.kd+1, A.ld+1);

    //auto iMesh = (this->mesh)->getMidpointsMesh();
    //auto temperatures = inTemperature(iMesh);
    auto iMeshN = this->mesh;
    auto temperaturesN = inTemperature(iMeshN);

//TODO    2e-6*pow((Me(T,e,point).c00*plask::phys::me*plask::phys::kB_eV*300.)/(2.*PI*plask::phys::hb_eV*plask::phys::hb_J),1.5);

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
        double hx = (e.getUpper0() - e.getLower0()) / mXx; // normalised element width
        double hy = (e.getUpper1() - e.getLower1()) / mXx; // normalised element height

        Vec <2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        double T;//(300.); //TODO
        // average temperature on the element
        T = 0.25 * (temperaturesN[loleftno] + temperaturesN[lorghtno] + temperaturesN[upleftno] + temperaturesN[uprghtno]); // in (K)
        double normT(T/mTx); // normalised temperature

        double n, p;
        if (calctype == CALC_PSI0) {
            if (material->kind() == Material::OXIDE || material->kind() == Material::DIELECTRIC || material->kind() == Material::EMPTY ) { // 26.01.2016
                n = 0.;
                p = 0.;
            }
            else {
                double normNc = Neff(material->Me(T, 0., '*'), T) / mNx;
                double normEc0 = material->CB(T, 0., '*') / mEx;
                double normNv = Neff(material->Mh(T, 0.), T) / mNx;
                double normEv0 = material->VB(T, 0., '*') / mEx;
                double normT = T / mTx;
                double ePsi = 0.25 * (dvnPsi0[loleftno] + dvnPsi0[lorghtno] + dvnPsi0[upleftno] + dvnPsi0[uprghtno]);
                n = calcN(normNc, 1., ePsi, normEc0, normT);
                p = calcP(normNv, 1., ePsi, normEv0, normT);
            }
        }
        else {
            if (material->kind() == Material::OXIDE || material->kind() == Material::DIELECTRIC || material->kind() == Material::EMPTY ) { // 26.01.2016
                n = 0.;
                p = 0.;
            }
            else { // earlier only this
                n = dveN[i];
                p = dveP[i];
            }
        }

        double kk, kx, ky, gg, ff;

        if (calctype == CALC_FN) {
            double normEc0(0.), normNc(0.), normNv(0.), normNe(0.), normNi(0.), normMobN(0.), yn(0.);

            if (material->kind() == Material::OXIDE || material->kind() == Material::DIELECTRIC || material->kind() == Material::EMPTY ) { // 26.01.2016
                yn = 1.; // ?
                normMobN = 1e-3; // ?
                normNe = 1e-20; // ?
            }
            else {
                normEc0 = material->CB(T, 0., '*') / mEx;
                normNc = Neff(material->Me(T, 0., '*'), T) / mNx;
                normNv = Neff(material->Mh(T, 0.), T) / mNx;
                normNe = normNc * exp(dvePsi[i]-normEc0);
                normNi = Ni(normNc,normNv,material->Eg(T, 0., '*'),T) / mNx;
                normMobN = 0.5*(material->mobe(T).c00+material->mobe(T).c11) / mMix; // TODO

                switch (stat) {
                    case STAT_MB: yn = 1.; break;
                    //case STAT_FD: yn = fermiDiracHalf(log(dveFnEta[i])+dvePsi[i]-normEc0)/(dveFnEta[i]*exp(dvePsi[i]-normEc0)); break;
                    case STAT_FD: yn = fermiDiracHalf((log(dveFnEta[i])+dvePsi[i]-normEc0)/normT) / (pow(dveFnEta[i],1./normT)*exp((dvePsi[i]-normEc0)/normT)); break;
                }
            }

            kk = 1. / (3.*(hx*0.5)*(hy*0.5));
            kx = normMobN * normNe * yn * (hy*0.5) * (hy*0.5);
            ky = normMobN * normNe * yn * (hx*0.5) * (hx*0.5);
            ff = gg = 0.;

            if (material->kind() != Material::OXIDE && material->kind() != Material::DIELECTRIC && material->kind() != Material::EMPTY ) /*if (ttE->getL()->getID() == "QW")*/ { // TODO (only in active?)
                if (mRsrh) {
                    //this->writelog(LOG_DATA, "Recombination SRH");
                    double normte = material->taue(T) * mAx * 1e-9;  // 1e-9: ns -> s
                    double normth = material->tauh(T) * mAx * 1e-9;
                    gg += ((1./9.) * (hx*0.5) * (hy*0.5) * normNe * yn * (p + normNi) * (normNi * normth  + p * normte)
                        / pow((n + normNi) * normth + (p + normNi) * normte, 2.));
                    ff += ((hx*0.5) * (hy*0.5) * (n * p - normNi * normNi) / ((n + normNi) * normth + (p + normNi) * normte));
                }
                if (mRrad) {
                    //this->writelog(LOG_DATA, "Recombination RAD");
                    double normB = material->B(T) / mBx;
                    gg += ((1./9.) * (hx*0.5) * (hy*0.5) * normB * normNe * yn * p);
                    ff += ((hx*0.5) * (hy*0.5) * normB * (n * p - normNi * normNi));
                }
                if (mRaug) {
                    //this->writelog(LOG_DATA, "Recombination AUG");
                    double normCe = material->Ce(T) / mCx;
                    double normCh = material->Ch(T) / mCx;
                    gg += ((1./9.) * (hx*0.5) * (hy*0.5) * normNe * yn * ((normCe * (2. * n * p - normNi * normNi) + normCh * p * p)));
                    ff += ((hx*0.5) * (hy*0.5) * (normCe * n + normCh * p) * (n * p - normNi * normNi));
                }
            }
        }
        else if (calctype == CALC_FP)  {
            double normEv0(0.), normNc(0.), normNv(0.), normNh(0.), normNi(0.), normMobP(0.), yp(0.);

            if (material->kind() == Material::OXIDE || material->kind() == Material::DIELECTRIC || material->kind() == Material::EMPTY ) { // 26.01.2016
                yp = 1.; // ?
                normMobP = 1e-3; // ?
                normNh = 1e-20; // ?
            }
            else {
                normEv0 = material->VB(T, 0., '*') / mEx;
                normNc = Neff(material->Me(T, 0., '*'), T) / mNx;
                normNv = Neff(material->Mh(T, 0.), T) / mNx;
                normNh = normNv * exp(-dvePsi[i]+normEv0);
                normNi = Ni(normNc,normNv,material->Eg(T, 0., '*'),T) / mNx;
                normMobP = 0.5*(material->mobh(T).c00+material->mobh(T).c11) / mMix; // TODO

                switch (stat) {
                    case STAT_MB: yp = 1.; break;
                    //case STAT_FD: yp = fermiDiracHalf(log(dveFpKsi[i])-dvePsi[i]+normEv0)/(dveFpKsi[i]*exp(-dvePsi[i]+normEv0)); break;
                    case STAT_FD: yp = fermiDiracHalf((log(dveFpKsi[i])-dvePsi[i]+normEv0)/normT) / (pow(dveFpKsi[i],1./normT)*exp((-dvePsi[i]+normEv0)/normT)); break;
                }
            }

            kk = 1. / (3.*(hx*0.5)*(hy*0.5));
            kx = normMobP * normNh * yp * (hy*0.5) * (hy*0.5);
            ky = normMobP * normNh * yp * (hx*0.5) * (hx*0.5);
            ff = gg = 0.;

            if (material->kind() != Material::OXIDE && material->kind() != Material::DIELECTRIC && material->kind() != Material::EMPTY ) /*if (ttE->getL()->getID() == "QW")*/ { // TODO (only in active?)
                if (mRsrh) {
                    //this->writelog(LOG_DATA, "Recombination SRH");
                    double normte = material->taue(T) * mAx * 1e-9;
                    double normth = material->tauh(T) * mAx * 1e-9;
                    gg += ((1./9.) * (hx*0.5) * (hy*0.5) * normNh * yp * (n + normNi) * (normNi * normte + n * normth)
                        / pow((n + normNi) * normth + (p + normNi) * normte, 2.));
                    ff += ((hx*0.5) * (hy*0.5) * (n * p - normNi * normNi) / ((n + normNi) * normth + (p + normNi) * normte));
                }
                if (mRrad) {
                    //this->writelog(LOG_DATA, "Recombination RAD");
                    double normB = material->B(T) / mBx;
                    gg += ((1./9.) * (hx*0.5) * (hy*0.5) * normB * normNh * yp * n);
                    ff += ((hx*0.5) * (hy*0.5) * normB * (n * p - normNi * normNi));
                }
                if (mRaug) {
                    //this->writelog(LOG_DATA, "Recombination AUG");
                    double normCe = material->Ce(T) / mCx;
                    double normCh = material->Ch(T) / mCx;
                    gg += ((1./9.) * (hx*0.5) * (hy*0.5) * normNh * yp * ((normCh * (2. * n * p - normNi * normNi) + normCe * n * n)));
                    ff += ((hx*0.5) * (hy*0.5) * (normCe * n + normCh * p) * (n * p - normNi * normNi));
                }
            }
        }
        else { // CALC_PSI
            double normEps = material->eps(T) / mEpsRx;

            kk = 1. / (3.*(hx*0.5)*(hy*0.5));
            kx = normT * normEps * (hy*0.5) * (hy*0.5);
            ky = normT * normEps * (hx*0.5) * (hx*0.5);

            if (material->kind() == Material::OXIDE || material->kind() == Material::DIELECTRIC || material->kind() == Material::EMPTY ) /*if (ttE->getL()->getID() == "QW")*/ { // TODO (only in active?)
                gg = 0.;
                ff = 0.;
            }
            else {
                gg = (1./9.) * (p + n) * (hx*0.5) * (hy*0.5);

                double normNc = Neff(material->Me(T, 0., '*'), T) / mNx;
                double normNv = Neff(material->Mh(T, 0.), T) / mNx;
                //double Ni = material->Ni(T) / mNx;
                double normNd = material->Nd() / mNx;
                double normNa = material->Na() / mNx;
                double normNdIon = normNd;
                double normNaIon = normNa;
                if (!mFullIon) {
                    //this->writelog(LOG_RESULT, "Full ionization false");
                    double gD(2.), gA(4.);
                    double normEd = material->EactD(T) / mEx;
                    double normEa = material->EactA(T) / mEx;
                    double normNdTmp = (normNc/gD)*exp(-normEd);
                    double normNaTmp = (normNv/gA)*exp(-normEa);
                    normNdIon = normNd * (normNdTmp/(normNdTmp+n));
                    normNaIon = normNa * (normNaTmp/(normNaTmp+p));
                }
                ff = - (hx*0.5) * (hy*0.5) * (p - n + normNdIon - normNaIon);
                if (mPol) {
                    double eII = (3.188 - material->lattC(T,'a')) / material->lattC(T,'a'); // TODO wstawic stala podloza
                    double eL = -2. * eII * material->c13(T) / material->c33(T); // TODO uzaleznic od kata teta
                    double Ppz = material->e33(T) * eL + 2. * material->e13(T) * eII; // TODO sprawdzic czy OK
                    double Ptot = material->Psp(T) + Ppz;
                    double normPtot = Ptot / mPx;
                    ff += normPtot;
                }
            }
        }

        // set symmetric matrix components
        double k44, k33, k22, k11, k43, k21, k42, k31, k32, k41;
        double g44, g33, g22, g11, g43, g21, g42, g31, g32, g41;
        double v1, v2, v3, v4;

        // local K
        k44 = k33 = k22 = k11 = (kx+ky)*kk;
        k43 = k21 = 0.5*(-2.*kx+ky)*kk;
        k42 = k31 = 0.5*(-kx-ky)*kk;
        k32 = k41 = 0.5*(kx-2.*ky)*kk;

        // local G
        g44 = g33 = g22 = g11 = 4.*gg;
        g21 = g41 = g32 = g43 = 2.*gg;
        g31 = g42 = gg;

        // set stiffness matrix
        addCurvature(k44, k33, k22, k11, k43, k21, k42, k31, k32, k41, ky, hx, midpoint); // TODO uncomment and correct after takng cylindrical structures into account

        A(loleftno, loleftno) += k11 + g11;
        A(lorghtno, lorghtno) += k22 + g22;
        A(uprghtno, uprghtno) += k33 + g33;
        A(upleftno, upleftno) += k44 + g44;

        A(lorghtno, loleftno) += k21 + g21;
        A(uprghtno, loleftno) += k31 + g31;
        A(upleftno, loleftno) += k41 + g41;
        A(uprghtno, lorghtno) += k32 + g32;
        A(upleftno, lorghtno) += k42 + g42;
        A(upleftno, uprghtno) += k43 + g43;

        switch (calctype) {
            case CALC_PSI0:
                v1 = dvnPsi0[loleftno];
                v2 = dvnPsi0[lorghtno];
                v3 = dvnPsi0[uprghtno];
                v4 = dvnPsi0[upleftno];
                break;
            case CALC_PSI:
                v1 = dvnPsi[loleftno];
                v2 = dvnPsi[lorghtno];
                v3 = dvnPsi[uprghtno];
                v4 = dvnPsi[upleftno];
                break;
            case CALC_FN:
                v1 = dvnFnEta[loleftno];
                v2 = dvnFnEta[lorghtno];
                v3 = dvnFnEta[uprghtno];
                v4 = dvnFnEta[upleftno];
                break;
            case CALC_FP:
                v1 = dvnFpKsi[loleftno];
                v2 = dvnFpKsi[lorghtno];
                v3 = dvnFpKsi[uprghtno];
                v4 = dvnFpKsi[upleftno];
        }

        B[loleftno] -= k11*v1 + k21*v2 + k31*v3 + k41*v4 + ff;
        B[lorghtno] -= k21*v1 + k22*v2 + k32*v3 + k42*v4 + ff;
        B[uprghtno] -= k31*v1 + k32*v2 + k33*v3 + k43*v4 + ff;
        B[upleftno] -= k41*v1 + k42*v2 + k43*v3 + k44*v4 + ff;
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


template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::savePsi()
{
    for (auto el: this->mesh->elements()) {
        size_t i = el.getIndex();
        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex();

        dvePsi[i] = 0.25 * (dvnPsi[loleftno] + dvnPsi[lorghtno] + dvnPsi[upleftno] + dvnPsi[uprghtno]);
    }
}


template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveFnEta()
{
    for (auto el: this->mesh->elements()) {
        size_t i = el.getIndex();
        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex();

        dveFnEta[i] = 0.25 * (dvnFnEta[loleftno] + dvnFnEta[lorghtno] + dvnFnEta[upleftno] + dvnFnEta[uprghtno]);
    }
}


template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveFpKsi()
{
    for (auto el: this->mesh->elements()) {
        size_t i = el.getIndex();
        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex();

        dveFpKsi[i] = 0.25 * (dvnFpKsi[loleftno] + dvnFpKsi[lorghtno] + dvnFpKsi[upleftno] + dvnFpKsi[uprghtno]);
    }
}


template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveN()
{
    //this->writelog(LOG_DETAIL, "Saving electron concentration");

    //auto iMesh = (this->mesh)->getMidpointsMesh();
    //auto temperatures = inTemperature(iMesh);
    auto iMeshE = (this->mesh)->getMidpointsMesh();
    auto temperaturesE = inTemperature(iMeshE);

    for (auto e: this->mesh->elements())
    {
        size_t i = e.getIndex();
        Vec < 2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        if (material->kind() == Material::OXIDE || material->kind() == Material::DIELECTRIC || material->kind() == Material::EMPTY ) { // 26.01.2016
            dveN[i] = 0.;
            continue;
        }
        //double T(300.); // TODO
        double normNc = Neff(material->Me(temperaturesE[i], 0., '*'), temperaturesE[i]) / mNx;
        double normEc0 = material->CB(temperaturesE[i], 0., '*') / mEx;
        double normT = temperaturesE[i] / mTx;

        dveN[i] = calcN(normNc, dveFnEta[i], dvePsi[i], normEc0, normT);
    }
}


template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveP()
{
    //this->writelog(LOG_DETAIL, "Saving hole concentration");

    //auto iMesh = (this->mesh)->getMidpointsMesh();
    //auto temperatures = inTemperature(iMesh);
    auto iMeshE = (this->mesh)->getMidpointsMesh();
    auto temperaturesE = inTemperature(iMeshE);

    for (auto e: this->mesh->elements())
    {
        size_t i = e.getIndex();
        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        if (material->kind() == Material::OXIDE || material->kind() == Material::DIELECTRIC || material->kind() == Material::EMPTY ) { // 26.01.2016
            dveP[i] = 0.;
            continue;
        }

        //double T(300.); // TODO
        double normNv = Neff(material->Mh(temperaturesE[i], 0.), temperaturesE[i]) / mNx;
        double normEv0 = material->VB(temperaturesE[i], 0., '*') / mEx;
        double normT = temperaturesE[i] / mTx;

        dveP[i] = calcP(normNv, dveFpKsi[i], dvePsi[i], normEv0, normT);
    }
}


template <typename Geometry2DType>
template <CalcType calctype>
double DriftDiffusionModel2DSolver<Geometry2DType>::addCorr(DataVector<double>& corr, const BoundaryConditionsWithMesh <RectangularMesh<2>::Boundary,double>& vconst)
{  
    //this->writelog(LOG_DEBUG, "Adding corrections");

    double err;

    //double tMaxRelUpd = 0.; // update/old_value = this will be the result

    for (auto cond: vconst)
        for (auto i: cond.place)
            corr[i] = 0.;

    if (calctype == CALC_PSI0) {
        err = 0.;
        double normDel = maxDelPsi0 / mEx;
        for (std::size_t i = 0; i < this->mesh->size(); ++i) {
            corr[i] = clamp(corr[i], -normDel, normDel);
            err = std::max(err, std::abs(corr[i]));
            dvnPsi0[i] += corr[i];
        }
        this->writelog(LOG_DETAIL, "Maximum update for the built-in potential: {:g} V", err*mEx);
    }
    else if (calctype == CALC_PSI) {
        err = 0.;
        double normDel = maxDelPsi / mEx;
        for (std::size_t i = 0; i < this->mesh->size(); ++i) {
            corr[i] = clamp(corr[i], -normDel, normDel);
            err = std::max(err, std::abs(corr[i]));
            dvnPsi[i] += corr[i];
        }
        this->writelog(LOG_DETAIL, "Maximum update for the potential: {:g} V", err*mEx);
    }
    else if (calctype == CALC_FN) {
        err = 0.;
        //double normDel = maxDelFn / mEx;
        for (std::size_t i = 0; i < this->mesh->size(); ++i) {
            dvnFnEta[i] += corr[i];
            err = std::max(err, std::abs(corr[i]/dvnFnEta[i]));
        }
        this->writelog(LOG_DETAIL, "Maximum relative update for the quasi-Fermi energy level for electrons: {0}.", err);
    }
    else if (calctype == CALC_FP) {
        err = 0.;
        //double normDel = maxDelFp / mEx;
        for (std::size_t i = 0; i < this->mesh->size(); ++i) {
            dvnFpKsi[i] += corr[i];
            err = std::max(err, std::abs(corr[i]/dvnFpKsi[i]));
        }
        this->writelog(LOG_DETAIL, "Maximum relative update for the quasi-Fermi energy level for holes: {0}.", err);
    }
    return err; // for Psi -> normalised (max. delPsi)

    /*double maxRelUpd(0.);
    double mcNorm;
    if (calctype == CALC_PSI0) {
        mcNorm = maxDelPsi0/mEx;
        for (int i = 0; i < this->mesh->size(); ++i) {
            if (dvnDeltaPsi[i] > mcNorm) dvnDeltaPsi[i] = mcNorm;
            else if (dvnDeltaPsi[i] < -mcNorm) dvnDeltaPsi[i] = -mcNorm;
            if (std::abs(dvnDeltaPsi[i]/dvnPsi[i]) > maxRelUpd) maxRelUpd = std::abs(dvnDeltaPsi[i]/dvnPsi[i]);
            dvnPsi[i] = dvnPsi[i] + dvnDeltaPsi[i];
        }
    }
    else if (calctype == CALC_PSI) {
        mcNorm = maxDelPsi/mEx;
        for (int i = 0; i < this->mesh->size(); ++i) {
            if (dvnDeltaPsi[i] > mcNorm) dvnDeltaPsi[i] = mcNorm;
            else if (dvnDeltaPsi[i] < -mcNorm) dvnDeltaPsi[i] = -mcNorm;
            if (std::abs(dvnDeltaPsi[i]/dvnPsi[i]) > maxRelUpd) maxRelUpd = std::abs(dvnDeltaPsi[i]/dvnPsi[i]);
            dvnPsi[i] = dvnPsi[i] + dvnDeltaPsi[i];
        }
    }
    return maxRelUpd;*/
}

template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::computePsiI() {

    this->writelog(LOG_INFO, "Calculating built-in potential");

    typedef std::pair<const Material*, unsigned> KeyT;
    std::map<KeyT, double> cache;

    dvnPsi0.reset(size, 0.);

    //auto iMesh = (this->mesh)->getMidpointsMesh();
    //auto temperatures = inTemperature(iMesh);
    auto iMeshE = (this->mesh)->getMidpointsMesh();
    auto temperaturesE = inTemperature(iMeshE);

    for (auto el: this->mesh->elements()) {
        size_t i = el.getIndex();
        // point and material in the middle of the element
        Vec < 2,double> midpoint = el.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        // average temperature on the element
        // double temp = 0.25 * (temperatures[loleftno] + temperatures[lorghtno] + temperatures[upleftno] + temperatures[uprghtno]); // LP_09.2015
        //double T(300.); // Temperature in the current element
        double T = temperaturesE[i]; // Temperature in the current element

        KeyT key = std::make_pair(material.get(), unsigned(0.5+T*100.)); // temperature precision 0.01 K
        auto found = cache.find(key);

        double epsi;
        if (found != cache.end()) {
            epsi = found->second;
        }
        else {
            if (material->kind() == Material::OXIDE || material->kind() == Material::DIELECTRIC || material->kind() == Material::EMPTY ) { // 26.01.2016
                cache[key] = epsi = 0.;
                continue;
            }
            // normalise material parameters and temperature
            double normEc0 = material->CB(T, 0., '*') / mEx;
            double normEv0 = material->VB(T, 0., '*', 'h') / mEx;
            double normNc = Neff(material->Me(T, 0., '*'), T) / mNx;
            double normNv = Neff(material->Mh(T, 0), T) / mNx;
            double normNd = material->Nd() / mNx;
            double normNa = material->Na() / mNx;
            double normEd = material->EactD(T) / mEx;
            double normEa = material->EactA(T) / mEx;
            double normT = T / mTx;
            std::size_t loop = 0;
            cache[key] = epsi = findPsiI(normEc0, normEv0, normNc, normNv, normNd, normNa, normEd, normEa, 1., 1., normT, loop);
        }

        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex();
        dvnPsi0[loleftno] += epsi;
        dvnPsi0[lorghtno] += epsi;
        dvnPsi0[upleftno] += epsi;
        dvnPsi0[uprghtno] += epsi;
    }
    divideByElements(dvnPsi0);

    if (conttype == SCHOTTKY) {
        // Store boundary conditions for current mesh
        auto vconst = voltage_boundary(this->mesh, this->geometry);
        for (auto cond: vconst) {
            for (auto i: cond.place) {
                if (cond.value == 0)
                    dvnPsi0[i] += mSchottkyN/mEx;
                else if (cond.value != 0)
                    dvnPsi0[i] += mSchottkyP/mEx;
            }
        }
    }
}

template <typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::findPsiI(double iEc0, double iEv0, double iNc, double iNv, double iNd, double iNa, double iEd, double iEa, double iFnEta, double iFpKsi, double iT, std::size_t &loop) const
{
    double tPsi0(0.), // calculated normalized initial potential
    tPsi0a = (-15.) / mEx, // normalized edge of the initial range
    tPsi0b = (15.) / mEx, // normalized edge of the initial range
    tPsi0h = (0.1) / mEx, // normalized step in the initial range calculations
    tN = 0., tP = 0., // normalized electron/hole concentrations
    tNtot, tNtota = (-1e30) / mNx, tNtotb = (1e30) / mNx; // normalized carrier concentration and its initial values for potentials at range edges

    // Initial calculations

    int tPsi0n = static_cast<int>(round((tPsi0b-tPsi0a)/tPsi0h)) + 1 ; // number of initial normalized potential values

    std::vector < double> tPsi0v(tPsi0n); // normalized potential values to check
    for (int i = 0; i < tPsi0n; ++i)
        tPsi0v[i] = tPsi0a + i*tPsi0h;

    for (int i = 0; i < tPsi0n; ++i) {
        tN = calcN(iNc, iFnEta, tPsi0v[i], iEc0, iT);
        tP = calcP(iNv, iFpKsi, tPsi0v[i], iEv0, iT);

        double iNdIon = iNd;
        double iNaIon = iNa;

        if (!mFullIon)
        {
            //this->writelog(LOG_RESULT, "Full ionization false");
            double gD(2.), gA(4.);
            double iNdTmp = (iNc/gD)*exp(-iEd);
            double iNaTmp = (iNv/gA)*exp(-iEa);
            iNdIon = iNd * (iNdTmp/(iNdTmp+tN));
            iNaIon = iNa * (iNaTmp/(iNaTmp+tP));
        }

        tNtot = tP - tN + iNdIon - iNaIon; // total normalized carrier concentration

        if (tNtot < 0.) {
            if (tNtot > tNtota) {
                tNtota = tNtot;
                tPsi0b = tPsi0v[i];
            }
        }
        else if (tNtot > 0.) {
            if (tNtot < tNtotb) {
                tNtotb = tNtot;
                tPsi0a = tPsi0v[i];
            }
        }
        else // found initial normalised potential
            return tPsi0v[i];
    }

    // Precise calculations

    double tPsiUpd = 1e30, // normalised potential update
            tTmpA, tTmpB; // temporary data

    std::size_t tL = 0; // loop counter
    while ((std::abs(tPsiUpd) > (maxerrPsiI)/mEx) && (tL < loopsPsiI)) {
        tTmpA = (tNtotb-tNtota) / (tPsi0b-tPsi0a);
        tTmpB = tNtota - tTmpA*tPsi0a;
        tPsi0 = - tTmpB/tTmpA; //Psi Check Value
        tN = calcN(iNc, iFnEta, tPsi0, iEc0, iT);
        tP = calcP(iNv, iFpKsi, tPsi0, iEv0, iT);

        double iNdIon = iNd;
        double iNaIon = iNa;

        if (!mFullIon) {
            double gD(2.), gA(4.);
            double iNdTmp = (iNc/gD) * exp(-iEd);
            double iNaTmp = (iNv/gA) * exp(-iEa);
            iNdIon = iNd * (iNdTmp / (iNdTmp+tN));
            iNaIon = iNa * (iNaTmp / (iNaTmp+tP));
        }

        tNtot = tP - tN + iNdIon - iNaIon; // total normalized carrier concentration

        if (tNtot < 0.) {
            tNtota = tNtot;
            tPsi0b = tPsi0;
        }
        else if (tNtot > 0.) {
            tNtotb = tNtot;
            tPsi0a = tPsi0;
        }
        else { // found initial normalized potential
            loop = tL;
            //this->writelog(LOG_DEBUG, "{0} loops done. Calculated energy level corresponding to the initial potential: {1} eV", tL, (tPsi0)*mEx); // TEST
            return tPsi0;
        }

        tPsiUpd = tPsi0b - tPsi0a;
        #ifndef NDEBUG
            if (!tL)
                this->writelog(LOG_DEBUG, "Initial potential correction: {0} eV", (tPsiUpd)*mEx); // TEST
            else
                this->writelog(LOG_DEBUG, " {0} eV", (tPsiUpd)*mEx); // TEST
        #endif
        ++tL;
    }

    loop = tL;
    //this->writelog(LOG_INFO, "{0} loops done. Calculated energy level corresponding to the initial potential: {1} eV", tL, (tPsi0)*mEx); // TEST

    return tPsi0;
}


template <typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::compute(unsigned loops) {
    switch (algorithm) {
        case ALGORITHM_CHOLESKY: return doCompute<DpbMatrix>(loops);
        case ALGORITHM_GAUSS: return doCompute<DgbMatrix>(loops);
        case ALGORITHM_ITERATIVE: return doCompute<SparseBandMatrix>(loops);
    }
    return 0.;
}

template <typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::findEnergyLevels() {
	this->writelog(LOG_INFO, "Finding energy levels..");
	
	hh2m = 0.5 * phys::hb_eV * phys::hb_J * 1e9 * 1e9 / phys::me; /// hb*hb/(2m), unit: eV*nm*nm, 10^9 is introduced to change m into nm
	Eshift = 20.; /// bands have to be up-shifted - we want only positive values of energy levels; unit: [eV]

	double kx = 0., ky = 0.; /// TODO

	bool potentialWell_el = false; /// true when electrons (el) are confined
	bool potentialWell_hh = false; /// true when heavy holes (hh) are confined
	bool potentialWell_lh = false; /// true when light holes (lh) are confined
	
	potentialWell_el = checkWell("el");

	if (potentialWell_el)
	{
		double dzdz1 = 1. / (dz*dz*1e6); /// 1/(dz*dz), unit: [1/nm^2]
		double dz1 = 1. / (dz*1e3); /// 1/(dz), unit: [1/nm]

		this->writelog(LOG_DETAIL, "Creating matrix for electrons..\n");

		int K = 1; /// the order of the small matrix for central-node for CB
		int N = nn * K; /// the order of the matrix for CB

		this->writelog(LOG_DETAIL, "\tsize of the matrix for CB: {0} x {1}", N, N);

		Eigen::MatrixXcd Hc(N, N); /// N - Hc size
		/// setting Hc = zeros
		std::complex<double> Hc_zero(0., 0.);
		for (size_t i=1; i<=N; ++i)
			for (size_t j=1; j <= N; ++j)
				Hc(i-1, j-1) = Hc_zero;

		for (size_t i=1; i<=nn; ++i) /// nn - number of nodes
		{
			Vec<2, double> point_LE; /// centre of the left element
            point_LE[0] = meshActMid->axis[0]->at(0); // TODO not only for 0
            point_LE[1] = meshActMid->axis[1]->at(i-1);

			Vec<2, double> point_RI; /// centre of the right element
            point_RI[0] = meshActMid->axis[0]->at(0); // TODO not only for 0
            point_RI[1] = meshActMid->axis[1]->at(i);
			
			shared_ptr<Material> m_LE = this->geometry->getMaterial(point_LE);
			shared_ptr<Material> m_RI = this->geometry->getMaterial(point_RI);

			double CBel = 0.5 * (m_LE->CB(T, 0.,'*') + m_RI->CB(T, 0., '*')) - (getPotentials(meshAct, INTERPOLATION_LINEAR)).at(i);
			this->writelog(LOG_DETAIL, "\tCBel for kp, node: {0}, CBel: {1}", i, CBel);

			//this->writelog(LOG_DETAIL, "creating Hc matrix, central node: {0}, materials: {1} and {2}", i, m_le->name(), m_ri->name());
						
			std::complex<double> Hc_11_LE(0., 0.), Hc_11_CE(0., 0.), Hc_11_RI(0., 0.); /// left/central/right 1x1 local matrix
			{
				double y0_LE = 1. / m_LE->Me(T).c00; // TODO or sth else than c00?
				double y0_RI = 1. / m_RI->Me(T).c00; // TODO or sth else than c00?
				double y0_CE = 0.5 * (y0_LE + y0_RI);

				/// 11
				double beta_11_LE = - y0_LE * hh2m;
				double beta_11_RI = - y0_RI * hh2m;
				double alpha_11_CE = y0_CE * hh2m * (kx * kx + ky * ky);
				double Ec0_11_CE = CBel + Eshift; // TODO CBel[i] must be CB from elem + Psi fo elem and then calc average for node
				double v_11_CE = Ec0_11_CE;
				double s_11_CE = 0.;
				/*if (strain) // TODO
				{
					double s_11_LE = 2. * vE[e_LE].ac() * (1. - vE[e_LE].c12() / vE[e_LE].c11()) * vE[e_LE].gEps(); /// [001]
					double s_11_RI = 2. * vE[e_RI].ac() * (1. - vE[e_RI].c12() / vE[e_RI].c11()) * vE[e_RI].gEps(); /// [001]
					s_11_CE = 0.5 * (s_11_LE + s_11_RI);
				}*/

				if (i>1)
				{
					Hc_11_LE.real(beta_11_LE * dzdz1); /// Re - real part., Im - imag part.
					Hc(i - 1, i - 2) = Hc_11_LE;
					//this->writelog(LOG_DETAIL, "Hc_11_LE: {0}", Hc_11_LE());
				}
				{
					Hc_11_CE.real(alpha_11_CE - beta_11_LE * dzdz1 - beta_11_RI * dzdz1 + v_11_CE + s_11_CE);
					Hc(i - 1, i - 1) = Hc_11_CE;
					//this->writelog(LOG_DETAIL, "Hc_11_CE: {0}", Hc_11_CE());
				}
				if (i<nn)
				{
					Hc_11_RI.real(beta_11_RI * dzdz1);
					Hc(i - 1, i) = Hc_11_RI;
					//this->writelog(LOG_DETAIL, "Hc_11_RI: {0}", Hc_11_RI());
				}
			}
		}
		this->writelog(LOG_INFO, "Done.");

		this->writelog(LOG_INFO, "Finding energy levels and wave functions for electrons..");
		Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
		ces.compute(Hc);
		int nEigVal = ces.eigenvalues().rows();
		this->writelog(LOG_INFO, "number of eigenvalues (Hc): {0}", nEigVal);
		if (nEigVal<1)
			return 1; /// no energy levels for electrons
		this->writelog(LOG_INFO, "Done.");

		{
			this->writelog(LOG_INFO, "Analyzing the solutions..");
			lev_el.clear(); /// lev_el - vector with energy levels for electrons
			n_lev_el = 0; /// n_lev_el - number of energy levels of electrons
			for (size_t i = 1; i < nEigVal; ++i)
			{
				if ((ces.eigenvalues()[i].real() - Eshift > CBelMin) && (ces.eigenvalues()[i].real() - Eshift < CBelMax))
				{
					std::vector<double> sum(K, 0.); /// we have to find out if this level corresponds to el
					for (size_t j = 1; j < nn*1; j += 1) /// "1" in both cases because only el are considered here
					{
						sum[0] += (pow(ces.eigenvectors().col(i)[j].real(), 2.) + pow(ces.eigenvectors().col(i)[j].imag(), 2.));
					}
					std::string carrier;
					if (sum[0])
					{
						carrier = "el";
						lev_el.push_back(ces.eigenvalues()[i].real() - Eshift);
						n_lev_el++;
					}
				}
			}
			this->writelog(LOG_INFO, "Done.");

			this->writelog(LOG_INFO, "Sorting electron energy levels..");
			std::sort(lev_el.begin(), lev_el.end()); /// sorting electron energy levels
			for (size_t i = 0; i < n_lev_el; ++i)
			{
				this->writelog(LOG_INFO, "energy level for electron {0}: {1} eV", i, lev_el[i]);
			}
			this->writelog(LOG_INFO, "Done.");
		}

		Hc.resize(0, 0);

		// TODO: hh and lh
	}

	this->writelog(LOG_INFO, "Done.");

    return 0;
}

template <typename Geometry2DType>
bool DriftDiffusionModel2DSolver<Geometry2DType>::checkWell(std::string _carrier) {
	if (_carrier == "el") /// electrons
	{
		std::vector<double> CBel;
		this->writelog(LOG_DETAIL, "Checking the confinement for electrons..");
		CBel.clear();
		for (size_t i = 0; i < ne+2; ++i) /// ne+2 because cladding elements also
		{
			//double z_avg = 0.5*(z[i] + z[i+1]);
			Vec<2, double> point = meshActMid->at(0,i);
            //point[0] = meshActMid->axis[0]->at(0); // TODO tu musi byc jakis element haxis dla danego obszaru
            //point[1] = meshActMid->axis[1]->at(i);

			//this->writelog(LOG_INFO, "position of element {0}: {1} um, {2} um", i, r_at_0, z_avg);
			
			shared_ptr<Material> material = this->geometry->getMaterial(point);
			//this->writelog(LOG_DETAIL, "material found");
			//this->writelog(LOG_DETAIL, "element {0}: {1}", i, material->name());

			//CBel.push_back(material->CB(T, 0., '*') - getPotentials(meshActMid->at(0,i)) * mEx);
		}
		/// filling CBel vector /// TODO
		//for (size_t i = 0; i < nn + 2; ++i)
		//{
		//	CBel.push_back(material->CB(T, 0., '*') - dvnPsi[..] * mEx);
		//}
		
		
		
		for (size_t i = 0; i < nn+2; ++i)
			CBel.push_back(5.0);
		for (size_t i = 60; i < 140; ++i)
			CBel[i] = 4.5;
		/// finding min. and max. for CB
		CBelMin = 1e6; 
		CBelMax = -1e6;
		for (size_t i = 0; i < nn+2; ++i) 
		{
			if (CBel[i] < CBelMin)
				CBelMin = CBel[i];
			if (CBel[i] > CBelMax)
				CBelMax = CBel[i];
		}
		/// max. CB at boundary
		CBel[0] = CBelMax; 
		CBel[nn+1] = CBelMax;
		//for (size_t i = 0; i < nz+2; ++i) /// TEST
		//	this->writelog(LOG_DETAIL, "node {0}: CBel = {1} eV", i, CBel[i]);

		this->writelog(LOG_INFO, "Done.");
	}

	return true; /// TODO
}

template <typename Geometry2DType>
template <typename MatrixT>
double DriftDiffusionModel2DSolver<Geometry2DType>::doCompute(unsigned loops)
{
    bool was_initialized = !this->initCalculation();
    needPsi0 |= !was_initialized;

    //heats.reset(); // LP_09.2015

    auto iMesh = (this->mesh)->getMidpointsMesh();
    auto temperatures = inTemperature(iMesh);

    // Store boundary conditions for current mesh
    auto vconst = voltage_boundary(this->mesh, this->geometry);

    this->writelog(LOG_INFO, "Running drift-diffusion calculations for a single voltage");

    MatrixT A(size, this->mesh->minorAxis()->size());
    DataVector<double> B(size);

    double errorPsi0 = 0.;

    if (needPsi0) {
        computePsiI();
        errorPsi0 = 2. * maxerrPsi0;
        unsigned iter = 0;
        while (errorPsi0 > maxerrPsi0 && iter < loopsPsi0) {
            setMatrix<CALC_PSI0>(A, B, vconst);
            solveMatrix(A, B);
            errorPsi0 = addCorr<CALC_PSI0>(B, vconst); // max. update
            this->writelog(LOG_DEBUG, "Initial potential maximum update: {0}", errorPsi0*mEx); // czy dla Fn i Fp tez bedzie mEx?
            iter += 1;
        }
        if (!dvnPsi) {
            dvnPsi = dvnPsi0.copy();
            savePsi();
        }
        saveN();
        saveP();
        needPsi0 = false;
    }

    assert(dvnPsi);

    // Apply boundary conditions of the first kind
    bool novoltage = true;
    for (auto cond: vconst) {
        for (auto i: cond.place) {
            double dU = cond.value / mEx;
            novoltage = novoltage && dU == 0.;
            dvnPsi[i] = dvnPsi0[i] + dU;
            dvnFnEta[i] = exp(-dU);
            dvnFpKsi[i] = exp(+dU);
        }
    }
    if (novoltage) {
        if (was_initialized) {
            std::copy(dvnPsi0.begin(), dvnPsi0.end(), dvnPsi.begin());
            dvnFnEta.fill(1.);
            dvnFpKsi.fill(1.);
        }
        return errorPsi0;
    }

    savePsi();
    saveFnEta();
    saveFpKsi();
    saveN();
    saveP();

    if (loops == 0) loops = std::numeric_limits<unsigned>::max();
    unsigned loopno = 0;
    double errorPsi = 2.*maxerrPsi, errorFn = 2.*maxerrFn, errorFp = 2.*maxerrFp, err;

    while ((errorPsi > maxerrPsi || errorFn > maxerrFn || errorFp > maxerrFp) && loopno < loops) {
        this->writelog(LOG_DETAIL, "Calculating potential");
        unsigned itersPsi = 0;
        errorPsi = 0.;
        err = 2.*maxerrPsi;
        while(err > maxerrPsi && itersPsi < loopsPsi) {
            setMatrix<CALC_PSI>(A, B, vconst);
            solveMatrix(A, B);
            err = addCorr<CALC_PSI>(B, vconst); // max. update
            if (err > errorPsi) errorPsi = err;
            this->writelog(LOG_DETAIL, "Maximum potential update: {0}", err*mEx); // czy dla Fn i Fp tez bedzie mEx?
            savePsi();
            saveN();
            saveP();
            itersPsi += 1;
        }
        this->writelog(LOG_DETAIL, "Calculating quasi-Fermi level for electrons");
        unsigned itersFn = 0;
        errorFn = 0.;
        err = 2.*maxerrFn;
        while(err > maxerrFn && itersFn < loopsFn) {
            setMatrix<CALC_FN>(A, B, vconst);
            solveMatrix(A, B);
            err = addCorr<CALC_FN>(B, vconst); // max. update
            if (err > errorFn) errorFn = err;
            this->writelog(LOG_DETAIL, "Maximum electrons quasi-Fermi level update: {0}", err*mEx); // czy dla Fn i Fp tez bedzie mEx?
            saveFnEta();
            saveN();
            itersFn += 1;
        }
        this->writelog(LOG_DETAIL, "Calculating quasi-Fermi energy level for holes");
        unsigned itersFp = 0;
        errorFp = 0.;
        err = 2.*maxerrFp;
        while(err > maxerrFp && itersFp < loopsFp) {
            setMatrix<CALC_FP>(A, B, vconst);
            solveMatrix(A, B);
            err = addCorr<CALC_FP>(B, vconst); // max. update
            if (err > errorFp) errorFp = err;
            this->writelog(LOG_DETAIL, "Maximum holes quasi-Fermi level update: {0}", err*mEx); // czy dla Fn i Fp tez bedzie mEx?
            saveFpKsi();
            saveP();
            itersFp += 1;
        }
    }

    // calculate electron and hole currents (jn and jp)
    for (auto el: this->mesh->elements()) { // PROBLEM
        size_t i = el.getIndex();
        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex();
        double dFnx = 0.5 * (- log(dvnFnEta[loleftno]) + log(dvnFnEta[lorghtno]) - log(dvnFnEta[upleftno]) + log(dvnFnEta[uprghtno])) // + before 0.5 due to ln(FnEta)=Fn relation
                             / ((el.getUpper0() - el.getLower0())/mXx); // normalised [dFn/dx]
        double dFny = 0.5 * (- log(dvnFnEta[loleftno]) - log(dvnFnEta[lorghtno]) + log(dvnFnEta[upleftno]) + log(dvnFnEta[uprghtno])) // + before 0.5 due to ln(FnEta)=Fn relation
                             / ((el.getUpper1() - el.getLower1())/mXx); // normalised [dFn/dy]
        double dFpx = - 0.5 * (- log(dvnFpKsi[loleftno]) + log(dvnFpKsi[lorghtno]) - log(dvnFpKsi[upleftno]) + log(dvnFpKsi[uprghtno])) // - before 0.5 due to -ln(FpKsi)=Fp relation
                             / ((el.getUpper0() - el.getLower0())/mXx); // normalised [dFp/dx]
        double dFpy = - 0.5 * (- log(dvnFpKsi[loleftno]) - log(dvnFpKsi[lorghtno]) + log(dvnFpKsi[upleftno]) + log(dvnFpKsi[uprghtno])) // - before 0.5 due to -ln(FpKsi)=Fp relation
                             / ((el.getUpper1() - el.getLower1())/mXx); // normalised [dFp/dy]

        double T = 0.25 * (temperatures[loleftno] + temperatures[lorghtno] + temperatures[upleftno] + temperatures[uprghtno]); // in (K)
        //double normT(T/mTx); // normalised temperature

        /*double n, p;
        if (calctype == CALC_PSI0) {
            double normNc = Neff(material->Me(T, 0., '*'), T) / mNx;
            double normEc0 = material->CB(T, 0., '*') / mEx;
            double normNv = Neff(material->Mh(T, 0.), T) / mNx;
            double normEv0 = material->VB(T, 0., '*') / mEx;
            double normT = T / mTx;
            double ePsi = 0.25 * (dvnPsi0[loleftno] + dvnPsi0[lorghtno] + dvnPsi0[upleftno] + dvnPsi0[uprghtno]);
            n = calcN(normNc, 1., ePsi, normEc0, normT);
            p = calcP(normNv, 1., ePsi, normEv0, normT);
        } else {
            n = dveN[i];
            p = dveP[i];
        }*/

        auto midpoint = el.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        double normMobN = 0.5*(material->mobe(T).c00+material->mobe(T).c11) / mMix;
        auto curN = vec(normMobN * dveN[i] * dFnx * mJx, normMobN * dveN[i] * dFny * mJx);

        double normMobP = 0.5*(material->mobh(T).c00+material->mobh(T).c11) / mMix;
        auto curP = vec(normMobP * dveP[i] * dFpx * mJx, normMobP * dveP[i] * dFpy * mJx);

        currentsN[i] = curN; // in (kA/cm^2)
        currentsP[i] = curP; // in (kA/cm^2)
    }

    outPotential.fireChanged();
    outFermiLevels.fireChanged();
    outBandEdges.fireChanged();
    outCurrentDensityForElectrons.fireChanged();
    outCurrentDensityForHoles.fireChanged();
    outCarriersConcentration.fireChanged();
    outHeat.fireChanged();

    return errorPsi + errorFn + errorFp;
}


template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::solveMatrix(DpbMatrix& A, DataVector<double>& B)
{
    int info = 0;

    this->writelog(LOG_DETAIL, "Solving matrix system");

    // Factorize matrix
    dpbtrf(UPLO, int(A.size), int(A.kd), A.data, int(A.ld+1), info);
    if (info < 0)
        throw CriticalException("{0}: Argument {1} of dpbtrf has illegal value", this->getId(), -info);
    else if (info > 0)
        throw ComputationError(this->getId(), "Leading minor of order {0} of the stiffness matrix is not positive-definite", info);

    // Find solutions
    dpbtrs(UPLO, int(A.size), int(A.kd), 1, A.data, int(A.ld+1), B.data(), int(B.size()), info);
    if (info < 0) throw CriticalException("{0}: Argument {1} of dpbtrs has illegal value", this->getId(), -info);

    // now A contains factorized matrix and B the solutions
}

template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::solveMatrix(DgbMatrix& A, DataVector<double>& B)
{
    int info = 0;
    this->writelog(LOG_DETAIL, "Solving matrix system");
    aligned_unique_ptr <int> ipiv(aligned_malloc<int>(A.size));

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

template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::solveMatrix(SparseBandMatrix& ioA, DataVector<double>& B)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    PrecondJacobi precond(ioA);

    //DataVector < double> x = dvnPsi.copy(); // We use previous potentials as initial solution // LP_09.2015
    DataVector < double> x(B.size(), 0.); // We use 0 as initial solution for corrections // LP_09.2015
    double err;
    try {
        std::size_t iter = solveDCG(ioA, precond, x.data(), B.data(), err, iterlim, itererr, logfreq, this->getId());
        this->writelog(LOG_DETAIL, "Conjugate gradient converged after {0} iterations.", iter);
    } catch (DCGError& exc) {
        throw ComputationError(this->getId(), "Conjugate gradient failed:, {0}", exc.what());
    }

    B = x;

    // now A contains factorized matrix and B the solutions
}

template <typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveHeatDensities()
{
    this->writelog(LOG_DETAIL, "Computing heat densities");

    heats.reset(this->mesh->getElementsCount());

    auto iMesh = (this->mesh)->getMidpointsMesh();
    auto temperatures = inTemperature(iMesh);

    /*if (heatmet == HEAT_JOULES)*/ {
        for (auto e: this->mesh->elements()) {
            size_t i = e.getIndex();
            size_t loleftno = e.getLoLoIndex();
            size_t lorghtno = e.getUpLoIndex();
            size_t upleftno = e.getLoUpIndex();
            size_t uprghtno = e.getUpUpIndex();
            auto midpoint = e.getMidpoint();
            auto material = this->geometry->getMaterial(midpoint);
            if (material->kind() == Material::EMPTY || this->geometry->hasRoleAt("noheat", midpoint))
                heats[i] = 0.;
            else {
                double T = 0.25 * (temperatures[loleftno] + temperatures[lorghtno] + temperatures[upleftno] + temperatures[uprghtno]); // in (K)

                double normMobN = 0.5*(material->mobe(T).c00+material->mobe(T).c11) / mMix;
                double normMobP = 0.5*(material->mobh(T).c00+material->mobh(T).c11) / mMix;

                heats[i] = ((currentsN[i].c0*currentsN[i].c0+currentsN[i].c1*currentsN[i].c1) / (normMobN*dveN[i]) + (currentsP[i].c0*currentsP[i].c0+currentsP[i].c1*currentsP[i].c1) / (normMobP*dveP[i])) * (1e12 / phys::qe);

                /*double dvx = 0.;//0.5e6 * (- potentials[loleftno] + potentials[lorghtno] - potentials[upleftno] + potentials[uprghtno]) // LP_09.2015
                               //     / (e.getUpper0() - e.getLower0()); // [grad(dV)] = V/m
                double dvy = 0.; //0.5e6 * (- potentials[loleftno] - potentials[lorghtno] + potentials[upleftno] + potentials[uprghtno]) // LP_09.2015
                              //      / (e.getUpper1() - e.getLower1()); // [grad(dV)] = V/m
                heats[i] = conds[i].c00 * dvx*dvx + conds[i].c11 * dvy*dvy;*/
            }
        }
    }
}


template <> double DriftDiffusionModel2DSolver<Geometry2DCartesian>::integrateCurrent(size_t vindex, bool onlyactive)
{
    if (!dvnPsi) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis[0]->size()-1; ++i) {
        auto element = mesh->element(i, vindex);
        if (!onlyactive || isActive(element.getMidpoint()))
            result += currentsN[element.getIndex()].c1 * element.getSize0() + currentsP[element.getIndex()].c1 * element.getSize0();
    }
    if (this->getGeometry()->isSymmetric(Geometry::DIRECTION_TRAN)) result *= 2.;
    return result * geometry->getExtrusion()->getLength() * 0.01; // kA/cm² µm² -->  mA;
}


template <> double DriftDiffusionModel2DSolver<Geometry2DCylindrical>::integrateCurrent(size_t vindex, bool onlyactive)
{
    if (!dvnPsi) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis[0]->size()-1; ++i) {
        auto element = mesh->element(i, vindex);
        if (!onlyactive || isActive(element.getMidpoint())) {
            double rin = element.getLower0(), rout = element.getUpper0();
            result += currentsN[element.getIndex()].c1 * (rout*rout - rin*rin) + currentsP[element.getIndex()].c1 * (rout*rout - rin*rin);
        }
    }
    return result * plask::PI * 0.01; // kA/cm² µm² -->  mA
}


template <typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::getTotalCurrent(size_t nact)
{
    size_t level = getActiveRegionMeshIndex(nact);
    return integrateCurrent(level, true);
}


template <typename Geometry2DType>
const LazyData < double> DriftDiffusionModel2DSolver<Geometry2DType>::getPotentials(shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) const
{
    if (!dvnPsi) throw NoValue("Potential");
    this->writelog(LOG_DEBUG, "Getting potentials");
    if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh, dvnPsi*mEx, dst_mesh, method, this->geometry); // here the potential is rescalled (*mEx)
}


template <typename Geometry2DType>
const LazyData <double> DriftDiffusionModel2DSolver<Geometry2DType>::getFermiLevels(
    FermiLevels::EnumType what, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) const
{
    if (what == FermiLevels::ELECTRONS) {
        if (!dvnFnEta) throw NoValue("Quasi-Fermi electron level");
        this->writelog(LOG_DEBUG, "Getting quasi-Fermi electron level");

        DataVector<double> dvnFn(size);
        for (size_t i = 0; i != dvnFnEta.size(); ++i) {
                if (dvnFnEta[i] > 0.) dvnFn[i] = log(dvnFnEta[i]) * mEx;
                else dvnFn[i] = 0.;
        }

        if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
        return interpolate(this->mesh, dvnFn, dst_mesh, method, this->geometry); // here the quasi-Fermi electron level is rescalled (*mEx)
    } else if (what == FermiLevels::HOLES) {
        if (!dvnFpKsi) throw NoValue("Quasi-Fermi hole level");
        this->writelog(LOG_DEBUG, "Getting quasi-Fermi hole level");

        DataVector<double> dvnFp(size);
        for (size_t i = 0; i != dvnFpKsi.size(); ++i) {
            if (dvnFpKsi[i] > 0.) dvnFp[i] = - log(dvnFpKsi[i]) * mEx;
            else dvnFp[i] = 0.;
        }

        if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
        return interpolate(this->mesh, dvnFp, dst_mesh, method, this->geometry); // here the quasi-Fermi hole level is rescalled (*mEx)
    }
    assert(0);
    std::abort();   // to silent warning in gcc/clang release build
}


template <typename Geometry2DType>
const LazyData < double> DriftDiffusionModel2DSolver<Geometry2DType>::getBandEdges(
    BandEdges::EnumType what, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    if (what == BandEdges::CONDUCTION) {
        if (!dvnPsi) throw NoValue("Conduction band edge");
        this->writelog(LOG_DEBUG, "Getting conduction band edge");

        DataVector<double> dvnEc(size, 0.);

        auto iMeshE = (this->mesh)->getMidpointsMesh();
        auto temperaturesE = inTemperature(iMeshE);

        //double T(300.); // TODO
        double T;

        for (auto e: this->mesh->elements()) {
            size_t i = e.getIndex();
            size_t loleftno = e.getLoLoIndex();
            size_t lorghtno = e.getUpLoIndex();
            size_t upleftno = e.getLoUpIndex();
            size_t uprghtno = e.getUpUpIndex();

            Vec <2,double> midpoint = e.getMidpoint();
            auto material = this->geometry->getMaterial(midpoint);

            T = temperaturesE[i]; // Temperature in the current element

            dvnEc[loleftno] += material->CB(T, 0., '*') - dvnPsi[loleftno] * mEx;
            dvnEc[lorghtno] += material->CB(T, 0., '*') - dvnPsi[lorghtno] * mEx;
            dvnEc[upleftno] += material->CB(T, 0., '*') - dvnPsi[upleftno] * mEx;
            dvnEc[uprghtno] += material->CB(T, 0., '*') - dvnPsi[uprghtno] * mEx;
        }
        divideByElements(dvnEc);

        if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
        return interpolate(this->mesh, dvnEc, dst_mesh, method, this->geometry); // here the conduction band edge is rescalled (*mEx)
    } else if (what == BandEdges::VALENCE_LIGHT || what == BandEdges::VALENCE_HEAVY) {
        if (!dvnPsi) throw NoValue("Valence band edge");
        this->writelog(LOG_DEBUG, "Getting valence band edge");

        DataVector<double> dvnEv(size, 0.);

        auto iMeshE = (this->mesh)->getMidpointsMesh();
        auto temperaturesE = inTemperature(iMeshE);

        //double T(300.); // TODO
        double T;

        for (auto e: this->mesh->elements()) {
            size_t i = e.getIndex();
            size_t loleftno = e.getLoLoIndex();
            size_t lorghtno = e.getUpLoIndex();
            size_t upleftno = e.getLoUpIndex();
            size_t uprghtno = e.getUpUpIndex();

            Vec<2,double> midpoint = e.getMidpoint();
            auto material = this->geometry->getMaterial(midpoint);

            T = temperaturesE[i]; // Temperature in the current element

            dvnEv[loleftno] += material->VB(T, 0., '*') - dvnPsi[loleftno] * mEx;
            dvnEv[lorghtno] += material->VB(T, 0., '*') - dvnPsi[lorghtno] * mEx;
            dvnEv[upleftno] += material->VB(T, 0., '*') - dvnPsi[upleftno] * mEx;
            dvnEv[uprghtno] += material->VB(T, 0., '*') - dvnPsi[uprghtno] * mEx;
        }
        divideByElements(dvnEv);

        if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
        return interpolate(this->mesh, dvnEv, dst_mesh, method, this->geometry); // here the valence band edge is rescalled (*mEx)
    }
    assert(0);
    std::abort();   // to silent warning in gcc/clang release build
}


template <typename Geometry2DType>
const LazyData < Vec<2>> DriftDiffusionModel2DSolver<Geometry2DType>::getCurrentDensitiesForElectrons(shared_ptr<const MeshD < 2> > dest_mesh, InterpolationMethod method)
{
    if (!dvnFnEta) throw NoValue("Current density");
    this->writelog(LOG_DEBUG, "Getting current densities");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    InterpolationFlags flags(this->geometry, InterpolationFlags::Symmetry::NP, InterpolationFlags::Symmetry::PN);
    auto result = interpolate(this->mesh->getMidpointsMesh(), currentsN, dest_mesh, method, flags);
    return LazyData < Vec<2>>(result.size(),
        [this, dest_mesh, result, flags](size_t i) {
            return this->geometry->getChildBoundingBox().contains(flags.wrap(dest_mesh->at(i)))? result[i] : Vec<2>(0.,0.);
        }
    );
}


template <typename Geometry2DType>
const LazyData < Vec<2>> DriftDiffusionModel2DSolver<Geometry2DType>::getCurrentDensitiesForHoles(shared_ptr<const MeshD < 2> > dest_mesh, InterpolationMethod method)
{
    if (!dvnFpKsi) throw NoValue("Current density");
    this->writelog(LOG_DEBUG, "Getting current densities");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    InterpolationFlags flags(this->geometry, InterpolationFlags::Symmetry::NP, InterpolationFlags::Symmetry::PN);
    auto result = interpolate(this->mesh->getMidpointsMesh(), currentsP, dest_mesh, method, flags);
    return LazyData < Vec<2>>(result.size(),
        [this, dest_mesh, result, flags](size_t i) {
            return this->geometry->getChildBoundingBox().contains(flags.wrap(dest_mesh->at(i)))? result[i] : Vec<2>(0.,0.);
        }
    );
}


template <typename Geometry2DType>
const LazyData < double> DriftDiffusionModel2DSolver<Geometry2DType>::getCarriersConcentration(
    CarriersConcentration::EnumType what, shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method)
{
    DataVector<double> dvn(size, 0.);

    //double T(300.); // TODO

    switch (what) {
        case CarriersConcentration::ELECTRONS:
            if (!dveN) throw NoValue("Electron concentration");
            this->writelog(LOG_DEBUG, "Getting electron concentration");

            for (auto e: this->mesh->elements()) {
                size_t i = e.getIndex();
                size_t loleftno = e.getLoLoIndex();
                size_t lorghtno = e.getUpLoIndex();
                size_t upleftno = e.getLoUpIndex();
                size_t uprghtno = e.getUpUpIndex();

                //Vec <2,double> midpoint = e.getMidpoint();
                //auto material = this->geometry->getMaterial(midpoint);

                dvn[loleftno] += dveN[i] * mNx;
                dvn[lorghtno] += dveN[i] * mNx;
                dvn[upleftno] += dveN[i] * mNx;
                dvn[uprghtno] += dveN[i] * mNx;
            }
            divideByElements(dvn);

            if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
            return interpolate(this->mesh, dvn, dst_mesh, method, this->geometry); // here the electron concentration is rescalled (*mNx)*/

        case CarriersConcentration::HOLES:
            if (!dveP) throw NoValue("Hole concentration");
            this->writelog(LOG_DEBUG, "Getting hole concentration");

            for (auto e: this->mesh->elements()) {
                size_t i = e.getIndex();
                size_t loleftno = e.getLoLoIndex();
                size_t lorghtno = e.getUpLoIndex();
                size_t upleftno = e.getLoUpIndex();
                size_t uprghtno = e.getUpUpIndex();

                //Vec <2,double> midpoint = e.getMidpoint();
                //auto material = this->geometry->getMaterial(midpoint);

                dvn[loleftno] += dveP[i] * mNx;
                dvn[lorghtno] += dveP[i] * mNx;
                dvn[upleftno] += dveP[i] * mNx;
                dvn[uprghtno] += dveP[i] * mNx;
            }
            divideByElements(dvn);

            if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
            return interpolate(this->mesh, dvn, dst_mesh, method, this->geometry); // here the hole concentration is rescalled (*mNx)*/
        
        default:
            throw NotImplemented("{}: Carriers concentration of this type", this->getId());
    }
}


template <typename Geometry2DType>
const LazyData < double> DriftDiffusionModel2DSolver<Geometry2DType>::getHeatDensities(shared_ptr<const MeshD < 2> > dest_mesh, InterpolationMethod method)
{
    if ((!dvnFnEta)||(!dvnFpKsi)) throw NoValue("Heat density");
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


template <typename Geometry2DType> /// COPIED FROM FREECARRIER!
void DriftDiffusionModel2DSolver<Geometry2DType>::detectActiveRegions()
{
	regions.clear();

	shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(this->geometry->getChild());
	shared_ptr<RectangularMesh<2>> points = mesh->getMidpointsMesh();

    size_t ileft = 0, iright = points->axis[0]->size();
	bool in_active = false;

	bool added_bottom_cladding = false;
	bool added_top_cladding = false;

    for (size_t r = 0; r < points->axis[1]->size(); ++r) {
		bool had_active = false; // indicates if we had active region in this layer
		shared_ptr<Material> layer_material;
		bool layer_QW = false;

        for (size_t c = 0; c < points->axis[0]->size(); ++c)
		{ // In the (possible) active region
			auto point = points->at(c, r);
			auto tags = this->geometry->getRolesAt(point);
			bool active = false; for (const auto& tag : tags) if (tag.substr(0, 6) == "active") { active = true; break; }
			bool QW = tags.find("QW") != tags.end()/* || tags.find("QD") != tags.end()*/;
			bool substrate = tags.find("substrate") != tags.end();

			if (substrate) {
				if (!materialSubstrate)
					materialSubstrate = this->geometry->getMaterial(point);
				else if (*materialSubstrate != *this->geometry->getMaterial(point))
					throw Exception("{0}: Non-uniform substrate layer.", this->getId());
			}

			if (QW && !active)
				throw Exception("{0}: All marked quantum wells must belong to marked active region.", this->getId());

			if (c < ileft) {
				if (active)
					throw Exception("{0}: Left edge of the active region not aligned.", this->getId());
			}
			else if (c >= iright) {
				if (active)
					throw Exception("{0}: Right edge of the active region not aligned.", this->getId());
			}
			else {
				// Here we are inside potential active region
				if (active) {
					if (!had_active) {
						if (!in_active)
						{ // active region is starting set-up new region info
							regions.emplace_back(mesh->at(c, r));
							ileft = c;
						}
						layer_material = this->geometry->getMaterial(point);
						layer_QW = QW;
					}
					else {
						if (*layer_material != *this->geometry->getMaterial(point))
							throw Exception("{0}: Non-uniform active region layer.", this->getId());
						if (layer_QW != QW)
							throw Exception("{0}: Quantum-well role of the active region layer not consistent.", this->getId());
					}
				}
				else if (had_active) {
					if (!in_active) {
						iright = c;

						// add layer below active region (cladding) LUKASZ
						/*auto bottom_material = this->geometry->getMaterial(points->at(ileft,r-1));
						for (size_t cc = ileft; cc < iright; ++cc)
						if (*this->geometry->getMaterial(points->at(cc,r-1)) != *bottom_material)
						throw Exception("{0}: Material below quantum well not uniform.", this->getId());
						auto& region = regions.back();
                        double w = mesh->axis[0]->at(iright) - mesh->axis[0]->at(ileft);
                        double h = mesh->axis[1]->at(r) - mesh->axis[1]->at(r-1);
						region.origin += Vec<2>(0., -h);
						this->writelog(LOG_DETAIL, "Adding bottom cladding; h = {0}",h);
						region.layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w, h), bottom_material));*/
					}
					else
						throw Exception("{0}: Right edge of the active region not aligned.", this->getId());
				}
				had_active |= active;
			}
		}
		in_active = had_active;

		// Now fill-in the layer info
		ActiveRegionInfo* region = regions.empty() ? nullptr : &regions.back();
		if (region) {
			if (!added_bottom_cladding) {
				if (r == 0)
					throw Exception("{0}: Active region cannot start from the edge of the structure.", this->getId());
				// add layer below active region (cladding) LUKASZ
				auto bottom_material = this->geometry->getMaterial(points->at(ileft, r - 1));
				for (size_t cc = ileft; cc < iright; ++cc)
					if (*this->geometry->getMaterial(points->at(cc, r - 1)) != *bottom_material)
						throw Exception("{0}: Material below active region not uniform.", this->getId());
				auto& region = regions.back();
                double w = mesh->axis[0]->at(iright) - mesh->axis[0]->at(ileft);
                double h = mesh->axis[1]->at(r) - mesh->axis[1]->at(r - 1);
				region.origin += Vec<2>(0., -h);
				//this->writelog(LOG_DETAIL, "Adding bottom cladding; h = {0}",h);
				region.layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w, h), bottom_material));
				region.bottom = h;
				added_bottom_cladding = true;
			}

            double h = mesh->axis[1]->at(r + 1) - mesh->axis[1]->at(r);
            double w = mesh->axis[0]->at(iright) - mesh->axis[0]->at(ileft);
			if (in_active) {
				size_t n = region->layers->getChildrenCount();
				shared_ptr<Block<2>> last;
				if (n > 0) last = static_pointer_cast<Block<2>>(static_pointer_cast<Translation<2>>(region->layers->getChildNo(n - 1))->getChild());
				assert(!last || last->size.c0 == w);
				if (last && layer_material == last->getRepresentativeMaterial() && layer_QW == region->isQW(region->size() - 1)) {
					//TODO check if usage of getRepresentativeMaterial is fine here (was material)
					last->setSize(w, last->size.c1 + h);
				}
				else {
					auto layer = plask::make_shared<Block<2>>(Vec<2>(w, h), layer_material);
					if (layer_QW) layer->addRole("QW");
					region->layers->push_back(layer);
				}
			}
			else {
				if (!added_top_cladding) {

					// add layer above active region (top cladding)
					auto top_material = this->geometry->getMaterial(points->at(ileft, r));
					for (size_t cc = ileft; cc < iright; ++cc)
						if (*this->geometry->getMaterial(points->at(cc, r)) != *top_material)
							throw Exception("{0}: Material above quantum well not uniform.", this->getId());
					region->layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w, h), top_material));
					//this->writelog(LOG_DETAIL, "Adding top cladding; h = {0}",h);

					ileft = 0;
                    iright = points->axis[0]->size();
					region->top = h;
					added_top_cladding = true;
				}
			}
		}
	}
	if (!regions.empty() && regions.back().isQW(regions.back().size() - 1))
		throw Exception("{0}: Quantum-well cannot be located at the edge of the structure.", this->getId());

	if (strained && !materialSubstrate)
		throw BadInput(this->getId(), "Strained quantum wells requested but no layer with substrate role set");

	this->writelog(LOG_DETAIL, "Found {0} active region{1}", regions.size(), (regions.size() == 1) ? "" : "s");
	for (auto& region : regions) region.summarize(this);
}












//template <typename GeometryType>
//void DriftDiffusionModel2DSolver<GeometryType>::detectActiveRegions()
//{
//	this->writelog(LOG_INFO, "Detecting active regions..");
//	
//	regions.clear();
//
//	shared_ptr< MeshAxis > axis_vert = this->mesh->vert(); /// for the whole structure
//	shared_ptr< MeshAxis > axis_tran = this->mesh->tran(); /// for the whole structure
//	double r_at_0 = 0.5 * (axis_tran->at(0) + axis_tran->at(1)); // TODO
//	
//    shared_ptr<RectangularMesh<2>> meshMid = this->mesh->getMidpointsMesh();
//	bool found_substrate = false;
//	bool found_active = false;
//
//	double z1(-1.), z2(-1.); /// min. and max. z for kp method [um]
//	double zSub(-1.); /// z for substrate [um]
//
//    //for (std::size_t i = 0; i < axis_vert->size(); ++i) // TEST
//	//{
//	//	this->writelog(LOG_INFO, "axis_vert[{0}]: {1}", i, axis_vert->at(i));
//	//}
//    for (std::size_t i = 0; i < axis_vert->size()-1; ++i)
//	{
//		double zA = axis_vert->at(i); /// z bottom
//		double zB = axis_vert->at(i+1); /// z top
//		double z_avg = 0.5*(zA + zB);
//		Vec<2, double> point;
//		point[0] = r_at_0;
//		point[1] = z_avg;
//
//		std::string role_ = "-";
//		auto tags = this->geometry->getRolesAt(point);
//
//		if (tags.find("substrate") != tags.end())
//		{
//			//this->writelog(LOG_INFO, "axis_vert_mid[{0}] {1}, substrate", i, z_avg);
//			if (!found_substrate) /// first time in substrate
//			{
//				zSub = z_avg;
//			}		
//			found_substrate = true;
//		}
//		if (tags.find("active") != tags.end())
//		{
//			//in_active = true;			
//			//this->writelog(LOG_INFO, "axis_vert_mid[{0}] {1}, active", i, z_avg);
//			if (!found_active) /// first time in active
//			{
//				z1 = zA;
//				z2 = zB;
//			}
//			else
//			{
//				if (z1 > zA)
//					z1 = zA;
//				if (z2 < zB)
//					z2 = zB;
//			}
//			found_active = true;
//		}
//	}
//	
//	this->writelog(LOG_INFO, "active region is from z = {0} um to z = {1} um", z1, z2); /// in [um]
//	this->writelog(LOG_INFO, "active region thickness: {0} nm", (z2-z1)*1e3); /// [um] -> [nm]
//
//	z.clear();
//	//z.push_back(0.); /// so here z[0] = 0, but later the z.size will be stored here
//	dz = 0.1e-3; /// in [um], default value: 0.1 nm
//	nn = static_cast<int> ((z2 - z1 + 1e-6) / dz) + 1;
//	this->writelog(LOG_INFO, "no of nodes for kp method: {0}", nn);
//	z.push_back(z1 - dz); /// bottom cladding, left edge of the element is set here: z[0]
//	for (std::size_t i = 0; i < nn; ++i)
//		z.push_back(z1 + i*dz);
//	z.push_back(z2 + dz); /// top cladding
//	ne = nn - 1;
//
//	auto vaxis = plask::make_shared<OrderedAxis>();
//	auto haxis = plask::make_shared<OrderedAxis>();
//	this->writelog(LOG_INFO, "vaxis size: {0}", vaxis->size());
//	this->writelog(LOG_INFO, "haxis size: {0}", haxis->size());
//	OrderedAxis::WarningOff vaxiswoff(vaxis);
//	OrderedAxis::WarningOff haxiswoff(haxis);
//	for (std::size_t i = 0; i != nn+2; ++i)
//	{
//		vaxis->addPoint(z[i]);
//	}
//	haxis->addPoint(1e-4); // TODO - tu musza byc wartosci z haxis dla danego obszaru czynnego
//	haxis->addPoint(2e-4);
//	haxis->addPoint(3e-4);
//
//	this->writelog(LOG_INFO, "vaxis size: {0}", vaxis->size());
//	this->writelog(LOG_INFO, "haxis size: {0}", haxis->size());
//
//	meshAct = plask::make_shared<RectangularMesh<2>>(haxis, vaxis, RectangularMesh<2>::ORDER_01);
//	meshActMid = meshAct->getMidpointsMesh(); // LUKI TODO
//	
//	this->writelog(LOG_INFO, "MeshAct 0 1: {0} {1} {2}", meshAct->at(0,0), meshAct->at(0,1), meshAct->at(0,2));
//	this->writelog(LOG_INFO, "MeshAct 0 1: {0} {1} {2}", meshAct->axis[1]->at(0), meshAct->axis[1]->at(1), meshAct->axis[1]->at(2));
//
//	this->writelog(LOG_INFO, "Done.");
//	
//	/*shared_ptr<RectangularMesh<2>> points = mesh->getMidpointsMesh();
//
//    for (size_t r = 0; r < points->axis[1]->size(); ++r) {
//        bool had_active = false; // indicates if we had active region in this layer
//        shared_ptr<Material> layer_material;
//        bool layer_QW = false;
//
//        for (size_t c = 0; c < points->axis[0]->size(); ++c)
//        { // In the (possible) active region
//
//
//            else {
//                // Here we are inside potential active region
//                if (active) {
//                    if (!had_active) {
//                        if (!in_active)
//                        { // active region is starting set-up new region info
//                            regions.emplace_back(mesh->at(c,r));
//                            ileft = c;
//                        }
//                        layer_material = this->geometry->getMaterial(point);
//                        layer_QW = QW;
//                    } else {
//                        if (*layer_material != *this->geometry->getMaterial(point))
//                            throw Exception("{0}: Non-uniform active region layer.", this->getId());
//                        if (layer_QW != QW)
//                            throw Exception("{0}: Quantum-well role of the active region layer not consistent.", this->getId());
//                    }
//                } else if (had_active) {
//                    if (!in_active) {
//                        iright = c;
//
//                        // add layer below active region (cladding) LUKASZ
//                        //auto bottom_material = this->geometry->getMaterial(points->at(ileft,r-1));
//                        //for (size_t cc = ileft; cc < iright; ++cc)
//                        //    if (*this->geometry->getMaterial(points->at(cc,r-1)) != *bottom_material)
//                        //        throw Exception("{0}: Material below quantum well not uniform.", this->getId());
//                        //auto& region = regions.back();
//                        //double w = mesh->axis[0]->at(iright) - mesh->axis[0]->at(ileft);
//                        //double h = mesh->axis[1]->at(r) - mesh->axis[1]->at(r-1);
//                        //region.origin += Vec<2>(0., -h);
//                        //this->writelog(LOG_DETAIL, "Adding bottom cladding; h = {0}",h);
//                        //region.layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w, h), bottom_material));
//                    } else
//                        throw Exception("{0}: Right edge of the active region not aligned.", this->getId());
//                }
//                had_active |= active;
//            }
//        }
//        in_active = had_active;
//
//        // Now fill-in the layer info
//        ActiveRegionInfo* region = regions.empty()? nullptr : &regions.back();
//        if (region) {
//            if (!added_bottom_cladding) {
//                if (r == 0)
//                    throw Exception("{0}: Active region cannot start from the edge of the structure.", this->getId());
//                // add layer below active region (cladding) LUKASZ
//                auto bottom_material = this->geometry->getMaterial(points->at(ileft,r-1));
//                for (size_t cc = ileft; cc < iright; ++cc)
//                    if (*this->geometry->getMaterial(points->at(cc,r-1)) != *bottom_material)
//                        throw Exception("{0}: Material below active region not uniform.", this->getId());
//                auto& region = regions.back();
//                double w = mesh->axis[0]->at(iright) - mesh->axis[0]->at(ileft);
//                double h = mesh->axis[1]->at(r) - mesh->axis[1]->at(r-1);
//                region.origin += Vec<2>(0., -h);
//                //this->writelog(LOG_DETAIL, "Adding bottom cladding; h = {0}",h);
//                region.layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w, h), bottom_material));
//                region.bottom = h;
//                added_bottom_cladding = true;
//            }
//
//            double h = mesh->axis[1]->at(r+1) - mesh->axis[1]->at(r);
//            double w = mesh->axis[0]->at(iright) - mesh->axis[0]->at(ileft);
//            if (in_active) {
//                size_t n = region->layers->getChildrenCount();
//                shared_ptr<Block<2>> last;
//                if (n > 0) last = static_pointer_cast<Block<2>>(static_pointer_cast<Translation<2>>(region->layers->getChildNo(n-1))->getChild());
//                assert(!last || last->size.c0 == w);
//                if (last && layer_material == last->getRepresentativeMaterial() && layer_QW == region->isQW(region->size()-1)) {
//                    //TODO check if usage of getRepresentativeMaterial is fine here (was material)
//                    last->setSize(w, last->size.c1 + h);
//                } else {
//                    auto layer = plask::make_shared<Block<2>>(Vec<2>(w,h), layer_material);
//                    if (layer_QW) layer->addRole("QW");
//                    region->layers->push_back(layer);
//                }
//            } else {
//                if (!added_top_cladding) {
//
//                    // add layer above active region (top cladding)
//                    auto top_material = this->geometry->getMaterial(points->at(ileft,r));
//                    for (size_t cc = ileft; cc < iright; ++cc)
//                        if (*this->geometry->getMaterial(points->at(cc,r)) != *top_material)
//                            throw Exception("{0}: Material above quantum well not uniform.", this->getId());
//                    region->layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w,h), top_material));
//                    //this->writelog(LOG_DETAIL, "Adding top cladding; h = {0}",h);
//
//                    ileft = 0;
//                    iright = points->axis[0]->size();
//                    region->top = h;
//                    added_top_cladding = true;
//                }
//            }
//        }
//    }
//    if (!regions.empty() && regions.back().isQW(regions.back().size()-1))
//        throw Exception("{0}: Quantum-well cannot be located at the edge of the structure.", this->getId());
//
//    if (strained && !materialSubstrate)
//        throw BadInput(this->getId(), "Strained quantum wells requested but no layer with substrate role set");
//	*/
//    this->writelog(LOG_DETAIL, "Found {0} active region{1}", regions.size(), (regions.size()==1)?"":"s");
//    for (auto& region: regions) region.summarize(this);
//}


template <typename Geometry2DType> /// COPIED FROM FREECARRIER!
void DriftDiffusionModel2DSolver<Geometry2DType>::ActiveRegionInfo::summarize(const DriftDiffusionModel2DSolver<Geometry2DType>* solver)
{
	holes = BOTH_HOLES;
	auto bbox = layers->getBoundingBox();
	total = bbox.upper[1] - bbox.lower[1] - bottom - top;
	solver->writelog(LOG_DETAIL, "coordinates | bbox.upper: {0} um, bbox.lower: {1} um, bottom: {2} um, top: {3} um, total: {4} um", bbox.upper[1], bbox.lower[1], bottom, top, total);
	materials.clear(); materials.reserve(layers->children.size());
	thicknesses.clear(); thicknesses.reserve(layers->children.size());
	for (const auto& layer : layers->children) {
		auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layer.get())->getChild().get());
		auto material = block->singleMaterial();
		if (!material) throw plask::Exception("{}: Active region can consist only of solid layers", solver->getId());
		auto bbox = static_cast<GeometryObjectD<2>*>(layer.get())->getBoundingBox();
		double thck = bbox.upper[1] - bbox.lower[1];
		solver->writelog(LOG_DETAIL, "layer | material: {0}, thickness: {1} um", material->name(), thck);
		materials.push_back(material);
		thicknesses.push_back(thck);
	}
	/*double substra = solver->strained ? solver->materialSubstrate->lattC(solver->T0, 'a') : 0.; // TODO moze cos z tego wziac???
	if (materials.size() > 2) {
		Material* material = materials[0].get();
		double e;
		if (solver->strained) { double latt = material->lattC(solver->T0, 'a'); e = (substra - latt) / latt; }
		else e = 0.;
		double el0 = material->CB(solver->T0, e, 'G'),
			hh0 = material->VB(solver->T0, e, 'G', 'H'),
			lh0 = material->VB(solver->T0, e, 'G', 'L');
		material = materials[1].get();
		if (solver->strained) { double latt = material->lattC(solver->T0, 'a'); e = (substra - latt) / latt; }
		else e = 0.;
		double el1 = material->CB(solver->T0, e, 'G'),
			hh1 = material->VB(solver->T0, e, 'G', 'H'),
			lh1 = material->VB(solver->T0, e, 'G', 'L');
		for (size_t i = 2; i < materials.size(); ++i) {
			material = materials[i].get();
			if (solver->strained) { double latt = material->lattC(solver->T0, 'a'); e = (substra - latt) / latt; }
			else e = 0.;
			double el2 = material->CB(solver->T0, e, 'G');
			double hh2 = material->VB(solver->T0, e, 'G', 'H');
			double lh2 = material->VB(solver->T0, e, 'G', 'L');
			if ((el0 < el1 && el1 > el2) || (hh0 > hh1 && hh1 < hh2) || (lh0 > lh1 && lh1 < lh2)) {
				if (i != 2 && i != materials.size() - 1) {
					bool eb = (el0 < el1 && el1 > el2);
					if (eb != (hh0 > hh1 && hh1 < hh2)) holes = ConsideredHoles(holes & ~HEAVY_HOLES);
					if (eb != (lh0 > lh1 && lh1 < lh2)) holes = ConsideredHoles(holes & ~LIGHT_HOLES);
				}
				if (holes == NO_HOLES)
					throw Exception("{0}: Quantum wells in conduction band do not coincide with wells is valence band", solver->getId());
				if ((el0 < el1 && el1 > el2) || (hh0 > hh1 && hh1 < hh2 && holes & HEAVY_HOLES) || (lh0 > lh1 && lh1 < lh2 && holes & LIGHT_HOLES))
					wells.push_back(i - 1);
			}
			else if (i == 2) wells.push_back(0);
			if (el2 != el1) { el0 = el1; el1 = el2; }
			if (hh2 != hh1) { hh0 = hh1; hh1 = hh2; }
			if (lh2 != lh1) { lh0 = lh1; lh1 = lh2; }
		}
	}
	if (wells.back() < materials.size() - 2) wells.push_back(materials.size() - 1);
	totalqw = 0.;
	for (size_t i = 0; i < thicknesses.size(); ++i)
		if (isQW(i)) totalqw += thicknesses[i];*/
}


/*
template <typename Geometry2DType>
const LazyData < Tensor2<double>> DriftDiffusionModel2DSolver<Geometry2DType>::getConductivity(shared_ptr<const MeshD < 2> > dest_mesh, InterpolationMethod) {
    this->initCalculation();
    this->writelog(LOG_DEBUG, "Getting conductivities");
    loadConductivities();
    InterpolationFlags flags(this->geometry);
    return LazyData < Tensor2<double>>(new LazyDataDelegateImpl < Tensor2<double>>(dest_mesh->size(),
        [this, dest_mesh, flags](size_t i) -> Tensor2 < double> {
            auto point = flags.wrap(dest_mesh->at(i));
            size_t x = this->mesh->axis[0]->findUpIndex(point[0]),
                   y = this->mesh->axis[1]->findUpIndex(point[1]);
            if (x == 0 || y == 0 || x == this->mesh->axis[0]->size() || y == this->mesh->axis[1]->size())
                return Tensor2<double>(NAN);
            else
                return this->conds[this->mesh->elements(x-1, y-1).getIndex()];
        }
    ));
}


template <>
double DriftDiffusionModel2DSolver<Geometry2DCartesian>::getTotalEnergy() {
    double W = 0.;
    auto T = inTemperature(this->mesh->getMidpointsMesh());
    for (auto e: this->mesh->elements) {
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
double DriftDiffusionModel2DSolver<Geometry2DCylindrical>::getTotalEnergy() {
    double W = 0.;
    auto T = inTemperature(this->mesh->getMidpointsMesh());
    for (auto e: this->mesh->elements) {
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
    return 2.*PI * 0.5e-18 * phys::epsilon0 * W; // 1e-18 µm³ -> m³
}


template <typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::getCapacitance() {

    if (this->voltage_boundary.size() != 2) {
        throw BadInput(this->getId(), "Cannot estimate applied voltage (exactly 2 voltage boundary conditions required)");
    }

    double U = voltage_boundary[0].value - voltage_boundary[1].value;

    return 2e12 * getTotalEnergy() / (U*U); // 1e12 F -> pF
}


template <>
double DriftDiffusionModel2DSolver<Geometry2DCartesian>::getTotalHeat() {
    double W = 0.;
    if (!heats) saveHeatDensities(); // we will compute heats only if they are needed
    for (auto e: this->mesh->elements) {
        double width = e.getUpper0() - e.getLower0();
        double height = e.getUpper1() - e.getLower1();
        W += width * height * heats[e.getIndex()];
    }
    return geometry->getExtrusion()->getLength() * 1e-15 * W; // 1e-15 µm³ -> m³, W -> mW
}

template <>
double DriftDiffusionModel2DSolver<Geometry2DCylindrical>::getTotalHeat() {
    double W = 0.;
    if (!heats) saveHeatDensities(); // we will compute heats only if they are needed
    for (auto e: this->mesh->elements) {
        double width = e.getUpper0() - e.getLower0();
        double height = e.getUpper1() - e.getLower1();
        double r = e.getMidpoint().rad_r();
        W += width * height * r * heats[e.getIndex()];
    }
    return 2e-15*PI * W; // 1e-15 µm³ -> m³, W -> mW
}*/


template < > std::string DriftDiffusionModel2DSolver<Geometry2DCartesian>::getClassName() const { return "ddm2d.DriftDiffusion2D"; }
template < > std::string DriftDiffusionModel2DSolver<Geometry2DCylindrical>::getClassName() const { return "ddm2d.DriftDiffusionCyl"; }

PLASK_NO_CONVERSION_WARNING_BEGIN
template struct PLASK_SOLVER_API DriftDiffusionModel2DSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API DriftDiffusionModel2DSolver<Geometry2DCylindrical>;
PLASK_NO_WARNING_END

}}} // namespace plask::electrical::thermal
