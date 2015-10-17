#include "ddm2d.h"

namespace plask { namespace solvers { namespace drift_diffusion {

template<typename Geometry2DType>
DriftDiffusionModel2DSolver<Geometry2DType>::DriftDiffusionModel2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DType, RectangularMesh<2>>(name),
    mRsrh(false),
    mRrad(false),
    mRaug(false),
    mPol(false),
    mFullIon(true),
    mTx(300.),
    mEx(phys::kB_eV*mTx),
    mNx(1e18),
    mEpsRx(12.9),
    mXx(sqrt((phys::epsilon0*phys::kB_J*mTx*mEpsRx)/(phys::qe*phys::qe*mNx))*1e3),
    //mKx(100.),
    mMix(1000.),
    mRx(((phys::kB_J*mTx*mMix*mNx)/(phys::qe*mXx*mXx))*1e8),
    mJx(((phys::kB_J*mNx)*mTx*mMix/mXx)*10.),
    mtx(mNx/mRx),
    mBx(mRx/(mNx*mNx)),
    mCx(mRx/(mNx*mNx*mNx)),
    //mHx(((mKx*mTx)/(mXx*mXx))*1e12),
    dU(0.002),
    maxDelPsi0(2.),
    maxDelPsi(0.1*dU),
    maxDelFn(1e20),
    maxDelFp(1e20),
    stat("MB"),
    //loopno(0),
    //maxerr(0.05),
    heatmet(HEAT_JOULES),
    outPotential(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getPotentials),
    outQuasiFermiEnergyLevelForElectrons(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getQuasiFermiEnergyLevelsForElectrons),
    outQuasiFermiEnergyLevelForHoles(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getQuasiFermiEnergyLevelsForHoles),
    outConductionBandEdge(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getConductionBandEdges),
    outValenceBandEdge(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getValenceBandEdges),
//     outCurrentDensity(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getCurrentDensities),
//     outHeat(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getHeatDensities),
//     outConductivity(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getConductivity),
    algorithm(ALGORITHM_CHOLESKY),
    maxerrPsiI(1e-12),
    maxerrPsi0(1e-12),
    maxerrPsi(1e-12),
    maxerrFn(1e-6),
    maxerrFp(1e-6),
    iterlimPsiI(10000),
    iterlimPsi0(200),
    iterlimPsi(3),
    iterlimFn(3),
    iterlimFp(3),
    itererr(1e-8),
    iterlim(10000),
    logfreq(500)
{
    onInvalidate();
    inTemperature = 300.;
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "voltage")
            this->readBoundaryConditions(manager, source, voltage_boundary);
        /*else if (param == "loop") {
            maxerr = source.getAttribute<double>("maxerr", maxerr);
            source.requireTagEnd();*/
        else if (param == "loop") {
            maxerrPsi0 = source.getAttribute<double>("maxerrPsi0", maxerrPsi0);
            maxerrPsi = source.getAttribute<double>("maxerrPsi", maxerrPsi);
            maxerrFn = source.getAttribute<double>("maxerrFn", maxerrFn);
            maxerrFp = source.getAttribute<double>("maxerrFp", maxerrFp);
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
            auto condjunc = source.getAttribute<double>("pnjcond");
            //if (condjunc) setCondJunc(*condjunc); // LP_09.2015
            auto wavelength = source.getAttribute<double>("wavelength");
            if (wavelength) inWavelength = *wavelength;
            heatmet = source.enumAttribute<HeatMethod>("heat")
                .value("joules", HEAT_JOULES)
                .value("wavelength", HEAT_BANDGAP)
                .get(heatmet);
            for (auto attr: source.getAttributes()) {
                if (attr.first == "beta" || attr.first == "Vt" || attr.first == "js" || attr.first == "pnjcond" || attr.first == "wavelength" || attr.first == "heat") continue;
                /*if (attr.first.substr(0,4) == "beta") {
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
                    throw XMLUnexpectedAttrException(source, attr.first);*/ //LP_09.2015
            }
            source.requireTagEnd();
        }
        else if (param == "contacts") {
            source.requireTagEnd();
        }
        else
            this->parseStandardConfiguration(source, manager);
    }
}


template<typename Geometry2DType>
DriftDiffusionModel2DSolver<Geometry2DType>::~DriftDiffusionModel2DSolver() {
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    //loopno = 0;
    size = this->mesh->size();

    dvnNodeInfo.reset(size,0.);
    dvnPsi0.reset(size, 0.);
    dvnPsi.reset(size, 0.);
    dvnDeltaPsi.reset(size, 0.);
    dvnFn.reset(size, 0.);
    dvnFp.reset(size, 0.);
    dvnFnEta.reset(size, 1.);
    dvnFpKsi.reset(size, 1.);
    dvnDeltaFn.reset(size, 0.);
    dvnDeltaFp.reset(size, 0.);
    dvnEc.reset(size, 0.);
    dvnEv.reset(size, 0.);

    dvePsiI.reset(this->mesh->elements.size(), 0.);
    dvePsi.reset(this->mesh->elements.size(), 0.);
    dveFn.reset(this->mesh->elements.size(), 0.);
    dveFp.reset(this->mesh->elements.size(), 0.);
    dveFnEta.reset(this->mesh->elements.size(), 1.);
    dveFpKsi.reset(this->mesh->elements.size(), 1.);
    dveN.reset(this->mesh->elements.size(), 0.);
    dveP.reset(this->mesh->elements.size(), 0.);


    //currents.reset(this->mesh->elements.size(), vec(0.,0.));
    //conds.reset(this->mesh->elements.size()); //LP_09.2015
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::onInvalidate() {
    //conds.reset(); //LP_09.2015
    dvnNodeInfo.reset();
    dvnPsi0.reset();
    dvnPsi.reset();
    dvnDeltaPsi.reset();
    dvnFn.reset();
    dvnFp.reset();
    dvnFnEta.reset();
    dvnFpKsi.reset();
    dvnDeltaFn.reset();
    dvnDeltaFp.reset();
    dvnEc.reset();
    dvnEv.reset();
    dvePsiI.reset();
    dvePsi.reset();
    dveFn.reset();
    dveFp.reset();
    dveFnEta.reset();
    dveFpKsi.reset();
    dveN.reset();
    dveP.reset();
    //currents.reset();
    //heats.reset();
    //junction_conductivity.reset(1, default_junction_conductivity); //LP_09.2015
}


template<>
inline void DriftDiffusionModel2DSolver<Geometry2DCartesian>::setLocalMatrix(double&, double&, double&, double&,
                            double&, double&, double&, double&, double&, double&,
                            double, double, const Vec<2,double>&) {
    return;
}

template<>
inline void DriftDiffusionModel2DSolver<Geometry2DCylindrical>::setLocalMatrix(double& k44, double& k33, double& k22, double& k11,
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
template <typename MatrixT> // add deltaPsi = 0 on p- and n-contacts
void DriftDiffusionModel2DSolver<Geometry2DType>::applyBC(MatrixT& A, DataVector<double>& B,
                                                          const BoundaryConditionsWithMesh<RectangularMesh<2>, double>& bvoltage) {
    // boundary conditions of the first kind
    for (auto cond: bvoltage) {
        for (auto r: cond.place) {
            A(r,r) = 1.;
            double val = B[r] = 0.;//cond.value; // LP_09.2015
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

template<typename Geometry2DType> // add deltaPsi = 0 on p- and n-contacts
void DriftDiffusionModel2DSolver<Geometry2DType>::applyBC(SparseBandMatrix& A, DataVector<double>& B,
                                                          const BoundaryConditionsWithMesh<RectangularMesh<2>, double> &bvoltage) {
    // boundary conditions of the first kind
    for (auto cond: bvoltage) {
        for (auto r: cond.place) {
            double* rdata = A.data + LDA*r;
            *rdata = 1.;
            double val = B[r] = 0.; // cond.value; // LP_09.2015
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
void DriftDiffusionModel2DSolver<Geometry2DType>::setMatrix(std::string calctype, MatrixT& A, DataVector<double>& B,
                                                            const BoundaryConditionsWithMesh<RectangularMesh<2>, double> &bvoltage)
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, bands=%2%{%3%})", A.size, A.kd+1, A.ld+1);

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
        auto material = this->geometry->getMaterial(midpoint);

        double hx = elemwidth/mXx; // normalised value
        double hy = elemheight/mXx; // normalised value
        double n = dveN[i]; // normalised value
        double p = dveP[i]; // normalised value
        double T(300.);
        double Ec0 = material->CB(T, 0., 'G')/mEx; // normalised value
        double Ev0 = material->VB(T, 0., 'G')/mEx; // normalised value
        double Nc = material->Nc(T, 0., 'G')/mNx; // normalised value
        double Nv = material->Nv(T, 0., 'G')/mNx; // normalised value
        //double Ni = material->Ni(T)/mNx; // normalised value
        double Ne = Nc * exp(dvePsi[i]-Ec0); // normalised value
        double Nh = Nv * exp(-dvePsi[i]+Ev0); // normalised value
        double eps = material->eps(T)/mEpsRx; // normalised value
        double Nd = material->Nd()/mNx; // normalised value
        double Na = material->Na()/mNx; // normalised value
        double Ed = 0.050/mEx; // normalised value
        double Ea = 0.150/mEx; // normalised value
        double mobN = 0.5*(material->mob(T).c00+material->mob(T).c11)/mMix; // normalised value // TODO
        double mobP = 0.5*(material->mob(T).c00+material->mob(T).c11)/mMix; // normalised value // TODO

        double yn(0.);
        if (stat == "MB") yn = 1.;
        else if (stat == "FD") yn = calcFD12(log(dveFnEta[i])+dvePsi[i]-Ec0)/(dveFnEta[i]*exp(dvePsi[i]-Ec0));

        double yp(0.);
        if (stat == "MB") yp = 1.;
        else if (stat == "FD") yp = calcFD12(log(dveFpKsi[i])-dvePsi[i]+Ev0)/(dveFpKsi[i]*exp(-dvePsi[i]+Ev0));

        double kk(0.), kx(0.), ky(0.), gg(0.), ff(0.);

        if ((calctype == "Psi") || (calctype == "Psi0")) {
            kk = 1. / (3.*(hx*0.5)*(hy*0.5));
            kx = eps * (hy*0.5) * (hy*0.5);
            ky = eps * (hx*0.5) * (hx*0.5);
            gg = (1./9.) * (p + n) * (hx*0.5) * (hy*0.5);
            double iNdIon = Nd;
            double iNaIon = Na;
            if (!mFullIon)
            {
                double gD(2.), gA(4.);
                double iNdTmp = (Nc/gD)*exp(-Ed);
                double iNaTmp = (Nv/gA)*exp(-Ea);
                iNdIon = Nd * (iNdTmp/(iNdTmp+n));
                iNaIon = Na * (iNaTmp/(iNaTmp+p));
            }
            ff = - (hx*0.5) * (hy*0.5) * (p - n + iNdIon - iNaIon);
        }
        else if (calctype == "Fn") {
            kk = 1. / (3.*(hx*0.5)*(hy*0.5));
            kx = mobN * Ne * yn * (hy*0.5) * (hy*0.5);
            ky = mobN * Ne * yn * (hx*0.5) * (hx*0.5);
            ff = gg = 0.;

            /*if (ttE->getL()->getID()=="active") {
                if (mRsrh) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * Ne * yn * (p + ni) * (ttE->getL()->getTpNorm() * ni + ttE->getL()->getTnNorm() * p)
                        / pow(ttE->getL()->getTpNorm() * (n + ni) + ttE->getL()->getTnNorm() * (p + ni), 2.));
                    tFtmp += ((hx*0.5) * (hy*0.5) * (n * p - ni * ni) / (ttE->getL()->getTpNorm() * (n + ni) + ttE->getL()->getTnNorm() * (p + ni)));
                }
                if (mRrad) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * ttE->getL()->getBNorm() * Ne * yn * p);
                    tFtmp += ((hx*0.5) * (hy*0.5) * ttE->getL()->getBNorm() * (n * p - ni * ni));
                }
                if (mRaug) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * Ne * yn * ((ttE->getL()->getCnNorm() * (2. * n * p - ni * ni) + ttE->getL()->getCpNorm() * p * p)));
                    tFtmp += ((hx*0.5) * (hy*0.5) * (ttE->getL()->getCnNorm() * n + ttE->getL()->getCpNorm() * p) * (n * p - ni * ni));
                }
            }*/
        }
        else if (calctype == "Fp")	{
            kk = 1. / (3.*(hx*0.5)*(hy*0.5));
            kx = mobP * Nh * yp * (hy*0.5) * (hy*0.5);
            ky = mobP * Nh * yp * (hx*0.5) * (hx*0.5);
            ff = gg = 0.;

            /*if (ttE->getL()->getID()=="active") { // TODO (only in active?)
                if (mRsrh) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * Nh * yp * (n + ni) * (ttE->getL()->getTnNorm() * ni + ttE->getL()->getTpNorm() * n)
                        / pow(ttE->getL()->getTpNorm() * (n + ni) + ttE->getL()->getTnNorm() * (p + ni), 2.));
                    tFtmp += ((hx*0.5) * (hy*0.5) * (n * p - ni * ni) / (ttE->getL()->getTpNorm() * (n + ni) + ttE->getL()->getTnNorm() * (p + ni)));
                }
                if (mRrad) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * ttE->getL()->getBNorm() * Nh * yp * n);
                    tFtmp += ((hx*0.5) * (hy*0.5) * ttE->getL()->getBNorm() * (n * p - ni * ni));
                }
                if (mRaug) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * Nh * yp * ((ttE->getL()->getCpNorm() * (2. * n * p - ni * ni) + ttE->getL()->getCnNorm() * n * n)));
                    tFtmp += ((hx*0.5) * (hy*0.5) * (ttE->getL()->getCnNorm() * n + ttE->getL()->getCpNorm() * p) * (n * p - ni * ni));
                }
            }*/
        }

        // set symmetric matrix components
        double k44, k33, k22, k11, k43, k21, k42, k31, k32, k41;
        double g44, g33, g22, g11, g43, g21, g42, g31, g32, g41;
        double f1, f2, f3, f4;
        double v1, v2, v3, v4;

        // local K
        k44 = k33 = k22 = k11 = (kx+ky)*kk;
        k43 = k21 = 0.5*(-2.*kx+ky)*kk;
        k42 = k31 = 0.5*(-kx-ky)*kk;
        k32 = k41 = 0.5*(kx-2.*ky)*kk;

        // local G
        g44 = g33 = g22 = g11 = 4.*gg;
        g21 = g41 = g32 = g43 = 2.*gg;
        g31 = g42 = 1.*gg;

        // local F
        f4 = f3 = f2 = f1 = ff;

        // set stiffness matrix
        //setLocalMatrix(k44, k33, k22, k11, k43, k21, k42, k31, k32, k41, ky, elemwidth, midpoint); // TODO uncomment and correct after takng cylindrical structures into account

        A(loleftno, loleftno) += (k11+g11);
        A(lorghtno, lorghtno) += (k22+g22);
        A(uprghtno, uprghtno) += (k33+g33);
        A(upleftno, upleftno) += (k44+g44);

        A(lorghtno, loleftno) += (k21+g21);
        A(uprghtno, loleftno) += (k31+g31);
        A(upleftno, loleftno) += (k41+g41);
        A(uprghtno, lorghtno) += (k32+g32);
        A(upleftno, lorghtno) += (k42+g42);
        A(upleftno, uprghtno) += (k43+g43);

        if ((calctype == "Psi") || (calctype == "Psi0")) {
            v1 = dvnPsi[loleftno];
            v2 = dvnPsi[lorghtno];
            v3 = dvnPsi[uprghtno];
            v4 = dvnPsi[upleftno];
        }
        else if (calctype == "Fn") {
            v1 = dvnFnEta[loleftno];
            v2 = dvnFnEta[lorghtno];
            v3 = dvnFnEta[uprghtno];
            v4 = dvnFnEta[upleftno];
        }
        else if (calctype == "Fp") {
            v1 = dvnFpKsi[loleftno];
            v2 = dvnFpKsi[lorghtno];
            v3 = dvnFpKsi[uprghtno];
            v4 = dvnFpKsi[upleftno];
        }

        B[loleftno] += -(k11*v1 + k21*v2 + k31*v3 + k41*v4 + f1);
        B[lorghtno] += -(k21*v1 + k22*v2 + k32*v3 + k42*v4 + f2);
        B[uprghtno] += -(k31*v1 + k32*v2 + k33*v3 + k43*v4 + f3);
        B[upleftno] += -(k41*v1 + k42*v2 + k43*v3 + k44*v4 + f4);

        /*double kx = conds[i].c00;
        double ky = conds[i].c11;*/
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


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::savePsi()
{
    for (auto el: this->mesh->elements) {
        size_t i = el.getIndex();
        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex();

        dvePsi[i] = 0.25 * (dvnPsi[loleftno] + dvnPsi[lorghtno] + dvnPsi[upleftno] + dvnPsi[uprghtno]);
    }
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveFn()
{
    for (auto el: this->mesh->elements) {
        size_t i = el.getIndex();
        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex();

        dveFn[i] = 0.25 * (dvnFn[loleftno] + dvnFn[lorghtno] + dvnFn[upleftno] + dvnFn[uprghtno]);
    }
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveFp()
{
    for (auto el: this->mesh->elements) {
        size_t i = el.getIndex();
        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex();

        dveFp[i] = 0.25 * (dvnFp[loleftno] + dvnFp[lorghtno] + dvnFp[upleftno] + dvnFp[uprghtno]);
    }
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveFnEta()
{
    for (auto el: this->mesh->elements) {
        size_t i = el.getIndex();
        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex();

        dveFnEta[i] = 0.25 * (dvnFnEta[loleftno] + dvnFnEta[lorghtno] + dvnFnEta[upleftno] + dvnFnEta[uprghtno]);
    }
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveFpKsi()
{
    for (auto el: this->mesh->elements) {
        size_t i = el.getIndex();
        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex();

        dveFpKsi[i] = 0.25 * (dvnFpKsi[loleftno] + dvnFpKsi[lorghtno] + dvnFpKsi[upleftno] + dvnFpKsi[uprghtno]);
    }
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveN()
{
    for (auto e: this->mesh->elements)
    {
        size_t i = e.getIndex();
        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        double T(300.); // TODO
        double normNc = material->Nc(T, 0., 'G')/mNx;
        double normEc0 = material->CB(T, 0., 'G')/mEx;
        double normT = T/mTx;

        dveN[i] = calcN(normNc, dveFnEta[i], dvePsi[i], normEc0, normT);
    }
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveP()
{
    for (auto e: this->mesh->elements)
    {
        size_t i = e.getIndex();
        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        double T(300.); // TODO
        double normNv = material->Nv(T, 0., 'G')/mNx;
        double normEv0 = material->VB(T, 0., 'G')/mEx;
        double normT = T/mTx;

        dveP[i] = calcP(normNv, dveFpKsi[i], dvePsi[i], normEv0, normT);
    }
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveEc()
{
    dvnEc.reset(size, 0.);

    double T(300.); // TODO

    for (auto e: this->mesh->elements) {
        //size_t i = e.getIndex();
        size_t loleftno = e.getLoLoIndex();
        size_t lorghtno = e.getUpLoIndex();
        size_t upleftno = e.getLoUpIndex();
        size_t uprghtno = e.getUpUpIndex(); //LP_09.2015

        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        dvnEc[loleftno] += material->CB(T, 0., 'G')/mEx - dvnPsi[loleftno];
        dvnEc[lorghtno] += material->CB(T, 0., 'G')/mEx - dvnPsi[lorghtno];
        dvnEc[upleftno] += material->CB(T, 0., 'G')/mEx - dvnPsi[upleftno];
        dvnEc[uprghtno] += material->CB(T, 0., 'G')/mEx - dvnPsi[uprghtno];
    }

    for (int i = 0; i < this->mesh->size(); ++i) {
        dvnEc[i] /= dvnNodeInfo[i];
    }


    /*for (auto e: this->mesh->elements)
    {
        size_t i = e.getIndex();
        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        double T(300.); // TODO
        double normEc0 = material->CB(T, 0., 'G')/mEx;
        dveEc[i] = normEc0 - dvePsi[i];
    }*/

    //for (auto e: this->mesh->size())
    /*for (int i = 0; i < this->mesh->size(); ++i)
    {
        //size_t i = e.getIndex();
        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        double T(300.); // TODO
        double normEc0 = material->CB(T, 0., 'G')/mEx;
        //dveEc[i] = 3.;
        //dveEc[i] = normEc0 - dvePsi[i];
    }*/
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveEv()
{
    dvnEv.reset(size, 0.);

    double T(300.); // TODO

    for (auto e: this->mesh->elements) {
        //size_t i = e.getIndex();
        size_t loleftno = e.getLoLoIndex();
        size_t lorghtno = e.getUpLoIndex();
        size_t upleftno = e.getLoUpIndex();
        size_t uprghtno = e.getUpUpIndex(); //LP_09.2015

        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        dvnEv[loleftno] += material->VB(T, 0., 'G')/mEx - dvnPsi[loleftno];
        dvnEv[lorghtno] += material->VB(T, 0., 'G')/mEx - dvnPsi[lorghtno];
        dvnEv[upleftno] += material->VB(T, 0., 'G')/mEx - dvnPsi[upleftno];
        dvnEv[uprghtno] += material->VB(T, 0., 'G')/mEx - dvnPsi[uprghtno];
    }

    for (int i = 0; i < this->mesh->size(); ++i) {
        dvnEv[i] /= dvnNodeInfo[i];
    }

    /*for (auto e: this->mesh->elements)
    {
        size_t i = e.getIndex();
        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        double T(300.); // TODO
        double normEv0 = material->VB(T, 0., 'G')/mEx;
        dveEv[i] = normEv0 - dvePsi[i];
    }*/

    //for (auto e: this->mesh->size())
    /*for (int i = 0; i < this->mesh->size(); ++i)
    {
        //size_t i = e.getIndex();
        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        double T(300.); // TODO
        double normEv0 = material->VB(T, 0., 'G')/mEx;
        //dveEc[i] = 4.;
        //dveEv[i] = normEv0 - dvePsi[i];
    }*/
}


template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::addCorr(std::string calctype, const BoundaryConditionsWithMesh<RectangularMesh<2>, double>& vconst)
{
    double err = -1.;

    for (auto cond: vconst)
        for (auto i: cond.place) {
            dvnDeltaPsi[i] = 0.;
            dvnDeltaFn[i] = 0.;
            dvnDeltaFp[i] = 0.;
        }

    // TODO dla innych calctype
    // Może tak:
    // std::vector<bool> isconst(mesh->size(), false);
    // for (auto cond: vconst)
    //     for (auto i: cond.place)
    //         isconst[i] = true;
    
    //double tMaxRelUpd = 0.; // update/old_value = this will be the result

    if (calctype == "Psi0") {
        err = 0.;
        double normDel = maxDelPsi0/mEx;
        for (int i = 0; i < this->mesh->size(); ++i) {
            //reduce update
            if (dvnDeltaPsi[i] > normDel) dvnDeltaPsi[i] = normDel;
            else if (dvnDeltaPsi[i] < -normDel) dvnDeltaPsi[i] = -normDel;
            //calculate max. update
            double corr = std::abs(dvnDeltaPsi[i]);
            if (corr > err) err = corr;
            //update datavector with potentials
            dvnPsi[i] = dvnPsi[i] + dvnDeltaPsi[i]; // KRAKOW - NIE ROBIC TEGO W WEZLACH GDZIE JEST ZADANY POTENCJAL
            dvnPsi0[i] = dvnPsi[i];
        }
        this->writelog(LOG_DETAIL, "Maximal update for the potential at U=0V: %1%V", err*mEx);
    }
    else if (calctype == "Psi") {
        err = 0.;
        double normDel = maxDelPsi/mEx;
        for (int i = 0; i < this->mesh->size(); ++i) {
            //reduce update
            if (dvnDeltaPsi[i] > normDel) dvnDeltaPsi[i] = normDel;
            else if (dvnDeltaPsi[i] < -normDel) dvnDeltaPsi[i] = -normDel;
            //calculate max. update
            double corr = std::abs(dvnDeltaPsi[i]);
            if (corr > err) err = corr;
            //update datavector with potentials
            dvnPsi[i] = dvnPsi[i] + dvnDeltaPsi[i]; // KRAKOW - NIE ROBIC TEGO W WEZLACH GDZIE JEST ZADANY POTENCJAL
        }
        this->writelog(LOG_DETAIL, "Maximal update for the potential: %1%V", err*mEx);
    }
    else if (calctype == "Fn") {
        err = 0.;
        //double normDel = maxDelFn/mEx;
        for (int i = 0; i < this->mesh->size(); ++i) {
            //double oldFn = dvnFn[i]; // normalised velue
            //double newFn = 0.; // normalised value
            double dFnEta = dvnDeltaFn[i];
            dvnFnEta[i] += dFnEta;
            if (std::abs(dFnEta/dvnFnEta[i]) > err) err = std::abs(dFnEta/dvnFnEta[i]);
            if (dvnFnEta[i]>0)
                dvnFn[i] = log(dvnFnEta[i]);
            else
                dvnFn[i] = 0.;

            /*if (dvnDeltaFn[i] > 0) newFn = log(dvnDeltaFn[i]); // normalised value
            double deltaFn = newFn - oldFn;
            //reduce update
            if (deltaFn > normDel) deltaFn = normDel;
            else if (deltaFn < -normDel) deltaFn = -normDel;
            //calculate max. update
            double corr = std::abs(deltaFn);
            if (corr > err) err = corr;
            //update datavector with quasi-Fermi energy level for electrons
            dvnFn[i] = dvnFn[i] + deltaFn; // KRAKOW - NIE ROBIC TEGO W WEZLACH GDZIE JEST ZADANY POTENCJAL
            dvnFnEta[i] = exp(dvnFn[i]);*/ // KRAKOW - NIE ROBIC TEGO W WEZLACH GDZIE JEST ZADANY POTENCJAL
        }
        this->writelog(LOG_DETAIL, "Maximal relative update for the quasi-Fermi energy level for electrons: %1%.", err);
    }
    else if (calctype == "Fp") {
        err = 0.;
        //double normDel = maxDelFp/mEx;
        for (int i = 0; i < this->mesh->size(); ++i) {
            //double oldFp = dvnFp[i]; // normalised velue
            //double newFp = 0.; // normalised value

            double dFpKsi = dvnDeltaFp[i];
            dvnFpKsi[i] += dFpKsi;
            if (std::abs(dFpKsi/dvnFpKsi[i]) > err) err = std::abs(dFpKsi/dvnFpKsi[i]);
            if (dvnFpKsi[i]>0)
                dvnFp[i] = -log(dvnFpKsi[i]);
            else
                dvnFn[i] = 0.;


            /*
            if (dvnDeltaFp[i] > 0) newFp = -log(dvnDeltaFp[i]); // normalised value
            double deltaFp = newFp - oldFp;
            //reduce update
            if (deltaFp > normDel) deltaFp = normDel;
            else if (deltaFp < -normDel) deltaFp = -normDel;
            //calculate max. update
            double corr = std::abs(deltaFp);
            if (corr > err) err = corr;
            //update datavector with quasi-Fermi energy level for electrons
            dvnFp[i] = dvnFp[i] + deltaFp; // KRAKOW - NIE ROBIC TEGO W WEZLACH GDZIE JEST ZADANY POTENCJAL
            dvnFpKsi[i] = exp(-dvnFp[i]);*/ // KRAKOW - NIE ROBIC TEGO W WEZLACH GDZIE JEST ZADANY POTENCJAL
        }
        this->writelog(LOG_DETAIL, "Maximal relative update for the quasi-Fermi energy level for holes: %1%.", err);
    }
    return err; // for Psi -> normalised (max. delPsi)

    /*double maxRelUpd(0.);
    double mcNorm;
    if (calctype =="Psi0") {
        mcNorm = maxDelPsi0/mEx;
        for (int i = 0; i < this->mesh->size(); ++i) {
            if (dvnDeltaPsi[i] > mcNorm) dvnDeltaPsi[i] = mcNorm;
            else if (dvnDeltaPsi[i] < -mcNorm) dvnDeltaPsi[i] = -mcNorm;
            if (std::abs(dvnDeltaPsi[i]/dvnPsi[i]) > maxRelUpd) maxRelUpd = std::abs(dvnDeltaPsi[i]/dvnPsi[i]);
            dvnPsi[i] = dvnPsi[i] + dvnDeltaPsi[i];
        }
    }
    else if (calctype == "Psi") {
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


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::increaseVoltage() {
    // boundary conditions of the first kind
    auto vconst = voltage_boundary(this->mesh, this->geometry);

    for (auto cond: vconst) {
        for (auto r: cond.place) {
            double dU = cond.value;
            // KRAKOW - W WEZLACH W KTORYCH MAM WARUNEK BRZEGOWY NA POTENCJAL (ALE TYLKO TAM GDZIE PRZYLOZONE NAPIECIE JEST POWYZEJ 0) CHCE USTAWIC WARTOSCI TAK JAK PONIZEJ
            //int i = 0; // CO ZAMIAST 0?
            dvnPsi[r] = (dvnPsi0[r] + dU/mEx);
            dvnFn[r] = -dU/mEx;
            dvnFp[r] = -dU/mEx;
            dvnFnEta[r] = exp(dvnFn[r]);
            dvnFpKsi[r] = exp(-dvnFp[r]);

        }
    }

    savePsi();
    saveEc();
    saveEv();
    saveFn();
    saveFp();
    saveFnEta();
    saveFpKsi();
    saveN();
    saveP();
}

template<typename Geometry2DType>
int DriftDiffusionModel2DSolver<Geometry2DType>::computePsiI(unsigned loops) {
    this->writelog(LOG_INFO, "Calculating initial potential");

    this->initCalculation();

    std::string oldmatname = "-";
    double oldNd = 0.;
    double oldNa = 0.;
    double oldPsiI = 0.;

    for (auto el: this->mesh->elements) {
        size_t i = el.getIndex();
        // point and material in the middle of the element
        Vec<2,double> midpoint = el.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        std::string newmatname = material->name();
        double newNd = material->Nd();
        double newNa = material->Na();

        if ((oldmatname == newmatname) && (oldNd == newNd) && (oldNa == newNa)) {
            dvePsiI[i] = oldPsiI;
            continue;
        }

        // average temperature on the element
        // double temp = 0.25 * (temperatures[loleftno] + temperatures[lorghtno] + temperatures[upleftno] + temperatures[uprghtno]); // LP_09.2015

        double T(300.); // Temperature in the current element
        // normalise material parameters and temperature
        double normEc0 = material->CB(T, 0., 'G')/mEx;
        double normEv0 = material->VB(T, 0., 'G', 'h')/mEx;
        double normNc = material->Nc(T, 0., 'G')/mNx;
        double normNv = material->Nv(T, 0., 'G')/mNx;
        double normNd = material->Nd()/mNx;
        double normNa = material->Na()/mNx;
        double normEd = material->EactD(T)/mEx;
        double normEa = material->EactA(T)/mEx;
        double normT = T/mTx;

        //std::tie(kx,ky) = std::tuple<double,double>(material->thermk(temp, thickness[elem.getIndex()]));

        int loop = 0.;
        double tPsiI = findPsiI(normEc0, normEv0, normNc, normNv, normNd, normNa, normEd, normEa, 1., 1., normT, loop);
        this->writelog(LOG_DETAIL, "Calculated initial potential: %1%V (element no. %2%, %3% iterations)", tPsiI*mEx, i, loop);
        dvePsiI[i] = tPsiI;
        oldmatname = newmatname;
        oldNd = newNd;
        oldNa = newNa;
        oldPsiI = tPsiI;
    }

    setPsiI();
    savePsi();
    saveEc();
    saveEv();
    saveN();
    saveP();

    outPotential.fireChanged();

    return 0;
}


template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::computePsi0(unsigned loops) {
    this->writelog(LOG_INFO, "Calculating potential for U = 0V");
    double errorPsi0 = compute("Psi0",1);
    int itersPsi0 = 0;
    while ((errorPsi0 > maxerrPsi0) && (itersPsi0<iterlimPsi0)) { // 100
         errorPsi0 = compute("Psi0",1);
         itersPsi0 += 1;
    }
    return errorPsi0;
}


template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::computePsi(unsigned loops) {
    this->writelog(LOG_INFO, "Calculating potential");
    double errorPsi = compute("Psi",1);
    int itersPsi = 0;
    while ((errorPsi > maxerrPsi) && (itersPsi<iterlimPsi)) { // 3
         errorPsi = compute("Psi",1);
         itersPsi += 1;
    }
    this->writelog(LOG_INFO, "Calculating quasi-Fermi energy level for electrons");
    double errorFn = compute("Fn",1);
    int itersFn = 0;
    while ((errorFn > maxerrFn) && (itersFn<iterlimFn)) { // 3
         errorFn = compute("Fn",1);
         itersFn += 1;
    }
    this->writelog(LOG_INFO, "Calculating quasi-Fermi energy level for holes");
    double errorFp = compute("Fp",1);
    int itersFp = 0;
    while ((errorFp > maxerrFp) && (itersFp<iterlimFp)) { // 3
         errorFp = compute("Fp",1);
         itersFp += 1;
    }
    return 0.;
}


template<typename Geometry2DType> //LP_09.2015
void DriftDiffusionModel2DSolver<Geometry2DType>::setPsiI() {
    this->writelog(LOG_INFO, "Setting initial potential in nodes");

    for (auto el: this->mesh->elements) {
        size_t i = el.getIndex();
        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex(); //LP_09.2015

        dvnNodeInfo[loleftno] += 1;
        dvnNodeInfo[lorghtno] += 1;
        dvnNodeInfo[upleftno] += 1;
        dvnNodeInfo[uprghtno] += 1;

        dvnPsi[loleftno] += dvePsiI[i];
        dvnPsi[lorghtno] += dvePsiI[i];
        dvnPsi[upleftno] += dvePsiI[i];
        dvnPsi[uprghtno] += dvePsiI[i];
    }

    for (int i = 0; i < this->mesh->size(); ++i) {
        dvnPsi[i] /= dvnNodeInfo[i];
    }
}


template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::compute(std::string calctype, unsigned loops) {
    switch (algorithm) {
        case ALGORITHM_CHOLESKY: return doCompute<DpbMatrix>(calctype, loops);
        case ALGORITHM_GAUSS: return doCompute<DgbMatrix>(calctype, loops);
        case ALGORITHM_ITERATIVE: return doCompute<SparseBandMatrix>(calctype, loops);
    }
    return 0.;
}


template<typename Geometry2DType>
template <typename MatrixT>
double DriftDiffusionModel2DSolver<Geometry2DType>::doCompute(std::string calctype, unsigned loops)
{
    //this->initCalculation(); // LP_09.2015

    //heats.reset(); // LP_09.2015

    // Store boundary conditions for current mesh
    auto vconst = voltage_boundary(this->mesh, this->geometry);

    //this->writelog(LOG_INFO, "Running drift_diffusion calculations");

    size_t loopno = 0;

    MatrixT A(size, this->mesh->minorAxis()->size());

    //double err = 0.;LP_09.2015
    //toterr = 0.;
    double totDel = 0.;

#   ifndef NDEBUG
        //if (!dvPsi.unique()) this->writelog(LOG_DEBUG, "Potential data held by something else...");// LP_09.2015
#   endif
    //potentials = potentials.claim();// LP_09.2015

    if (calctype == "Psi0") {
        do {
            setMatrix(calctype, A, dvnDeltaPsi, vconst);    // corr holds RHS now czy nie moze byc po prostu dvnDelta?
            solveMatrix(A, dvnDeltaPsi);

            totDel = addCorr(calctype, vconst); // max. update
            this->writelog(LOG_DETAIL, "Maximal update: %1%", totDel*mEx); // czy dla Fn i Fp tez bedzie mEx?
            savePsi();
            saveEc();
            saveEv();
            saveN();
            saveP();

            ++loopno;
        } while ((totDel > maxerrPsi0) && (loopno < loops));
        outPotential.fireChanged();
    }
    else if (calctype == "Psi") {
        do {
            setMatrix(calctype, A, dvnDeltaPsi, vconst);    // corr holds RHS now czy nie moze byc po prostu dvnDelta?
            solveMatrix(A, dvnDeltaPsi);

            totDel = addCorr(calctype, vconst); // max. update
            this->writelog(LOG_DETAIL, "Maximal update: %1%", totDel*mEx); // czy dla Fn i Fp tez bedzie mEx?
            savePsi();
            saveEc();
            saveEv();
            saveN();
            saveP();

            ++loopno;
        } while ((totDel > maxerrPsi) && (loopno < loops));
        outPotential.fireChanged();
    }
    else if (calctype == "Fn") {
        do {
            setMatrix(calctype, A, dvnDeltaFn, vconst);    // corr holds RHS now czy nie moze byc po prostu dvnDelta?
            solveMatrix(A, dvnDeltaFn);

            totDel = addCorr(calctype, vconst); // max. update
            this->writelog(LOG_DETAIL, "Maximal update: %1%", totDel*mEx); // czy dla Fn i Fp tez bedzie mEx?
            saveFn();
            saveFnEta();
            saveN();

            ++loopno;
        } while ((totDel > maxerrFn) && (loopno < loops));
        outQuasiFermiEnergyLevelForElectrons.fireChanged();
    }
    else if (calctype == "Fp") {
        do {
            setMatrix(calctype, A, dvnDeltaFp, vconst);    // corr holds RHS now czy nie moze byc po prostu dvnDelta?
            solveMatrix(A, dvnDeltaFp);

            totDel = addCorr(calctype, vconst); // max. update
            this->writelog(LOG_DETAIL, "Maximal update: %1%", totDel*mEx); // czy dla Fn i Fp tez bedzie mEx?
            saveFp();
            saveFpKsi();
            saveP();

            ++loopno;
        } while ((totDel > maxerrFp) && (loopno < loops));
        outQuasiFermiEnergyLevelForHoles.fireChanged();
    }

  //if ((loop != 0 || mcur >= minj) && err > toterr) toterr = err;
    //this->writelog(LOG_RESULT, "Loop %d(%d): max(j%s) = %g kA/cm2, error = %g%%",
    //               loop, loopno, noactive?"":"@junc", mcur, err);

    //while (err > maxerr && (loops == 0 || loop < loops)); wczeniej dla femV


    /*err = 0.;
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
    mcur = sqrt(mcur);
    err = 100. * sqrt(err) / mcur;*/


    /*saveConductivities();

    outCurrentDensity.fireChanged();
    outHeat.fireChanged();
*/ // LP_09.2015

    return totDel;
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::solveMatrix(DpbMatrix& A, DataVector<double>& B)
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

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::solveMatrix(DgbMatrix& A, DataVector<double>& B)
{
    int info = 0;
    this->writelog(LOG_DETAIL, "Solving matrix system");
    aligned_unique_ptr<int> ipiv(aligned_malloc<int>(A.size));

    A.mirror();

    // Factorize matrix
    dgbtrf(A.size, A.size, A.kd, A.kd, A.data, A.ld+1, ipiv.get(), info);
    if (info < 0) {
        throw CriticalException("%1%: Argument %2% of dgbtrf has illegal value", this->getId(), -info);
    } else if (info > 0) {
        throw ComputationError(this->getId(), "Matrix is singlar (at %1%)", info);
    }

    // Find solutions
    dgbtrs('N', A.size, A.kd, A.kd, 1, A.data, A.ld+1, ipiv.get(), B.data(), B.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dgbtrs has illegal value", this->getId(), -info);

    // now A contains factorized matrix and B the solutions
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::solveMatrix(SparseBandMatrix& ioA, DataVector<double>& B)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    PrecondJacobi precond(ioA);

    //DataVector<double> x = dvnPsi.copy(); // We use previous potentials as initial solution // LP_09.2015
    DataVector<double> x(B.size(), 0.); // We use 0 as initial solution for corrections // LP_09.2015
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

/*
template<typename Geometry2DType> //LP_09.2015
void DriftDiffusionModel2DSolver<Geometry2DType>::saveHeatDensities()
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
                double dvx = 0.;//0.5e6 * (- potentials[loleftno] + potentials[lorghtno] - potentials[upleftno] + potentials[uprghtno]) // LP_09.2015
                               //     / (e.getUpper0() - e.getLower0()); // [grad(dV)] = V/m
                double dvy = 0.; //0.5e6 * (- potentials[loleftno] - potentials[lorghtno] + potentials[upleftno] + potentials[uprghtno]) // LP_09.2015
                              //      / (e.getUpper1() - e.getLower1()); // [grad(dV)] = V/m
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
            double dvx = 0.;//0.5e6 * (- potentials[loleftno] + potentials[lorghtno] - potentials[upleftno] + potentials[uprghtno])// LP_09.2015
                                // (e.getUpper0() - e.getLower0()); // [grad(dV)] = V/m
            double dvy = 0.; //0.5e6 * (- potentials[loleftno] - potentials[lorghtno] + potentials[upleftno] + potentials[uprghtno])// LP_09.2015
                               // (e.getUpper1() - e.getLower1()); // [grad(dV)] = V/m
            auto midpoint = e.getMidpoint();
            if (size_t nact = isActive(midpoint)) {
                const auto& act = active[nact];
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
*/

/*template<> double DriftDiffusionModel2DSolver<Geometry2DCartesian>::integrateCurrent(size_t vindex, bool onlyactive)// LP_09.2015
{
    if (!potentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis0->size()-1; ++i) {
        auto element = mesh->elements(i, vindex);
        if (!onlyactive || isActive(element.getMidpoint()))
            result += currents[element.getIndex()].c1 * element.getSize0();
    }
    if (this->getGeometry()->isSymmetric(Geometry::DIRECTION_TRAN)) result *= 2.;
    return result * geometry->getExtrusion()->getLength() * 0.01; // kA/cm² µm² -->  mA;
}


template<> double DriftDiffusionModel2DSolver<Geometry2DCylindrical>::integrateCurrent(size_t vindex, bool onlyactive)
{
    if (!potentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis0->size()-1; ++i) {
        auto element = mesh->elements(i, vindex);
        if (!onlyactive || isActive(element.getMidpoint())) {
            double rin = element.getLower0(), rout = element.getUpper0();
            result += currents[element.getIndex()].c1 * (rout*rout - rin*rin);
        }
    }
    return result * M_PI * 0.01; // kA/cm² µm² -->  mA
}


template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::getTotalCurrent(size_t nact)
{
    if (nact >= active.size()) throw BadInput(this->getId(), "Wrong active region number");
    const auto& act = active[nact];
    // Find the average of the active region
    size_t level = (act.bottom + act.top) / 2;
    return integrateCurrent(level, true);
}
*/

template<typename Geometry2DType>
const LazyData<double> DriftDiffusionModel2DSolver<Geometry2DType>::getPotentials(shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) const
{
    if (!dvnPsi) throw NoValue("Potential");
    this->writelog(LOG_DEBUG, "Getting potentials");
    if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh, dvnPsi*mEx, dst_mesh, method, this->geometry); // here the potential is rescalled (*mEx)
}


template<typename Geometry2DType>
const LazyData<double> DriftDiffusionModel2DSolver<Geometry2DType>::getQuasiFermiEnergyLevelsForElectrons(shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) const
{
    if (!dvnFn) throw NoValue("Quasi-Fermi electron level");
    this->writelog(LOG_DEBUG, "Getting quasi-Fermi electron level");
    if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh, dvnFn*mEx, dst_mesh, method, this->geometry); // here the quasi-Fermi electron level is rescalled (*mEx)
}


template<typename Geometry2DType>
const LazyData<double> DriftDiffusionModel2DSolver<Geometry2DType>::getQuasiFermiEnergyLevelsForHoles(shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) const
{
    if (!dvnFp) throw NoValue("Quasi-Fermi hole level");
    this->writelog(LOG_DEBUG, "Getting quasi-Fermi hole level");
    if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh, dvnFp*mEx, dst_mesh, method, this->geometry); // here the quasi-Fermi hole level is rescalled (*mEx)
}


template<typename Geometry2DType>
const LazyData<double> DriftDiffusionModel2DSolver<Geometry2DType>::getConductionBandEdges(shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) const
{
    if (!dvnEc) throw NoValue("Conduction band edge");
    this->writelog(LOG_DEBUG, "Getting conduction band edge");
    if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh, dvnEc*mEx, dst_mesh, method, this->geometry); // here the conduction band edge is rescalled (*mEx)
}


template<typename Geometry2DType>
const LazyData<double> DriftDiffusionModel2DSolver<Geometry2DType>::getValenceBandEdges(shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) const
{
    if (!dvnEv) throw NoValue("Valence band edge");
    this->writelog(LOG_DEBUG, "Getting valence band edge");
    if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh, dvnEv*mEx, dst_mesh, method, this->geometry); // here the valence band edge is rescalled (*mEx)
}

/*
template<typename Geometry2DType>
const LazyData<Vec<2>> DriftDiffusionModel2DSolver<Geometry2DType>::getCurrentDensities(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method)
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
const LazyData<double> DriftDiffusionModel2DSolver<Geometry2DType>::getHeatDensities(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method)
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
const LazyData<Tensor2<double>> DriftDiffusionModel2DSolver<Geometry2DType>::getConductivity(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod) {
    this->initCalculation();
    this->writelog(LOG_DEBUG, "Getting conductivities");
    loadConductivities();
    InterpolationFlags flags(this->geometry);
    return LazyData<Tensor2<double>>(new LazyDataDelegateImpl<Tensor2<double>>(dest_mesh->size(),
        [this, dest_mesh, flags](size_t i) -> Tensor2<double> {
            auto point = flags.wrap(dest_mesh->at(i));
            size_t x = std::upper_bound(this->mesh->axis0->begin(), this->mesh->axis0->end(), point[0]) - this->mesh->axis0->begin();
            size_t y = std::upper_bound(this->mesh->axis1->begin(), this->mesh->axis1->end(), point[1]) - this->mesh->axis1->begin();
            if (x == 0 || y == 0 || x == this->mesh->axis0->size() || y == this->mesh->axis1->size())
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
    return 2.*M_PI * 0.5e-18 * phys::epsilon0 * W; // 1e-18 µm³ -> m³
}


template<typename Geometry2DType>
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
    return 2e-15*M_PI * W; // 1e-15 µm³ -> m³, W -> mW
}*/

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::findPsiI(double iEc0, double iEv0, double iNc, double iNv, double iNd, double iNa, double iEd, double iEa, double iFnEta, double iFpKsi, double iT, int& loop)
{
    double tPsi0(0.), // calculated normalized initial potential
    tPsi0a = (-15.)/mEx, // normalized edge of the initial range
    tPsi0b = (15.)/mEx, // normalized edge of the initial range
    tPsi0h = (0.1)/mEx, // normalized step in the initial range calculations
    tN = 0., tP = 0., // normalized electron/hole concentrations
    tNtot, tNtota = (-1e30)/mNx, tNtotb = (1e30)/mNx; // normalized carrier concentration and its initial values for potentials at range edges

    // initial calculations

    int tPsi0n = static_cast<int>(round((tPsi0b-tPsi0a)/tPsi0h)) + 1 ; // number of initial normalized potential values

    std::vector<double> tPsi0v(tPsi0n); // normalized potential values to check
    for (int i=0; i<tPsi0n; ++i)
        tPsi0v[i] = tPsi0a + i*tPsi0h;

    for (int i=0; i<tPsi0n; ++i) {
        tN = calcN(iNc, iFnEta, tPsi0v[i], iEc0, iT);
        tP = calcP(iNv, iFpKsi, tPsi0v[i], iEv0, iT);

        double iNdIon = iNd;
        double iNaIon = iNa;

        if (!mFullIon)
        {
            double gD(2.), gA(4.);
            double iNdTmp = (iNc/gD)*exp(-iEd);
            double iNaTmp = (iNv/gA)*exp(-iEa);
            iNdIon = iNd * (iNdTmp/(iNdTmp+tN));
            iNaIon = iNa * (iNaTmp/(iNaTmp+tP));
        }

        tNtot = tP - tN + iNdIon - iNaIon; // total normalized carrier concentration

        if (tNtot<0) {
            if (tNtot>tNtota) {
                tNtota = tNtot;
                tPsi0b = tPsi0v[i];
            }
        }
        else if (tNtot>0) {
            if (tNtot<tNtotb) {
                tNtotb = tNtot;
                tPsi0a = tPsi0v[i];
            }
        }
        else // found initial normalised potential
            return tPsi0v[i];
    }

    // precise calculations

    double tPsiUpd = 1e30, // normalised potential update
        tTmpA, tTmpB; // temporary data

        int tL=0; // loop counter
        while ((std::abs(tPsiUpd) > (maxerrPsiI)/mEx) && (tL < iterlimPsiI)) {
            tTmpA = (tNtotb-tNtota) / (tPsi0b-tPsi0a);
            tTmpB = tNtota - tTmpA*tPsi0a;
            tPsi0 = - tTmpB/tTmpA; //Psi Check Value
            tN = calcN(iNc, iFnEta, tPsi0, iEc0, iT);
            tP = calcP(iNv, iFpKsi, tPsi0, iEv0, iT);

            double iNdIon = iNd;
            double iNaIon = iNa;

            if (!mFullIon) {
                double gD(2.), gA(4.);
                double iNdTmp = (iNc/gD)*exp(-iEd);
                double iNaTmp = (iNv/gA)*exp(-iEa);
                iNdIon = iNd * (iNdTmp/(iNdTmp+tN));
                iNaIon = iNa * (iNaTmp/(iNaTmp+tP));
            }

            tNtot = tP - tN + iNdIon - iNaIon; // total normalized carrier concentration

            if (tNtot<0) {
                tNtota = tNtot;
                tPsi0b = tPsi0;
            }
            else if (tNtot>0) {
                tNtotb = tNtot;
                tPsi0a = tPsi0;
            }
            else { // found initial normalized potential
                loop = tL;
                //this->writelog(LOG_DETAIL, "%1% loops done. Calculated energy level corresponding to the initial potential: %2% eV", tL, (tPsi0)*mEx); // TEST

                return tPsi0;
            }

            tPsiUpd = tPsi0b-tPsi0a;
            //TODO 2015
            #ifndef NDEBUG
                if (!tL)
                    this->writelog(LOG_DEBUG, "Initial potential correction: %1% eV", (tPsiUpd)*mEx); // TEST
                else
                    this->writelog(LOG_DEBUG, " %1% eV", (tPsiUpd)*mEx); // TEST
            #endif
            ++tL;
        }

        loop = tL;
        //this->writelog(LOG_INFO, "%1% loops done. Calculated energy level corresponding to the initial potential: %2% eV", tL, (tPsi0)*mEx); // TEST

        return tPsi0;
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::calcN(double iNc, double iFnEta, double iPsi, double iEc0, double iT)
{
    double yn(0.);
    if (stat == "MB") yn = 1.;
    else if (stat == "FD") yn = calcFD12(log(iFnEta)+iPsi-iEc0)/(iFnEta*exp(iPsi-iEc0));
    return ( iNc*iFnEta*yn*exp(iPsi-iEc0) );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::calcP(double iNv, double iFpKsi, double iPsi, double iEv0, double iT)
{
    double yp(0.);
    if (stat == "MB") yp = 1.;
    else if (stat == "FD") yp = calcFD12(log(iFpKsi)-iPsi+iEv0)/(iFpKsi*exp(-iPsi+iEv0));
    return ( iNv*iFpKsi*yp*exp(iEv0-iPsi) );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::calcFD12(double iEta)
{
    double tKsi = pow(iEta,4.) + 33.6*iEta*(1.-0.68*exp(-0.17*(iEta+1.)*(iEta+1.))) + 50.;
    return ( 0.5*sqrt(M_PI) / (0.75*sqrt(M_PI)*pow(tKsi,-0.375)+exp(-iEta)) );
}


template<> std::string DriftDiffusionModel2DSolver<Geometry2DCartesian>::getClassName() const { return "drift_diffusion.Shockley2D"; }
template<> std::string DriftDiffusionModel2DSolver<Geometry2DCylindrical>::getClassName() const { return "drift_diffusion.ShockleyCyl"; }

template struct PLASK_SOLVER_API DriftDiffusionModel2DSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API DriftDiffusionModel2DSolver<Geometry2DCylindrical>;

}}} // namespace plask::solvers::thermal
