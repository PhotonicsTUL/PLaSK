#include "ddm2d.h"

namespace plask { namespace solvers { namespace drift_diffusion {

template<typename Geometry2DType>
DriftDiffusionModel2DSolver<Geometry2DType>::DriftDiffusionModel2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DType, RectangularMesh<2>>(name),
    //pcond(5.), //LP_09.2015
    //ncond(50.), //LP_09.2015
    mRsrh(false), //LP_09.2015
    mRrad(false), //LP_09.2015
    mRaug(false), //LP_09.2015
    mPol(false), //LP_09.2015
    mFullIon(true), //LP_09.2015
    mTx(300.), //LP_09.2015
    mEx(phys::kB_eV*mTx), //LP_09.2015
    mNx(1e18), //LP_09.2015
    mEpsRx(10.), //LP_09.2015
    mXx(sqrt((phys::epsilon0*phys::kB_J*mTx*mEpsRx)/(phys::qe*phys::qe*mNx))*1e3), //LP_09.2015
    mKx(100.), //LP_09.2015
    mMix(1000.), //LP_09.2015
    mRx(((phys::kB_J*mTx*mMix*mNx)/(phys::qe*mXx*mXx))*1e8), //LP_09.2015
    mJx(((phys::kB_J*mNx)*mTx*mMix/mXx)*10.), //LP_09.2015
    mtx(mNx/mRx), //LP_09.2015
    mBx(mRx/(mNx*mNx)), //LP_09.2015
    mCx(mRx/(mNx*mNx*mNx)), //LP_09.2015
    mHx(((mKx*mTx)/(mXx*mXx))*1e12), //LP_09.2015
    dU(0.002),
    maxDelPsi0(10.*dU/mEx), // normalised
    maxDelPsi(0.1*dU/mEx), // normalised
    mAccPsiI(1e-12), //LP_09.2015
    mLoopPsiI(20), //LP_09.2015
    mStat("MB"), //LP_09.2015
    loopno(0),
    //default_junction_conductivity(5.), // LP_09.2015
    maxerr(0.05),
    heatmet(HEAT_JOULES),
    outPotential(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getPotentials),
//     outPotential(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getPotentials),
//     outCurrentDensity(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getCurrentDensities),
//     outHeat(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getHeatDensities),
//     outConductivity(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getConductivity),
    algorithm(ALGORITHM_CHOLESKY),
    itererr(1e-8),
    iterlim(10000),
    logfreq(500)
{
    //js.assign(1, 1.), //LP_09.2015
    //beta.assign(1, NAN), //LP_09.2015
    onInvalidate();
    inTemperature = 300.;
    //junction_conductivity.reset(1, default_junction_conductivity); //LP_09.2015
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
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
            //js[0] = source.getAttribute<double>("js", js[0]); //LP_09.2015
            //beta[0] = source.getAttribute<double>("beta", beta[0]); //LP_09.2015
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
            //pcond = source.getAttribute<double>("pcond", pcond); //LP_09.2015
            //ncond = source.getAttribute<double>("ncond", ncond); //LP_09.2015
            source.requireTagEnd();
        }

        else
            this->parseStandardConfiguration(source, manager);
    }
}


template<typename Geometry2DType>
DriftDiffusionModel2DSolver<Geometry2DType>::~DriftDiffusionModel2DSolver() {
}

/*template<typename Geometry2DType> // LP_09.2015
void DriftDiffusionModel2DSolver<Geometry2DType>::setActiveRegions()
{
    if (!this->geometry || !this->mesh) {
        if (junction_conductivity.size() != 1) {
            double condy = 0.;
            for (auto cond: junction_conductivity) condy += cond;
            junction_conductivity.reset(1, condy / junction_conductivity.size());
        }
        return;
    }

    shared_ptr<RectangularMesh<2>> points = this->mesh->getMidpointsMesh();

    std::vector<typename Active::Temp> regions;

    for (size_t r = 0; r < points->axis1->size(); ++r) {
        size_t prev = 0;
        for (size_t c = 0; c < points->axis0->size(); ++c) { // In the (possible) active region
            auto point = points->at(c,r);
            size_t num = isActive(point);

            if (num) { // here we are inside the active region
                regions.resize(max(regions.size(), num));
                auto& reg = regions[num-1];
                if (prev != num) { // this region starts in the current row
                    if (reg.top < r) {
                        throw Exception("%1%: Junction %2% is disjoint", this->getId(), num-1);
                    }
                    if (reg.bottom >= r) reg.bottom = r; // first row
                    else if (reg.rowr <= c) throw Exception("%1%: Junction %2% is disjoint", this->getId(), num-1);
                    reg.top = r + 1;
                    reg.rowl = c; if (reg.left > reg.rowl) reg.left = reg.rowl;
                }
            }
            if (prev && prev != num) { // previous region ended
                auto& reg = regions[prev-1];
                if (reg.bottom < r && reg.rowl >= c) throw Exception("%1%: Junction %2% is disjoint", this->getId(), prev-1);
                reg.rowr = c; if (reg.right < reg.rowr) reg.right = reg.rowr;
            }
            prev = num;
        }
        if (prev) // junction reached the edge
            regions[prev-1].rowr = regions[prev-1].right = points->axis0->size();
    }

    size_t condsize = 0;
    active.clear();
    active.reserve(regions.size());
    size_t i = 0;
    for (auto& reg: regions) {
        if (reg.bottom == size_t(-1)) reg.bottom = reg.top = 0;
        active.emplace_back(condsize, reg.left, reg.right, reg.bottom, reg.top, this->mesh->axis1->at(reg.top) - this->mesh->axis1->at(reg.bottom));
        condsize += reg.right - reg.left;
        this->writelog(LOG_DETAIL, "Detected junction %1% thickness = %2%nm", i++, 1e3 * active.back().height);
        this->writelog(LOG_DEBUG, "Junction %1% span: [%2%,%4%]-[%3%,%5%]", i-1, reg.left, reg.right, reg.bottom, reg.top);
    }

    if (junction_conductivity.size() != condsize) {
        double condy = 0.;
        for (auto cond: junction_conductivity) condy += cond;
        junction_conductivity.reset(max(condsize, size_t(1)), condy / junction_conductivity.size());
    }
}*/


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    loopno = 0;
    size = this->mesh->size();
    //potentials.reset(size, 0.); //LP_09.2015

    dvnNodeInfo.reset(size,0.); // LP 09.2015
    dvnPsi0.reset(size, 0.); //LP_09.2015
    dvnPsi.reset(size, 0.); //LP_09.2015
    dvnDeltaPsi.reset(size, 0.); //LP_09.2015
    dvnFn.reset(size, 0.); //LP_09.2015
    dvnFp.reset(size, 0.); //LP_09.2015

    dvePsiI.reset(this->mesh->elements.size(), 0.); //LP_09.2015
    dvePsi.reset(this->mesh->elements.size(), 0.); //LP_09.2015
    dveFnEta.reset(this->mesh->elements.size(), 1.); //LP_09.2015
    dveFpKsi.reset(this->mesh->elements.size(), 1.); //LP_09.2015
    dveN.reset(this->mesh->elements.size(), 0.); //LP_09.2015
    dveP.reset(this->mesh->elements.size(), 0.); //LP_09.2015

    currents.reset(this->mesh->elements.size(), vec(0.,0.));
    //conds.reset(this->mesh->elements.size()); //LP_09.2015
    /*if (junction_conductivity.size() == 1) {
        size_t condsize = 0;
        for (const auto& act: active) condsize += act.right - act.left;
        condsize = max(condsize, size_t(1));
        junction_conductivity.reset(condsize, junction_conductivity[0]);
    }*/ //LP_09.2015
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::onInvalidate() {
    //conds.reset(); //LP_09.2015
    dvnNodeInfo.reset();
    dvnPsi0.reset(); //LP_09.2015
    dvnPsi.reset(); //LP_09.2015
    dvnDeltaPsi.reset(); //LP_09.2015
    dvnFn.reset(); //LP_09.2015
    dvnFp.reset(); //LP_09.2015
    dvePsiI.reset(); //LP_09.2015
    dvePsi.reset(); //LP_09.2015
    dveFnEta.reset(); //LP_09.2015
    dveFpKsi.reset(); //LP_09.2015
    dveN.reset(); //LP_09.2015
    dveP.reset(); //LP_09.2015
    //potentials.reset(); //LP_09.2015
    currents.reset();
    heats.reset();
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

    // Update junction conductivities // LP_09.2015
    /*if (loopno != 0) {
        for (auto e: this->mesh->elements) {
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
            }
        }
    }*/

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
        //double Ec0 = material->CB(T, 0., 'G')/mEx; // normalised value
        //double Ev0 = material->VB(T, 0., 'G')/mEx; // normalised value
        double Nc = material->Nc(T, 0., 'G')/mNx; // normalised value
        double Nv = material->Nv(T, 0., 'G')/mNx; // normalised value
        //double Ni = material->Ni(T)/mNx; // normalised value
        //double Ne = Nc * exp(dvePsi[i]-Ec0); // normalised value
        //double Nh = Nv * exp(-dvePsi[i]+Ev0); // normalised value
        double eps = material->eps(T)/mEpsRx; // normalised value
        double Nd = material->Nd()/mNx; // normalised value
        double Na = material->Na()/mNx; // normalised value
        double Ed = 0.050/mEx; // normalised value
        double Ea = 0.150/mEx; // normalised value

        /*double yn(0.);
        if (mStat == "MB") yn = 1.;
        else if (mStat == "FD") yn = calcFD12(log(dveFnEta[i])+dvePsi[i]-Ec0)/(dveFnEta[i]*exp(dvePsi[i]-Ec0));

        double yp(0.);
        if (mStat == "MB") yp = 1.;
        else if (mStat == "FD") yp = calcFD12(log(dveFpKsi[i])-dvePsi[i]+Ev0)/(dveFpKsi[i]*exp(-dvePsi[i]+Ev0));*/

        double kk(0.), kx(0.), ky(0.), gg(0.), ff(0.);

        if (calctype == "Psi") {
            kk = 1. / (3.*(hx*0.5)*(hy*0.5));
            ky = eps * (hy*0.5) * (hy*0.5);
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

        // set symmetric matrix components
        double k44, k33, k22, k11, k43, k21, k42, k31, k32, k41;
        double g44, g33, g22, g11, g43, g21, g42, g31, g32, g41;
        double f1, f2, f3, f4;

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

        B[loleftno] += -(k11*dvnPsi[loleftno] + k21*dvnPsi[lorghtno] + k31*dvnPsi[uprghtno] + k41*dvnPsi[upleftno] + f1);
        B[lorghtno] += -(k21*dvnPsi[loleftno] + k22*dvnPsi[lorghtno] + k32*dvnPsi[uprghtno] + k42*dvnPsi[upleftno] + f2);
        B[uprghtno] += -(k31*dvnPsi[loleftno] + k32*dvnPsi[lorghtno] + k33*dvnPsi[uprghtno] + k43*dvnPsi[upleftno] + f3);
        B[upleftno] += -(k41*dvnPsi[loleftno] + k42*dvnPsi[lorghtno] + k43*dvnPsi[uprghtno] + k44*dvnPsi[upleftno] + f4);

        /*double kx = conds[i].c00;
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
        A(upleftno, uprghtno) += k43;*/
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
void DriftDiffusionModel2DSolver<Geometry2DType>::loadConductivities()
{
    auto midmesh = (this->mesh)->getMidpointsMesh();
    auto temperature = inTemperature(midmesh);

    for (auto e: this->mesh->elements)
    {
        size_t i = e.getIndex();
        Vec<2,double> midpoint = e.getMidpoint();

        auto roles = this->geometry->getRolesAt(midpoint);
        /*if (size_t actn = isActive(midpoint)) { // LP_09.2015
            const auto& act = active[actn-1];
            conds[i] = Tensor2<double>(0., junction_conductivity[act.offset + e.getIndex0()]);
        } else if (roles.find("p-contact") != roles.end()) {
            conds[i] = Tensor2<double>(pcond, pcond);
        } else if (roles.find("n-contact") != roles.end()) {
            conds[i] = Tensor2<double>(ncond, ncond);
        } else*/
            conds[i] = this->geometry->getMaterial(midpoint)->cond(temperature[i]);
    }
}
/*
template<typename Geometry2DType> // LP_09.2015
void DriftDiffusionModel2DSolver<Geometry2DType>::saveConductivities()
{
    for (size_t n = 0; n < active.size(); ++n) {
        const auto& act = active[n];
        for (size_t i = act.left, r = (act.top + act.bottom)/2; i != act.right; ++i)
            junction_conductivity[act.offset + i] = conds[this->mesh->elements(i,r).getIndex()].c11;
    }
}*/



template<typename Geometry2DType> // LP_09.2015
void DriftDiffusionModel2DSolver<Geometry2DType>::savePsi()
{
    for (auto el: this->mesh->elements) {
        size_t i = el.getIndex();
        size_t loleftno = el.getLoLoIndex();
        size_t lorghtno = el.getUpLoIndex();
        size_t upleftno = el.getLoUpIndex();
        size_t uprghtno = el.getUpUpIndex(); //LP_09.2015

        dvePsi[i] = 0.25 * (dvnPsi[loleftno] + dvnPsi[lorghtno] + dvnPsi[upleftno] + dvnPsi[uprghtno]);
    }
}


template<typename Geometry2DType> // LP_09.2015
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

        /*auto roles = this->geometry->getRolesAt(midpoint);
        if (size_t actn = isActive(midpoint)) { // LP_09.2015
            const auto& act = active[actn-1];
            conds[i] = Tensor2<double>(0., junction_conductivity[act.offset + e.getIndex0()]);
        } else if (roles.find("p-contact") != roles.end()) {
            conds[i] = Tensor2<double>(pcond, pcond);
        } else if (roles.find("n-contact") != roles.end()) {
            conds[i] = Tensor2<double>(ncond, ncond);
        } else
            conds[i] = this->geometry->getMaterial(midpoint)->cond(temperature[i]);*/
    }
}


template<typename Geometry2DType> // LP_09.2015
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

        /*auto roles = this->geometry->getRolesAt(midpoint);
        if (size_t actn = isActive(midpoint)) { // LP_09.2015
            const auto& act = active[actn-1];
            conds[i] = Tensor2<double>(0., junction_conductivity[act.offset + e.getIndex0()]);
        } else if (roles.find("p-contact") != roles.end()) {
            conds[i] = Tensor2<double>(pcond, pcond);
        } else if (roles.find("n-contact") != roles.end()) {
            conds[i] = Tensor2<double>(ncond, ncond);
        } else
            conds[i] = this->geometry->getMaterial(midpoint)->cond(temperature[i]);*/
    }
}


template<typename Geometry2DType> // LP_09.2015
double DriftDiffusionModel2DSolver<Geometry2DType>::addCorr(std::string calctype)
{
    double maxRelDel(0.);
    if (calctype == "Psi") {
        for (int i = 0; i < this->mesh->size(); ++i) {
            if (dvnDeltaPsi[i] > maxDelPsi0) dvnDeltaPsi[i] = maxDelPsi0;
            else if (dvnDeltaPsi[i] < -maxDelPsi0) dvnDeltaPsi[i] = -maxDelPsi0;
            if (std::abs(dvnDeltaPsi[i]/dvnPsi[i]) > maxRelDel) maxRelDel = std::abs(dvnDeltaPsi[i]/dvnPsi[i]);
            dvnPsi[i] += dvnDeltaPsi[i];
        }
    }
    return maxRelDel;
}


template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::compute(std::string calctype, unsigned loops) {
    if (calctype=="PsiI") {
        computePsiI(loops);
    }
    else {
        switch (algorithm) {
            case ALGORITHM_CHOLESKY: return doCompute<DpbMatrix>(calctype, loops);
            case ALGORITHM_GAUSS: return doCompute<DgbMatrix>(calctype, loops);
            case ALGORITHM_ITERATIVE: return doCompute<SparseBandMatrix>(calctype, loops);
        }
    }
    return 0.;
}

template<typename Geometry2DType> //LP_09.2015
int DriftDiffusionModel2DSolver<Geometry2DType>::computePsiI(unsigned loops) {
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

        double T(300.); // TODO
        double normEc0 = material->CB(T, 0., 'G')/mEx; // TODO
        double normEv0 = material->VB(T, 0., 'G', 'h')/mEx; // TODO
        double normNc = material->Nc(T, 0., 'G')/mNx; // TODO
        double normNv = material->Nv(T, 0., 'G')/mNx; // TODO
        double normNd = material->Nd()/mNx; // TODO
        double normNa = material->Na()/mNx; // TODO
        double normEd = 0.050/mEx; // TODO
        double normEa = 0.150/mEx; // TODO
        double normT = T/mTx; // TODO

        //std::tie(kx,ky) = std::tuple<double,double>(material->thermk(temp, thickness[elem.getIndex()]));

        double tPsiI = findPsiI(normEc0, normEv0, normNc, normNv, normNd, normNa, normEd, normEa, 1., 1., normT);
        this->writelog(LOG_DETAIL, "Calculated initial potential PsiI = %1% eV for element %2%", tPsiI*mEx, i); // LP_09.2015 // TEST
        dvePsiI[i] = tPsiI;
        oldmatname = newmatname;
        oldNd = newNd;
        oldNa = newNa;
        oldPsiI = tPsiI;
    }

    setPsiI();
    savePsi();
    saveN();
    saveP();

    //this->writelog(LOG_RESULT, "Loop %d(%d): max(j%s) = %g kA/cm2, error = %g%%",
    //               loop, loopno, noactive?"":"@junc", mcur, err);

    outPotential.fireChanged();

    return 0;
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
template <typename MatrixT>
double DriftDiffusionModel2DSolver<Geometry2DType>::doCompute(std::string calctype, unsigned loops)
{
    //this->initCalculation(); // LP_09.2015

    //heats.reset(); // LP_09.2015

    // Store boundary conditions for current mesh
    auto vconst = voltage_boundary(this->mesh, this->geometry);

    this->writelog(LOG_INFO, "Running drift_diffusion calculations");

    //unsigned loop = 0;LP_09.2015

    MatrixT A(size, this->mesh->minorAxis()->size());

    //double err = 0.;LP_09.2015
    toterr = 0.;

#   ifndef NDEBUG
        //if (!dvPsi.unique()) this->writelog(LOG_DEBUG, "Potential data held by something else...");// LP_09.2015
#   endif
    //potentials = potentials.claim();// LP_09.2015

    //loadConductivities();

    //bool noactive = (active.size() == 0);LP_09.2015
    /*double minj = js[0]; // assume no significant heating below this current// LP_09.2015
    for (auto j: js) if (j < minj) minj = j;
    minj *= 100e-7;

if (v_new.size() == 0) {
    v_new = DataVector<double>(N, 0.);
    v_old = DataVector<double>(N);
}
    
    do {
        setMatrix(A, potentials, vconst);    // potentials holds RHS now
        solveMatrix(A, potentials);

std::swap(v_new, v_old);
        
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
        mcur = sqrt(mcur);
        err = 100. * sqrt(err) / mcur;

        if ((loop != 0 || mcur >= minj) && err > toterr) toterr = err;

        ++loopno;
        ++loop;

        this->writelog(LOG_RESULT, "Loop %d(%d): max(j%s) = %g kA/cm2, error = %g%%",
                       loop, loopno, noactive?"":"@junc", mcur, err);

    } while (err > maxerr && (loops == 0 || loop < loops));

    saveConductivities();

    outPotential.fireChanged();
    outCurrentDensity.fireChanged();
    outHeat.fireChanged();
*/ // LP_09.2015

    for (int i(0); i<20; ++i) {
        setMatrix("Psi", A, dvnDeltaPsi, vconst);    // corr holds RHS now
        solveMatrix(A, dvnDeltaPsi);
        std::cout << "deltaPsi w zerze" << dvnDeltaPsi[0] << "\n";
        double maxRelDel = addCorr(calctype);
        this->writelog(LOG_RESULT, "maximal relative update: %1% V", maxRelDel);
        savePsi();
        saveN();
        saveP();
    }

    outPotential.fireChanged();

    return toterr;
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

/*
template<typename Geometry2DType>
const LazyData<double> DriftDiffusionModel2DSolver<Geometry2DType>::getPotentials(shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod method) const
{
    if (!potentials) throw NoValue("Potential");
    this->writelog(LOG_DEBUG, "Getting potentials");
    if (method == INTERPOLATION_DEFAULT)  method = INTERPOLATION_LINEAR;
    return interpolate(this->mesh, potentials, dst_mesh, method, this->geometry);
}


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
double DriftDiffusionModel2DSolver<Geometry2DType>::findPsiI(double iEc0, double iEv0, double iNc, double iNv, double iNd, double iNa, double iEd, double iEa, double iFnEta, double iFpKsi, double iT)
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
        while ((std::abs(tPsiUpd) > (mAccPsiI)/mEx) && (tL < mLoopPsiI)) {
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
                this->writelog(LOG_INFO, "%1% loops done. Calculated energy level corresponding to the initial potential: %2% eV", tL, (tPsi0)*mEx); // TEST

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

        this->writelog(LOG_INFO, "%1% loops done. Calculated energy level corresponding to the initial potential: %2% eV", tL, (tPsi0)*mEx); // TEST

        return tPsi0;
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::calcN(double iNc, double iFnEta, double iPsi, double iEc0, double iT)
{
    double yn(0.);
    if (mStat == "MB") yn = 1.;
    else if (mStat == "FD") yn = calcFD12(log(iFnEta)+iPsi-iEc0)/(iFnEta*exp(iPsi-iEc0));
    return ( iNc*iFnEta*yn*exp(iPsi-iEc0) );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::calcP(double iNv, double iFpKsi, double iPsi, double iEv0, double iT)
{
    double yp(0.);
    if (mStat == "MB") yp = 1.;
    else if (mStat == "FD") yp = calcFD12(log(iFpKsi)-iPsi+iEv0)/(iFpKsi*exp(-iPsi+iEv0));
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
