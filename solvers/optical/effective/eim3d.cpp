// #include <exception>
// #include "eim3d.h"
//
// using plask::dcomplex;
//
// namespace plask { namespace solvers { namespace effective {
//
// EffectiveIndex3D::EffectiveIndex3D(const std::string& name) :
//     SolverWithMesh<Geometry3D, RectangularMesh<3>>(name),
//     log_value(dataLog<dcomplex, dcomplex>("Neff", "Neff", "det")),
//     stripex(0.),
//     polarization(TE),
//     recompute_neffs(true),
//     emission(FRONT),
//     vneff(0.),
//     outNeff(this, &EffectiveIndex3D::getEffectiveIndex, &EffectiveIndex3D::nmodes),
//     outLightMagnitude(this, &EffectiveIndex3D::getLightMagnitude, &EffectiveIndex3D::nmodes),
//     outRefractiveIndex(this, &EffectiveIndex3D::getRefractiveIndex),
//     outHeat(this, &EffectiveIndex3D::getHeat),
//     k0(2e3*M_PI/980) {
//     inTemperature = 300.;
//     inGain = NAN;
//     root_tran.tolx = 1.0e-6;
//     root_tran.tolf_min = 1.0e-7;
//     root_tran.tolf_max = 2.0e-5;
//     root_tran.maxiter = 500;
//     root_tran.method = RootDigger::ROOT_MULLER;
//     root_vert.tolx = 1.0e-6;
//     root_vert.tolf_min = 1.0e-7;
//     root_vert.tolf_max = 1.0e-5;
//     root_vert.maxiter = 500;
//     root_vert.method = RootDigger::ROOT_MULLER;
//     inTemperature.changedConnectMethod(this, &EffectiveIndex3D::onInputChange);
//     inGain.changedConnectMethod(this, &EffectiveIndex3D::onInputChange);
// }
//
//
// void EffectiveIndex3D::loadConfiguration(XMLReader& reader, Manager& manager) {
//     while (reader.requireTagOrEnd()) {
//         std::string param = reader.getNodeName();
//         if (param == "mode") {
//             auto pol = reader.getAttribute("polarization");
//             if (pol) {
//                 if (*pol == "TE") polarization = TE;
//                 else if (*pol == "TM") polarization = TM;
//                 else throw BadInput(getId(), "Wrong polarization specification '{0}' in XML", *pol);
//             }
//             k0 = 2e3*M_PI / reader.getAttribute<double>("wavelength",  real(2e3*M_PI / k0));
//             stripex = reader.getAttribute<double>("vat", stripex);
//             vneff = reader.getAttribute<dcomplex>("vneff", vneff);
//             reader.requireTagEnd();
//         } else if (param == "root_tran") {
//             RootDigger::readRootDiggerConfig(reader, root_tran);
//         } else if (param == "stripe-root_tran") {
//             RootDigger::readRootDiggerConfig(reader, root_vert);
//         } else if (param == "mirrors") {
//             double R1 = reader.requireAttribute<double>("R1");
//             double R2 = reader.requireAttribute<double>("R2");
//             mirrors.reset(std::make_pair(R1,R2));
//             reader.requireTagEnd();
//         } else if (param == "mesh") {
//             auto name = reader.getAttribute("ref");
//             if (!name) name.reset(reader.requireTextInCurrentTag());
//             else reader.requireTagEnd();
//             auto found = manager.meshes.find(*name);
//             if (found != manager.meshes.end()) {
//                 auto mesh = dynamic_pointer_cast<RectangularMesh<3>>(found->second);
//                 if (mesh) this->setMesh(mesh);
//                 else throw BadInput(this->getId(), "Mesh '{0}' of wrong type", *name);
//             } else {
//                 auto found = manager.generators.find(*name);
//                 if (found != manager.generators.end()) {
//                     auto generator = dynamic_pointer_cast<MeshGeneratorD<3>>(found->second);
//                     if (generator) this->setMesh(generator);
//                     else throw BadInput(this->getId(), "Mesh generator '{0}' of wrong type", *name);
//                 } else
//                     throw BadInput(this->getId(), "Neither mesh nor mesh generator '{0}' found", *name);
//             }
//         } else
//             parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <mode>, <root-tran>, <root-vert>, or <outer>");
//     }
// }
//
//
// std::vector<dcomplex> EffectiveIndex3D::searchVNeffs(dcomplex neff1, dcomplex neff2, size_t resteps, size_t imsteps, dcomplex eps)
// {
//     updateCache();
//
//     size_t stripe = mesh->tran()->findIndex(stripex);
//     if (stripe < xbegin) stripe = xbegin;
//     else if (stripe >= xend) stripe = xend-1;
//
//     if (eps.imag() == 0.) eps.imag(eps.real());
//
//     if (real(eps) <= 0. || imag(eps) <= 0.)
//         throw BadInput(this->getId(), "Bad precision specified");
//
//     double re0 = real(neff1), im0 = imag(neff1);
//     double re1 = real(neff2), im1 = imag(neff2);
//     if (re0 > re1) std::swap(re0, re1);
//     if (im0 > im1) std::swap(im0, im1);
//
//     if (re0 == 0. && re1 == 0.) {
//         re0 = 1e30;
//         re1 = -1e30;
//         for (size_t i = ybegin; i != yend; ++i) {
//             dcomplex n = nrCache[stripe][i];
//             if (n.real() < re0) re0 = n.real();
//             if (n.real() > re1) re1 = n.real();
//         }
//     } else if (re0 == 0. || re1 == 0.)
//         throw BadInput(getId(), "Bad area to browse specified");
//     if (im0 == 0. && im1 == 0.) {
//         im0 = 1e30;
//         im1 = -1e30;
//         for (size_t i = ybegin; i != yend; ++i) {
//             dcomplex n = nrCache[stripe][i];
//             if (n.imag() < im0) im0 = n.imag();
//             if (n.imag() > im1) im1 = n.imag();
//         }
//     }
//     neff1 = dcomplex(re0,im0);
//     neff2 = dcomplex(re1,im1);
//
//     auto ranges = findZeros(this, [&](const dcomplex& z){return this->detS1(z,nrCache[stripe]);}, neff1, neff2, resteps, imsteps, eps);
//     std::vector<dcomplex> results; results.reserve(ranges.size());
//     for (auto zz: ranges) results.push_back(0.5 * (zz.first+zz.second));
//
//     if (maxLoglevel >= LOG_RESULT) {
//         if (results.size() != 0) {
//             Data2DLog<dcomplex,dcomplex> logger(getId(), format("stripe[{0}]", stripe-xbegin), "neff", "det");
//             std::string msg = "Found vertical effective indices at: ";
//             for (auto z: results) {
//                 msg += str(z) + ", ";
//                 logger(z, detS1(z,nrCache[stripe]));
//             }
//             writelog(LOG_RESULT, msg.substr(0, msg.length()-2));
//         } else
//             writelog(LOG_RESULT, "Did not find any vertical effective indices");
//     }
//
//     return results;
// }
//
//
// size_t EffectiveIndex3D::findMode(dcomplex neff, Symmetry symmetry)
// {
//     writelog(LOG_INFO, "Searching for the mode starting from Neff = {0}", str(neff));
//     stageOne();
//     Mode mode(this, symmetry);
//     mode.neff = RootDigger::get(this, [this,&mode](const dcomplex& x){return this->detS(x,mode);}, log_value, root_tran)->find(neff);
//     return insertMode(mode);
// }
//
//
// std::vector<size_t> EffectiveIndex3D::findModes(dcomplex neff1, dcomplex neff2, Symmetry symmetry, size_t resteps, size_t imsteps, dcomplex eps)
// {
//     stageOne();
//
//     if (eps.imag() == 0.) eps.imag(eps.real());
//
//     if (real(eps) <= 0. || imag(eps) <= 0.)
//         throw BadInput(this->getId(), "Bad precision specified");
//
//     double re0 = real(neff1), im0 = imag(neff1);
//     double re1 = real(neff2), im1 = imag(neff2);
//     if (re0 > re1) std::swap(re0, re1);
//     if (im0 > im1) std::swap(im0, im1);
//
//     if (re0 == 0. && re1 == 0.) {
//         re0 = 1e30;
//         re1 = -1e30;
//         for (size_t i = xbegin; i != xend; ++i) {
//             dcomplex n = sqrt(epsilons[i]);
//             if (n.real() < re0) re0 = n.real();
//             if (n.real() > re1) re1 = n.real();
//         }
//     } else if (re0 == 0. || re1 == 0.)
//         throw BadInput(getId(), "Bad area to browse specified");
//     if (im0 == 0. && im1 == 0.) {
//         im0 = 1e30;
//         im1 = -1e30;
//         for (size_t i = xbegin; i != xend; ++i) {
//             dcomplex n = sqrt(epsilons[i]);
//             if (n.imag() < im0) im0 = n.imag();
//             if (n.imag() > im1) im1 = n.imag();
//         }
//     }
//     neff1 = dcomplex(re0,im0);
//     neff2 = dcomplex(re1,im1);
//
//     Mode mode(this, symmetry);
//     auto results = findZeros(this, [this,&mode](dcomplex z){return this->detS(z,mode);}, neff1, neff2, resteps, imsteps, eps);
//
//     std::vector<size_t> idx(results.size());
//
//     if (results.size() != 0) {
//         Data2DLog<dcomplex,dcomplex> logger(getId(), "Neffs", "Neff", "det");
//         auto refine = RootDigger::get(this, [this,&mode](const dcomplex& v){return this->detS(v,mode);}, logger, root_tran);
//         std::string msg = "Found modes at: ";
//         for (auto zz: results) {
//             dcomplex z;
//             try {
//                 z = refine->find(0.5*(zz.first+zz.second));
//             } catch (ComputationError) {
//                 continue;
//             }
//             mode.neff = z;
//             idx.push_back(insertMode(mode));
//             msg += str(z) + ", ";
//         }
//         writelog(LOG_RESULT, msg.substr(0, msg.length()-2));
//     } else
//         writelog(LOG_RESULT, "Did not find any modes");
//
//     return idx;
// }
//
//
// size_t EffectiveIndex3D::setMode(dcomplex neff, Symmetry symmetry_tran, Symmetry symmetry_long)
// {
//     stageOne();
//     Mode mode(this, symmetry_tran);
//     mode.neff = neff;
//     double det = abs(detS(neff, mode));
//     if (det > root_tran.tolf_max)
//         writelog(LOG_WARNING, "Provided effective index does not correspond to any mode (det = {0})", det);
//     writelog(LOG_INFO, "Setting mode at {0}", str(neff));
//     return insertMode(mode);
// }
//
// void EffectiveIndex3D::onInitialize()
// {
//     if (!geometry) throw NoGeometryException(getId());
//
//     // Set default mesh
//     if (!mesh) setSimpleMesh();
//
//     xbegin = 0;
//     ybegin = 0;
//     xend = mesh->axis0->size() + 1;
//     yend = mesh->axis1->size() + 1;
//
//     if (geometry->isExtended(Geometry::DIRECTION_TRAN, false) &&
//         abs(mesh->axis0->at(0) - geometry->getChild()->getBoundingBox().lower.c0) < SMALL)
//         xbegin = 1;
//     if (geometry->isExtended(Geometry::DIRECTION_VERT, false) &&
//         abs(mesh->axis1->at(0) - geometry->getChild()->getBoundingBox().lower.c1) < SMALL)
//         ybegin = 1;
//     if (geometry->isExtended(Geometry::DIRECTION_TRAN, true) &&
//         abs(mesh->axis0->at(mesh->axis0->size()-1) - geometry->getChild()->getBoundingBox().upper.c0) < SMALL)
//         --xend;
//     if (geometry->isExtended(Geometry::DIRECTION_VERT, true) &&
//         abs(mesh->axis1->at(mesh->axis1->size()-1) - geometry->getChild()->getBoundingBox().upper.c1) < SMALL)
//         --yend;
//
//     // Assign space for refractive indices cache and stripe effective indices
//     nrCache.assign(xend, std::vector<dcomplex,aligned_allocator<dcomplex>>(yend));
//     epsilons.resize(xend);
//
//     yfields.resize(yend);
//
//     need_gain = false;
// }
//
//
// void EffectiveIndex3D::onInvalidate()
// {
//     if (!modes.empty()) writelog(LOG_DETAIL, "Clearing computed modes");
//     modes.clear();
//     outNeff.fireChanged();
//     outLightMagnitude.fireChanged();
//     recompute_neffs = true;
// }
//
// /// Here are the computations ///
//
// void EffectiveIndex3D::updateCache()
// {
//     bool fresh = initCalculation();
//
//     if (fresh || inTemperature.changed() || (need_gain && inGain.changed())) {
//         // we need to update something
//
//         if (geometry->isSymmetric(Geometry::DIRECTION_TRAN)) {
//             if (fresh) // Make sure we have only positive points
//                 for (auto x: *mesh->axis0) if (x < 0.) throw BadMesh(getId(), "for symmetric geometry no horizontal points can be negative");
//             if (mesh->axis0->at(0) == 0.) xbegin = 1;
//         }
//
//         if (!modes.empty()) writelog(LOG_DETAIL, "Clearing computed modes");
//         modes.clear();
//
//         double w = real(2e3*M_PI / k0);
//
//         shared_ptr<OrderedAxis> axis0, axis1;
//         {
//             shared_ptr<RectangularMesh<2>> midmesh = mesh->getMidpointsMesh();
//             axis0 = plask::make_shared<OrderedAxis>(*midmesh->axis0);
//             axis1 = plask::make_shared<OrderedAxis>(*midmesh->axis1);
//         }
//
//         if (xbegin == 0) {
//             if (geometry->isSymmetric(Geometry::DIRECTION_TRAN)) axis0->addPoint(0.5 * mesh->axis0->at(0));
//             else axis0->addPoint(mesh->axis0->at(0) - 2.*OrderedAxis::MIN_DISTANCE);
//         }
//         if (xend == mesh->axis0->size()+1)
//             axis0->addPoint(mesh->axis0->at(mesh->axis0->size()-1) + 2.*OrderedAxis::MIN_DISTANCE);
//         if (ybegin == 0)
//             axis1->addPoint(mesh->axis1->at(0) - 2.*OrderedAxis::MIN_DISTANCE);
//         if (yend == mesh->axis1->size()+1)
//             axis1->addPoint(mesh->axis1->at(mesh->axis1->size()-1) + 2.*OrderedAxis::MIN_DISTANCE);
//
//         writelog(LOG_DEBUG, "Updating refractive indices cache");
//         auto midmesh = plask::make_shared<RectangularMesh<2>>(axis0, axis1, mesh->getIterationOrder());
//         auto temp = inTemperature(midmesh);
//         bool have_gain = false;
//         LazyData<double> gain;
//
//         for (size_t ix = xbegin; ix < xend; ++ix) {
//             for (size_t iy = ybegin; iy < yend; ++iy) {
//                 size_t idx = midmesh->index(ix-xbegin, iy-ybegin);
//                 double T = temp[idx];
//                 auto point = midmesh->at(idx);
//                 auto roles = geometry->getRolesAt(point);
//                 if (roles.find("QW") == roles.end() && roles.find("QD") == roles.end() && roles.find("gain") == roles.end())
//                     nrCache[ix][iy] = geometry->getMaterial(point)->Nr(w, T);
//                 else {  // we ignore the material absorption as it should be considered in the gain already
//                     need_gain = true;
//                     if (!have_gain) {
//                         gain = inGain(midmesh, w);
//                         have_gain = true;
//                     }
//                     double g = gain[idx];
//                     nrCache[ix][iy] = dcomplex( real(geometry->getMaterial(point)->Nr(w, T)),
//                                                 w * g * (0.25e-7/M_PI) );
//                 }
//             }
//         }
//         recompute_neffs = true;
//     }
// }
//
//
// void EffectiveIndex3D::stageOne()
// {
//     updateCache();
//
//     if (recompute_neffs) {
//
//         // Compute effective index of the main stripe
//         size_t stripe = mesh->tran()->findIndex(stripex);
//         if (stripe < xbegin) stripe = xbegin;
//         else if (stripe >= xend) stripe = xend-1;
//         writelog(LOG_DETAIL, "Computing effective index for vertical stripe {0} (polarization {1})", stripe-xbegin, (polarization==TE)?"TE":"TM");
// #ifndef NDEBUG
//         {
//             std::stringstream nrs; for (ptrdiff_t j = yend-1; j >= ptrdiff_t(ybegin); --j) nrs << ", " << str(nrCache[stripe][j]);
//             writelog(LOG_DEBUG, "Nr[{0}] = [{1} ]", stripe-xbegin, nrs.str().substr(1));
//         }
// #endif
//         Data3DLog<dcomplex,dcomplex> log_stripe(getId(), format("stripe[{0}]", stripe-xbegin), "neff", "det");
//         auto rootdigger = RootDigger::get(this, [&](const dcomplex& x){return this->detS1(x,nrCache[stripe]);}, log_stripe, root_vert);
//         if (vneff == 0.) {
//             dcomplex maxn = *std::max_element(nrCache[stripe].begin(), nrCache[stripe].end(),
//                                               [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
//             vneff = 0.999 * real(maxn);
//         }
//         vneff = rootdigger->find(vneff);
//
//         // Compute field weights
//         computeWeights(stripe);
//
//         // Compute effective indices
//         for (size_t i = xbegin; i < xend; ++i) {
//             epsilons[i] = vneff*vneff;
//             for (size_t j = ybegin; j < yend; ++j) {
//                 epsilons[i] += yweights[j] * (nrCache[i][j]*nrCache[i][j] - nrCache[stripe][j]*nrCache[stripe][j]);
//             }
//         }
// #ifndef NDEBUG
//         {
//             std::stringstream nrs; for (size_t i = xbegin; i < xend; ++i) {
//                 dcomplex n = sqrt(epsilons[i]); if (abs(n.real()) < 1e-10) n.real(0.); if (abs(n.imag()) < 1e-10) n.imag(0.);
//                 nrs << ", " << str(n);
//             }
//             writelog(LOG_DEBUG, "horizontal neffs = [{0} ]", nrs.str().substr(1));
//         }
// #endif
//         double rmin=INFINITY, rmax=-INFINITY, imin=INFINITY, imax=-INFINITY;
//         for (size_t i = xbegin; i < xend; ++i) {
//             dcomplex n = sqrt(epsilons[i]);
//             if (real(n) < rmin) rmin = real(n);
//             if (real(n) > rmax) rmax = real(n);
//             if (imag(n) < imin) imin = imag(n);
//             if (imag(n) > imax) imax = imag(n);
//         }
//         writelog(LOG_DETAIL, "Effective index should be between {0} and {1}", str(dcomplex(rmin,imin)), str(dcomplex(rmax,imax)));
//         recompute_neffs = false;
//     }
// }
//
// dcomplex EffectiveIndex3D::detS1(const plask::dcomplex& x, const std::vector<dcomplex,aligned_allocator<dcomplex>>& NR, bool save)
// {
//     if (save) yfields[ybegin] = Field(0., 1.);
//
//     std::vector<dcomplex,aligned_allocator<dcomplex>> ky(yend);
//     for (size_t i = ybegin; i < yend; ++i) {
//         ky[i] = k0 * sqrt(NR[i]*NR[i] - x*x);
//         if (imag(ky[i]) > 0.) ky[i] = -ky[i];
//     }
//
//     // dcomplex s1 = 1., s2 = 0., s3 = 0., s4 = 1.; // matrix S
//     //
//     // dcomplex phas = 1.;
//     // if (ybegin != 0)
//     //     phas = exp(I * ky[ybegin] * (mesh->axis1->at(ybegin)-mesh->axis1->at(ybegin-1)));
//     //
//     // for (size_t i = ybegin+1; i < yend; ++i) {
//     //     // Compute shift inside one layer
//     //     s1 *= phas;
//     //     s3 *= phas * phas;
//     //     s4 *= phas;
//     //     // Compute matrix after boundary
//     //     dcomplex f = (polarization==TM)? NR[i-1]/NR[i] : 1.;
//     //     dcomplex p = 0.5 + 0.5 * ky[i] / ky[i-1] * f*f;
//     //     dcomplex m = 1.0 - p;
//     //     dcomplex chi = 1. / (p - m * s3);
//     //     // F0 = [ (-m*m + p*p)*chi*s1  m*s1*s4*chi + s2 ] [ F2 ]
//     //     // B2 = [ (-m + p*s3)*chi      s4*chi           ] [ B0 ]
//     //     s2 += s1*m*chi*s4;
//     //     s1 *= (p*p - m*m) * chi;
//     //     s3  = (p*s3-m) * chi;
//     //     s4 *= chi;
//     //     // Compute phase shift for the next step
//     //     if (i != mesh->axis1->size())
//     //         phas = exp(I * ky[i] * (mesh->axis1->at(i)-mesh->axis1->at(i-1)));
//     //
//     //     // Compute fields
//     //     if (save) {
//     //         dcomplex F = -s2/s1, B = (s1*s4-s2*s3)/s1;    // Assume  F0 = 0  B0 = 1
//     //         double aF = abs(F), aB = abs(B);
//     //         // zero very small fields to avoid errors in plotting for long layers
//     //         if (aF < 1e-8 * aB) F = 0.;
//     //         if (aB < 1e-8 * aF) B = 0.;
//     //         yfields[i] = Field(F, B);
//     //     }
//     // }
//
//     Matrix T = Matrix::eye();
//     for (size_t i = ybegin; i < yend-1; ++i) {
//         double d;
//         if (i != ybegin || ybegin != 0) d = mesh->axis1->at(i) - mesh->axis1->at(i-1);
//         else d = 0.;
//         dcomplex phas = exp(- I * ky[i] * d);
//         // Transfer through boundary
//         dcomplex f = (polarization==TM)? (NR[i+1]/NR[i]) : 1.;
//         dcomplex n = 0.5 * ky[i]/ky[i+1] * f*f;
//         Matrix T1 = Matrix( (0.5+n), (0.5-n),
//                             (0.5-n), (0.5+n) );
//         T1.ff *= phas; T1.fb /= phas;
//         T1.bf *= phas; T1.bb /= phas;
//         T = T1 * T;
//         if (save) {
//             dcomplex F = T.fb, B = T.bb;    // Assume  F0 = 0  B0 = 1
//             double aF = abs(F), aB = abs(B);
//             // zero very small fields to avoid errors in plotting for long layers
//             if (aF < 1e-8 * aB) F = 0.;
//             if (aB < 1e-8 * aF) B = 0.;
//             yfields[i+1] = Field(F, B);
//         }
//     }
//
//     if (save) {
//         yfields[yend-1].B = 0.;
// #ifndef NDEBUG
//         std::stringstream nrs; for (size_t i = ybegin; i < yend; ++i)
//             nrs << "), (" << str(yfields[i].F) << ":" << str(yfields[i].B);
//         writelog(LOG_DEBUG, "vertical fields = [{0}) ]", nrs.str().substr(2));
// #endif
//     }
//
//     // return s1*s4 - s2*s3;
//
//     return T.bb;    // F0 = 0    Bn = 0
// }
//
//
// void EffectiveIndex3D::computeWeights(size_t stripe)
// {
//     // Compute fields
//     detS1(vneff, nrCache[stripe], true);
//
//     yweights.resize(yend);
//     {
//         double ky = abs(imag(k0 * sqrt(nrCache[stripe][ybegin]*nrCache[stripe][ybegin] - vneff*vneff)));
//         yweights[ybegin] = abs2(yfields[ybegin].B) * 0.5 / ky;
//     }
//     {
//         double ky = abs(imag(k0 * sqrt(nrCache[stripe][yend-1]*nrCache[stripe][yend-1] - vneff*vneff)));
//         yweights[yend-1] = abs2(yfields[yend-1].F) * 0.5 / ky;
//     }
//     double sum = yweights[ybegin] + yweights[yend-1];
//
//     for (size_t i = ybegin+1; i < yend-1; ++i) {
//         double d = mesh->axis1->at(i)-mesh->axis1->at(i-1);
//         dcomplex ky = k0 * sqrt(nrCache[stripe][i]*nrCache[stripe][i] - vneff*vneff); if (imag(ky) > 0.) ky = -ky;
//         dcomplex w_ff, w_bb, w_fb, w_bf;
//         if (d != 0.) {
//             if (abs(imag(ky)) > SMALL) {
//                 dcomplex kk = ky - conj(ky);
//                 w_ff =   (exp(-I*d*kk) - 1.) / kk;
//                 w_bb = - (exp(+I*d*kk) - 1.) / kk;
//             } else
//                 w_ff = w_bb = dcomplex(0., -d);
//             if (abs(real(ky)) > SMALL) {
//                 dcomplex kk = ky + conj(ky);
//                 w_fb =   (exp(-I*d*kk) - 1.) / kk;
//                 w_bf = - (exp(+I*d*kk) - 1.) / kk;
//             } else
//                 w_ff = w_bb = dcomplex(0., -d);
//             dcomplex weight = yfields[i].F * conj(yfields[i].F) * w_ff +
//                               yfields[i].F * conj(yfields[i].B) * w_fb +
//                               yfields[i].B * conj(yfields[i].F) * w_bf +
//                               yfields[i].B * conj(yfields[i].B) * w_bb;
//             yweights[i] = -imag(weight);
//         } else
//             yweights[i] = 0.;
//         sum += yweights[i];
//     }
//
//     sum = 1. / sum;
//     double fact = sqrt(sum);
//     for (size_t i = ybegin; i < yend; ++i) {
//         yweights[i] *= sum;
//         yfields[i] *= fact;
//     }
// // #ifndef NDEBUG
// //     {
// //         std::stringstream nrs; for (size_t i = ybegin; i < yend; ++i) nrs << ", " << str(weights[i]);
// //         writelog(LOG_DEBUG, "vertical weights = [{0}) ]", nrs.str().substr(2));
// //     }
// // #endif
// }
//
// void EffectiveIndex3D::normalizeFields(Mode& mode, const std::vector<dcomplex,aligned_allocator<dcomplex>>& kx) {
//
//     size_t start;
//
//     double sum = mode.xweights[xend-1] = abs2(mode.xfields[xend-1].F) * 0.5 / abs(imag(kx[xend-1]));
//     if (mode.symmetry == SYMMETRY_NONE) {
//         sum += mode.xweights[0] = abs2(mode.xfields[xbegin].B) * 0.5 / abs(imag(kx[xbegin]));
//         start = xbegin+1;
//     } else {
//         start = xbegin;
//     }
//
//     for (size_t i = start; i < xend-1; ++i) {
//         double d = mesh->axis0->at(i) - ((i == 0)? 0. : mesh->axis0->at(i-1));
//         dcomplex w_ff, w_bb, w_fb, w_bf;
//         if (d != 0.) {
//             if (abs(imag(kx[i])) > SMALL) {
//                 dcomplex kk = kx[i] - conj(kx[i]);
//                 w_ff =   (exp(-I*d*kk) - 1.) / kk;
//                 w_bb = - (exp(+I*d*kk) - 1.) / kk;
//             } else
//                 w_ff = w_bb = dcomplex(0., -d);
//             if (abs(real(kx[i])) > SMALL) {
//                 dcomplex kk = kx[i] + conj(kx[i]);
//                 w_fb =   (exp(-I*d*kk) - 1.) / kk;
//                 w_bf = - (exp(+I*d*kk) - 1.) / kk;
//             } else
//                 w_ff = w_bb = dcomplex(0., -d);
//             mode.xweights[i] = - imag(mode.xfields[i].F * conj(mode.xfields[i].F) * w_ff +
//                                       mode.xfields[i].F * conj(mode.xfields[i].B) * w_fb +
//                                       mode.xfields[i].B * conj(mode.xfields[i].F) * w_bf +
//                                       mode.xfields[i].B * conj(mode.xfields[i].B) * w_bb
//                                      );
//             sum += mode.xweights[i];
//         }
//     }
//     if (mode.symmetry != SYMMETRY_NONE) sum *= 2.;
//
//     // Consider loss on the mirror
//     double R1, R2;
//     if (mirrors) {
//         std::tie(R1,R2) = *mirrors;
//     } else {
//         const double lambda = real(2e3*M_PI / k0);
//         const double n = real(mode.neff);
//         const double n1 = real(geometry->getFrontMaterial()->Nr(lambda, 300.)),
//                      n2 = real(geometry->getBackMaterial()->Nr(lambda, 300.));
//         R1 = abs((n-n1) / (n+n1));
//         R2 = abs((n-n2) / (n+n2));
//     }
//
//     if (emission == FRONT) {
//         if (R1 == 1.)
//             this->writelog(LOG_WARNING, "Mirror refletion on emission side equal to 1. Field will be infinite.");
//         sum *= (1. - R1);
//     } else {
//         if (R2 == 1.)
//             this->writelog(LOG_WARNING, "Mirror refletion on emission side equal to 1. Field will be infinite.");
//         sum *= (1. - R2);
//     }
//
//     double ff = 1e12 / sum; // 1e12 because intensity in W/m² and integral computed in µm
//     double f = sqrt(ff);
//
//     for (auto& val: mode.xfields) val *= f;
//     for (auto& val: mode.xweights) val *= ff;
// }
//
// double EffectiveIndex3D::getTotalAbsorption(const Mode& mode)
// {
//     double result = 0.;
//
//     for (size_t ix = 0; ix < xend; ++ix) {
//         for (size_t iy = ybegin; iy < yend; ++iy) {
//             double absp = - 2. * real(nrCache[ix][iy]) * imag(nrCache[ix][iy]);
//             result += absp * mode.xweights[ix] * yweights[iy]; // [dV] = µm³
//         }
//     }
//     if (mode.symmetry != SYMMETRY_NONE) result *= 2.;
//     result *= 1e-9 * real(k0) * mode.power; // 1e-9: µm³ / nm -> m², ½ is already hidden in mode.power
//     return result;
// }
//
//
// double EffectiveIndex3D::getTotalAbsorption(size_t num)
// {
//     if (modes.size() <= num) throw NoValue("absorption");
//
//     if (!modes[num].have_fields) detS(modes[num].neff, modes[num], true);
//
//     return getTotalAbsorption(modes[num]);
// }
//
// dcomplex EffectiveIndex3D::detS(const dcomplex& x, EffectiveIndex3D::Mode& mode, bool save)
// {
//     // Adjust for mirror losses
//     dcomplex neff2 = dcomplex(real(x), imag(x)-getMirrorLosses(x)); neff2 *= neff2;
//
//     std::vector<dcomplex,aligned_allocator<dcomplex>> kx(xend);
//     for (size_t i = xbegin; i < xend; ++i) {
//         kx[i] = k0 * sqrt(epsilons[i] - neff2);
//         if (imag(kx[i]) > 0.) kx[i] = -kx[i];
//     }
//
//     aligned_unique_ptr<Matrix[]> matrices;
//     if (save) matrices.reset(aligned_malloc<Matrix>(xend-1));
//
//     Matrix T = Matrix::eye();
//     for (size_t i = xbegin; i < xend-1; ++i) {
//         double d;
//         if (i != xbegin) d = mesh->axis0->at(i) - mesh->axis0->at(i-1);
//         else if (mode.symmetry != SYMMETRY_NONE) d = mesh->axis0->at(i);     // we have symmetry, so beginning of the transfer matrix is at the axis
//         else d = 0.;
//         dcomplex phas = exp(- I * kx[i] * d);
//         // Transfer through boundary
//         dcomplex f = (polarization==TE)? (epsilons[i+1]/epsilons[i]) : 1.;
//         dcomplex n = 0.5 * kx[i]/kx[i+1] * f;
//         Matrix T1 = Matrix( (0.5+n), (0.5-n),
//                             (0.5-n), (0.5+n) );
//         T1.ff *= phas; T1.fb /= phas;
//         T1.bf *= phas; T1.bb /= phas;
//         if (save) matrices[i] = T1;
//         T = T1 * T;
//     }
//
//     if (save) {
//         mode.neff = x;
//
//         mode.xfields[xend-1] = Field(1., 0.);
//         for (size_t i = xend-1; i != xbegin; --i) {
//             mode.xfields[i-1] = matrices[i-1].solve(mode.xfields[i]);
//         }
//         normalizeFields(mode, kx);
//         mode.have_fields = true;
// #ifndef NDEBUG
//         {
//             std::stringstream nrs; for (size_t i = xbegin; i < xend; ++i)
//                 nrs << "), (" << str(mode.xfields[i].F) << ":" << str(mode.xfields[i].B);
//             writelog(LOG_DEBUG, "horizontal fields = [{0}) ]", nrs.str().substr(2));
//         }
// #endif
//     }
//
//     if (mode.symmetry == SYMMETRY_POSITIVE) return T.bf + T.bb;      // B0 = F0   Bn = 0
//     else if (mode.symmetry == SYMMETRY_NEGATIVE) return T.bf - T.bb; // B0 = -F0  Bn = 0
//     else return T.bb;                                                // F0 = 0    Bn = 0
// }
//
//
// struct EffectiveIndex3D::LightMagnitudeDataBase: public LazyDataImpl<double>
// {
//     EffectiveIndex3D* solver;
//     int num;
//     std::vector<dcomplex,aligned_allocator<dcomplex>> kx, ky;
//     size_t stripe;
//
//     LightMagnitudeDataBase(EffectiveIndex3D* solver, int num):
//         solver(solver), num(num), kx(solver->xend), ky(solver->yend),
//         stripe(solver->mesh->tran()->findIndex(solver->stripex))
//     {
//         dcomplex neff = solver->modes[num].neff;
//         if (!solver->modes[num].have_fields) solver->detS(neff, solver->modes[num], true);
//
//         if (stripe < solver->xbegin) stripe = solver->xbegin;
//         else if (stripe >= solver->xend) stripe = solver->xend-1;
//
//         solver->writelog(LOG_INFO, "Computing field distribution for Neff = {0}", str(neff));
//
//         dcomplex neff2 = dcomplex(real(neff), imag(neff)-solver->getMirrorLosses(real(neff))); neff2 *= neff2;
//         for (size_t i = 0; i < solver->xend; ++i) {
//             kx[i] = solver->k0 * sqrt(solver->epsilons[i] - neff2);
//             if (imag(kx[i]) > 0.) kx[i] = -kx[i];
//         }
//
//         for (size_t i = solver->ybegin; i < solver->yend; ++i) {
//             ky[i] = solver->k0 * sqrt(solver->nrCache[stripe][i]*solver->nrCache[stripe][i] - solver->vneff*solver->vneff);
//             if (imag(ky[i]) > 0.) ky[i] = -ky[i];
//         }
//     }
// };
//
// struct EffectiveIndex3D::LightMagnitudeDataInefficient: public EffectiveIndex3D::LightMagnitudeDataBase
// {
//     shared_ptr<const MeshD<2>> dst_mesh;
//
//      LightMagnitudeDataInefficient(EffectiveIndex3D* solver, int num, const shared_ptr<const MeshD<2>>& dst_mesh):
//         LightMagnitudeDataBase(solver, num), dst_mesh(dst_mesh) {}
//
//     size_t size() const override { return dst_mesh->size(); }
//
//     double at(size_t idx) const override {
//         auto point = dst_mesh->at(idx);
//         double x = point.tran();
//         double y = point.vert();
//
//         bool negate = false;
//         if (x < 0. && solver->modes[num].symmetry != EffectiveIndex3D::SYMMETRY_NONE) {
//             x = -x; if (solver->modes[num].symmetry == EffectiveIndex3D::SYMMETRY_NEGATIVE) negate = true;
//         }
//         size_t ix = solver->mesh->tran()->findIndex(x);
//         if (ix >= solver->xend) ix = solver->xend-1;
//         if (ix < solver->xbegin) ix = solver->xbegin;
//         if (ix != 0) x -= solver->mesh->tran()->at(ix-1);
//         else if (solver->modes[num].symmetry == EffectiveIndex3D::SYMMETRY_NONE) x -= solver->mesh->tran()->at(0);
//         dcomplex phasx = exp(- I * kx[ix] * x);
//         dcomplex val = solver->modes[num].xfields[ix].F * phasx + solver->modes[num].xfields[ix].B / phasx;
//         if (negate) val = - val;
//
//         size_t iy = solver->mesh->vert()->findIndex(y);
//         if (iy >= solver->yend) iy = solver->yend-1;
//         if (iy < solver->ybegin) iy = solver->ybegin;
//         y -= solver->mesh->vert()->at(max(int(iy)-1, 0));
//         dcomplex phasy = exp(- I * ky[iy] * y);
//         val *= solver->yfields[iy].F * phasy + solver->yfields[iy].B / phasy;
//
//         return 1e-3 * solver->modes[num].power * abs2(val);
//     }
// };
//
// struct EffectiveIndex3D::LightMagnitudeDataEfficient: public EffectiveIndex3D::LightMagnitudeDataBase
// {
//     shared_ptr<const RectangularMesh<2>> rect_mesh;
//     std::vector<dcomplex,aligned_allocator<dcomplex>> valx, valy;
//
//     size_t size() const override { return rect_mesh->size(); }
//
//     LightMagnitudeDataEfficient(EffectiveIndex3D* solver, int num, const shared_ptr<const RectangularMesh<2>>& rect_mesh):
//         LightMagnitudeDataBase(solver, num), rect_mesh(rect_mesh), valx(rect_mesh->tran()->size()), valy(rect_mesh->vert()->size())
//     {
//         #pragma omp parallel
//         {
//             #pragma omp for nowait
//             for (size_t idx = 0; idx < rect_mesh->tran()->size(); ++idx) {
//                 double x = rect_mesh->tran()->at(idx);
//                 bool negate = false;
//                 if (x < 0. && solver->modes[num].symmetry != EffectiveIndex3D::SYMMETRY_NONE) {
//                     x = -x; if (solver->modes[num].symmetry == EffectiveIndex3D::SYMMETRY_NEGATIVE) negate = true;
//                 }
//                 size_t ix = solver->mesh->tran()->findIndex(x);
//                 if (ix >= solver->xend) ix = solver->xend-1;
//                 if (ix < solver->xbegin) ix = solver->xbegin;
//                 if (ix != 0) x -= solver->mesh->tran()->at(ix-1);
//                 else if (solver->modes[num].symmetry == EffectiveIndex3D::SYMMETRY_NONE) x -= solver->mesh->tran()->at(0);
//                 dcomplex phasx = exp(- I * kx[ix] * x);
//                 dcomplex val = solver->modes[num].xfields[ix].F * phasx + solver->modes[num].xfields[ix].B / phasx;
//                 if (negate) val = - val;
//                 valx[idx] = val;
//             }
//
//             #pragma omp for
//             for (size_t idy = 0; idy < rect_mesh->vert()->size(); ++idy) {
//                 double y = rect_mesh->vert()->at(idy);
//                 size_t iy = solver->mesh->vert()->findIndex(y);
//                 if (iy >= solver->yend) iy = solver->yend-1;
//                 if (iy < solver->ybegin) iy = solver->ybegin;
//                 y -= solver->mesh->vert()->at(max(int(iy)-1, 0));
//                 dcomplex phasy = exp(- I * ky[iy] * y);
//                 valy[idy] = solver->yfields[iy].F * phasy + solver->yfields[iy].B / phasy;
//             }
//         }
//         // Free no longer needed memory
//         kx.clear();
//         ky.clear();
//     }
//
//     double at(size_t idx) const override {
//         size_t i0 = rect_mesh->index0(idx);
//         size_t i1 = rect_mesh->index1(idx);
//         return 1e-3 * solver->modes[num].power * abs2(valx[i0] * valy[i1]);
//     }
//
//     DataVector<const double> getAll() const override {
//         const double power = 1e-3 * solver->modes[num].power; // 1e-3 mW->W
//         DataVector<double> results(rect_mesh->size());
//         if (rect_mesh->getIterationOrder() == RectangularMesh<2>::ORDER_10) {
//             #pragma omp parallel for
//             for (plask::openmp_size_t i1 = 0; i1 < rect_mesh->axis1->size(); ++i1) {
//                 double* data = results.data() + i1 * rect_mesh->axis0->size();
//                 for (size_t i0 = 0; i0 < rect_mesh->axis0->size(); ++i0) {
//                     dcomplex f = valx[i0] * valy[i1];
//                     data[i0] = power * abs2(f);
//                 }
//             }
//         } else {
//             #pragma omp parallel for
//             for (plask::openmp_size_t i0 = 0; i0 < rect_mesh->axis0->size(); ++i0) {
//                 double* data = results.data() + i0 * rect_mesh->axis1->size();
//                 for (size_t i1 = 0; i1 < rect_mesh->axis1->size(); ++i1) {
//                     dcomplex f = valx[i0] * valy[i1];
//                     data[i1] = power * abs2(f);
//                 }
//             }
//         }
//         return results;
//     }
// };
//
// const LazyData<double> EffectiveIndex3D::getLightMagnitude(int num, shared_ptr<const plask::MeshD<2>> dst_mesh, plask::InterpolationMethod)
// {
//     this->writelog(LOG_DEBUG, "Getting light intensity");
//
//     if (auto rect_mesh = dynamic_pointer_cast<const RectangularMesh<2>>(dst_mesh))
//         return LazyData<double>(new LightMagnitudeDataEfficient(this, num, rect_mesh));
//     else
//         return LazyData<double>(new LightMagnitudeDataInefficient(this, num, dst_mesh));
// }
//
//
// const LazyData<Tensor3<dcomplex>> EffectiveIndex3D::getRefractiveIndex(shared_ptr<const MeshD<2>> dst_mesh, InterpolationMethod) {
//     this->writelog(LOG_DEBUG, "Getting refractive indices");
//     updateCache();
//     auto target_mesh = WrappedMesh<2>(dst_mesh, this->geometry);
//     return LazyData<Tensor3<dcomplex>>(dst_mesh->size(),
//         [this, target_mesh](size_t i) -> Tensor3<dcomplex> {
//             auto point = target_mesh[i];
//             size_t ix = this->mesh->axis0->findIndex(point.c0); if (ix < this->xbegin) ix = this->xbegin;
//             size_t iy = this->mesh->axis1->findIndex(point.c1);
//             return Tensor3<dcomplex>(this->nrCache[ix][iy]);
//         });
// }
//
//
// struct EffectiveIndex3D::HeatDataImpl: public LazyDataImpl<double>
// {
//     EffectiveIndex3D* solver;
//     WrappedMesh<2> mat_mesh;
//     std::vector<LazyData<double>> EE;
//     dcomplex lam0;
//
//     HeatDataImpl(EffectiveIndex3D* solver, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method):
//         solver(solver), mat_mesh(dst_mesh, solver->geometry), EE(solver->modes.size()), lam0(2e3*M_PI / solver->k0)
//     {
//         for (size_t m = 0; m != solver->modes.size(); ++m)
//             EE[m] = solver->getLightMagnitude(m, dst_mesh, method);
//     }
//
//     size_t size() const override { return mat_mesh.size(); }
//
//     double at(size_t j) const override {
//         double result = 0.;
//         auto point = mat_mesh[j];
//         size_t ix = solver->mesh->axis0->findIndex(point.c0); if (ix < solver->xbegin) ix = solver->xbegin;
//         size_t iy = solver->mesh->axis1->findIndex(point.c1);
//         for (size_t m = 0; m != solver->modes.size(); ++m) { // we sum heats from all modes
//             result += EE[m][j]; // 1e9: 1/nm -> 1/m
//         }
//         double absp = - 2. * real(solver->nrCache[ix][iy]) * imag(solver->nrCache[ix][iy]);
//         result *= 1e6 * real(solver->k0) * absp;
//         return result;
//     }
// };
//
// const LazyData<double> EffectiveIndex3D::getHeat(shared_ptr<const MeshD<2>> dst_mesh, plask::InterpolationMethod method)
// {
//     // This is somehow naive implementation using the field value from the mesh points. The heat may be slightly off
//     // in case of fast varying light intensity and too sparse mesh.
//     writelog(LOG_DEBUG, "Getting heat absorbed from {0} mode{1}", modes.size(), (modes.size()==1)? "" : "s");
//     if (modes.size() == 0) return LazyData<double>(dst_mesh->size(), 0.);
//     return LazyData<double>(new HeatDataImpl(this, dst_mesh, method));
// }
//
// }}} // namespace plask::solvers::effective
