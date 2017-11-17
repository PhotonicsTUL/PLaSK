#include "simple_optical.h"

namespace plask { namespace solvers { namespace simple_optical {

  

SimpleOptical::SimpleOptical(const std::string& name):plask::SolverOver<plask::Geometry2DCylindrical>(name)
{

}

void SimpleOptical::loadConfiguration(XMLReader& reader, Manager& manager) {
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd()) {

            parseStandardConfiguration(reader, manager, "<geometry>");
    }
}

void SimpleOptical::say_hello()
{
    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(this->geometry->getChild());
    shared_ptr<RectangularMesh<2>> points = mesh->getMidpointsMesh();

    std::cout<<"Hello world!!!!!!!!2222222222222 "<<std::endl;

    std::cout << "mesh: "; for (double p: *mesh->vert()) std::cout << p << " "; std::cout << std::endl;
    std::cout << "material: "; for (double p: *points->vert()) std::cout << geometry->getMaterial(vec(0.5, p))->name() << " "; std::cout << std::endl;
    std::cout << "nr: "; for (double p: *points->vert()) std::cout << str(geometry->getMaterial(vec(0.5, p))->Nr(1300, 300)) << " "; std::cout << std::endl;
}

void SimpleOptical::simpleVerticalSolver()
{
    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(this->geometry->getChild());
    std::cout << "mesh: "; for (double p: *mesh->vert()) std::cout<< p << " "; std::cout << std::endl;
  
    size_t stripe = mesh->tran()->findIndex(stripex);
    std::cout<<"stripe: "<<stripe << std::endl;
  
    if (stripe < xbegin) stripe = xbegin;
	  else if (stripe >= xend) stripe = xend-1;
	  writelog(LOG_DETAIL, "Computing effective index for vertical stripe {0} (polarization {1})", stripe-xbegin, (polarization==TE)?"TE":"TM");
  
    Data2DLog<dcomplex,dcomplex> log_stripe(getId(), format("stripe[{0}]", stripe-xbegin), "neff", "det");
    auto rootdigger = plask::optical::simple_optical::RootDigger::get(this, [&](const dcomplex& x){return this->detS1(x,nrCache[stripe]);}, log_stripe, stripe_root);
    //if (vneff == 0.) {
    //       dcomplex maxn = *std::max_element(nrCache[stripe].begin(), nrCache[stripe].end(),
    //                                         [](const dcomplex& a, const dcomplex& b){return real(a) < real(b);} );
    //       vneff = 0.999 * real(maxn);
    //   }
    //writelog(LOG_DETAIL, "vneff = {0}", vneff);
    //vneff = rootdigger->find(vneff);
  
  
}

dcomplex SimpleOptical::detS1(const dcomplex& x, const std::vector< dcomplex, aligned_allocator< dcomplex > >& NR, bool save)
{
    if (save) yfields[ybegin] = Field(0., 1.);

    std::vector<dcomplex,aligned_allocator<dcomplex>> ky(yend);
    for (size_t i = ybegin; i < yend; ++i) {
        ky[i] = k0 * sqrt(NR[i]*NR[i] - x*x);
        if (imag(ky[i]) > 0.) ky[i] = -ky[i];
    }

    Matrix T = Matrix::eye();
    for (size_t i = ybegin; i < yend-1; ++i) {
        double d;
        if (i != ybegin || ybegin != 0) d = mesh->axis1->at(i) - mesh->axis1->at(i-1);
        else d = 0.;
        dcomplex phas = exp(- I * ky[i] * d);
        // Transfer through boundary
        dcomplex f = (polarization==TM)? (NR[i+1]/NR[i]) : 1.;
        dcomplex n = 0.5 * ky[i]/ky[i+1] * f*f;
        Matrix T1 = Matrix( (0.5+n), (0.5-n),
                            (0.5-n), (0.5+n) );
        T1.ff *= phas; T1.fb /= phas;
        T1.bf *= phas; T1.bb /= phas;
        T = T1 * T;
        if (save) {
            dcomplex F = T.fb, B = T.bb;    // Assume  F0 = 0  B0 = 1
            double aF = abs(F), aB = abs(B);
            // zero very small fields to avoid errors in plotting for long layers
            if (aF < 1e-8 * aB) F = 0.;
            if (aB < 1e-8 * aF) B = 0.;
            yfields[i+1] = Field(F, B);
        }
    }

    if (save) {
        yfields[yend-1].B = 0.;
#ifndef NDEBUG
        std::stringstream nrs; for (size_t i = ybegin; i < yend; ++i)
            nrs << "), (" << str(yfields[i].F) << ":" << str(yfields[i].B);
        writelog(LOG_DEBUG, "vertical fields = [{0}) ]", nrs.str().substr(2));
#endif
    }

    // return s1*s4 - s2*s3;

    return T.bb;    // F0 = 0    Bn = 0
}
  
}}}



