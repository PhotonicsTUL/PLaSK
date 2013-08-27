#include "fft.h"
#include <optical/slab/config.h>

namespace plask { namespace solvers { namespace slab {

#ifdef USE_FFTW
    
void performFFT(size_t howmany, size_t n, dcomplex* data);
    

void performSymFFT(size_t howmany, size_t n, dcomplex* data);


void performFFT(size_t howmany, size_t n1, size_t n2, dcomplex* data);


void performSymFFT(size_t howmany, size_t n1, size_t n2, dcomplex* data);



void performIFFT(size_t howmany, size_t n, dcomplex* data);
    

void performEvenIFFT(size_t howmany, size_t n, dcomplex* data);


void performOddIFFT(size_t howmany, size_t n, dcomplex* data);



void performIFFT(size_t howmany, size_t n1, size_t n2, dcomplex* data);
    

void performEvenIFFT(size_t howmany, size_t n1, size_t n2, dcomplex* data);


void performOddIFFT(size_t howmany, size_t n1, size_t n2, dcomplex* data);

#endif


}}} // namespace plask::solvers::slab
