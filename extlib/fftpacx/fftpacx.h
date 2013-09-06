#include <complex>

extern "C" {

int cfft1i_(const int& n, double* wsave, const int& lensav, int& ier);
int cfft1f_(const int& n, const int& inc, std::complex<double>* c, const int& lenc, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int cfft1b_(const int& n, const int& inc, std::complex<double>* c, const int& lenc, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);

int cfft2i_(const int& l, const int& m, double* wsave, const int& lensav, int& ier);
int cfft2f_(const int& ldim, const int& l, const int& m, std::complex<double>* c, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int cfft2b_(const int& ldim, const int& l, const int& m, std::complex<double>* c, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);

int cfftmi_(const int& n, double* wsave, const int& lensav, int& ier);
int cfftmf_(const int& lot, const int& jump, const int& n, const int& inc, std::complex<double>* c, const int& lenc, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int cfftmb_(const int& lot, const int& jump, const int& n, const int& inc, std::complex<double>* c, const int& lenc, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);


int rfft1i_(const int& n, double* wsave, const int& lensav, int& ier);
int rfft1f_(const int& n, const int& inc, double* r, const int& lenr, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int rfft1b_(const int& n, const int& inc, double* r, const int& lenr, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);

int rfft2i_(const int& l, const int& m, double* wsave, const int& lensav, int& ier);
int rfft2f_(const int& ldim, const int& l, const int& m, double* r, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int rfft2b_(const int& ldim, const int& l, const int& m, double* r, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);

int rfftmi_(const int& n, double* wsave, const int& lensav, int& ier);
int rfftmf_(const int& lot, const int& jump, const int& n, const int& inc, double* r, const int& lenr, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int rfftmb_(const int& lot, const int& jump, const int& n, const int& inc, double* r, const int& lenr, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);


int cost1i_(const int& n, double* wsave, const int& lensav, int& ier);
int cost1f_(const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int cost1b_(const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);

int costmi_(const int& n, double* wsave, const int& lensav, int& ier);
int costmf_(const int& lot, const int& jump, const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int costmb_(const int& lot, const int& jump, const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);


int sint1i_(const int& n, double* wsave, const int& lensav, int& ier);
int sint1f_(const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int sint1b_(const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);

int sintmi_(const int& n, double* wsave, const int& lensav, int& ier);
int sintmf_(const int& lot, const int& jump, const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int sintmb_(const int& lot, const int& jump, const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);


int cosq1i_(const int& n, double* wsave, const int& lensav, int& ier);
int cosq1f_(const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int cosq1b_(const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);

int cosqmi_(const int& n, double* wsave, const int& lensav, int& ier);
int cosqmf_(const int& lot, const int& jump, const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int cosqmb_(const int& lot, const int& jump, const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);


int sinq1i_(const int& n, double* wsave, const int& lensav, int& ier);
int sinq1f_(const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int sinq1b_(const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);

int sinqmi_(const int& n, double* wsave, const int& lensav, int& ier);
int sinqmf_(const int& lot, const int& jump, const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);
int sinqmb_(const int& lot, const int& jump, const int& n, const int& inc, double* x, const int& lenx, double* wsave, const int& lensav, double* work, const int& lenwrk, int& ier);

} // extern "C"
