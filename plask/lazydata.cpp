#include "lazydata.h"

namespace plask {

#ifdef _MSC_VER // MSVC require this while MingW does not accept

#define TEMPLATE_CLASS_FOR_LAZY_DATA(...) \
    template class PLASK_API LazyData< __VA_ARGS__ >; \
    template struct PLASK_API LazyDataImpl< __VA_ARGS__ >;

TEMPLATE_CLASS_FOR_LAZY_DATA(Tensor3< complex<double> >)
TEMPLATE_CLASS_FOR_LAZY_DATA(Tensor2< double >)
TEMPLATE_CLASS_FOR_LAZY_DATA(Vec<3, complex<double>>)
TEMPLATE_CLASS_FOR_LAZY_DATA(Vec<3, double>)
TEMPLATE_CLASS_FOR_LAZY_DATA(Vec<2, double>)
TEMPLATE_CLASS_FOR_LAZY_DATA(double)

#endif

}   // namespace plask
