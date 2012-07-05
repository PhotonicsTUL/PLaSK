#include "geometry.h"

GeometryWrapper::~GeometryWrapper()
{
    if (this->wrappedElement)
        this->wrappedElement->changedDisconnectMethod(this, &GeometryWrapper::onWrappedChange);
}
