#ifndef CONST_PROVIDERS_H
#define CONST_PROVIDERS_H

#include "../geometry/element.h"
#include "../data.h"
#include "../geometry/path.h"

namespace plask {

/**
 * Helper class which allow to easy implementation of providers which allows to define value in each geometry place pointed as geometry element.
 */
template <int dims, typename ValueT>
struct ConstByPlaceProviderImpl {

    enum  { DIMS = dims };

    /// Element for which coordinates we specify the values
    weak_ptr<const GeometryElementD<DIMS>> rootGeometry;

    struct ValueForPlace {
        /// Element for which we specify the value
        weak_ptr<GeometryElementD<DIMS>> geometryElement;

        /// Hints specifying pointed element
        PathHints hints;

        /// Value
        ValueT value;

        ValueForPlace(weak_ptr<GeometryElementD<DIMS>> geometryElement, const PathHints& hints, ValueT value)
            : GeometryElement(geometryElement), hints(hints), value(value) {}
    };

    /// Values for places.
    std::vector<ValueForPlace> values;

    /// Default value, provided for places where there is no other value
    ValueT defaultValue;

    ConstByPlaceProviderImpl(const ValueT& defaultValue = ValueT(), weak_ptr<const GeometryElementD<DIMS>> rootGeometry = weak_ptr<const GeometryElementD<DIMS>>())
        : defaultValue(defaultValue), rootGeometry(rootGeometry) {}

    /**
     * Get values in places showed by @p dst_mesh.
     * @param dst_mesh mesh
     */
    DataVector<ValueT> get(const plask::MeshD<DIMS>& dst_mesh) const {
        shared_ptr<const GeometryElementD<DIMS>> geometry = rootGeometry.lock();
        if (geometry) return DataVector<ValueT>(dst_mesh.size(), defaultValue);

        DataVector<double> result(dst_mesh.size());

        size_t i = 0;
        for (Vec<DIMS, double> point: dst_mesh) {
            bool assigned = false;
            for (ValueForPlace& place: values) {
                auto place_geom = place.geometryElement.lock();
                if (!place_geom) continue;
                std::vector< shared_ptr< Translation<DIMS> > > regions = geometry->getElementInThisCoordinates(place_geom, place.hints);
                for (shared_ptr< Translation<DIMS> > region: regions) {
                    if (region && region->includes(point)) {
                        result[i] = place.value;
                        assigned = true;
                        break;
                    }
                }
                if (assigned) break;
            }
            if (!assigned) result[i] = defaultValue;
            ++i;
        }

        return result;
    }

};

}   // namespace plask

#endif // CONST_PROVIDERS_H
