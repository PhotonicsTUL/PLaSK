#ifndef PLASK__CONST_PROVIDERS_H
#define PLASK__CONST_PROVIDERS_H

/** @file
This file includes templates and base classes for providers of constant values.
*/

#include "../geometry/object.h"
#include "../data.h"
#include "../geometry/path.h"
#include <deque>

namespace plask {

template <typename PropertyT, typename SpaceT> struct ProviderFor;

/**
 * Helper class which allow to easy implementation of providers which allows to define value in each geometry place pointed as geometry object.
 */
template <typename PropertyT, typename SpaceT>
struct ConstByPlaceProviderImpl: public ProviderFor<PropertyT, SpaceT> {

    enum  { DIMS = SpaceT::DIMS };
    typedef typename PropertyT::ValueType ValueType;

    struct Place {
        /// Object for which we specify the value
        weak_ptr<GeometryObjectD<DIMS>> object;

        /// Hints specifying pointed object
        PathHints hints;

        /**
         * Create place
         * \param object geometry object of the place
         * \param hints path hints further specifying the place
         */
        Place(weak_ptr<GeometryObjectD<DIMS>> object, const PathHints& hints=PathHints())
            : object(object), hints(hints) {}

        /// Comparison operator for std::find
        inline bool operator==(const Place& other) const {
            return !(object < other.object || other.object < object ||
                     hints  < other.hints  || other.hints  < hints);
        }
    };

  protected:

    /// Object for which coordinates we specify the values
    weak_ptr<const GeometryObjectD<DIMS>> root_geometry;

    /// Values for places.
    std::deque<Place> places;
    std::deque<ValueType> values;


    /// Default value, provided for places where there is no other value
    ValueType default_value;


    ConstByPlaceProviderImpl(weak_ptr<const GeometryObjectD<DIMS>> root=weak_ptr<const GeometryObjectD<DIMS>>(), const ValueType& default_value=ValueType())
        : root_geometry(root), default_value(default_value) {}

    /**
     * Get values in places showed by @p dst_mesh.
     * @param dst_mesh mesh
     */
    DataVector<ValueType> get(const plask::MeshD<DIMS>& dst_mesh) const {
        shared_ptr<const GeometryObjectD<DIMS>> geometry = root_geometry.lock();
        if (!geometry) return DataVector<ValueType>(dst_mesh.size(), default_value);
        auto root = geometry->getChild();
        if (!root) throw DataVector<ValueType>(dst_mesh.size(), default_value);

        DataVector<ValueType> result(dst_mesh.size());

        size_t i = 0;
        for (Vec<DIMS, double> point: dst_mesh) {
            bool assigned = false;
            for (auto place = places.begin(); place != places.end(); ++place) {
                auto object = place->object.lock();
                if (!object) continue;
                if (root->objectIncludes(point, *object, place->hints)) {
                    result[i] = values[place-places.begin()];
                    assigned = true;
                    break;
                }
            }
            if (!assigned) result[i] = default_value;
            ++i;
        }

        return result;
    }

  public:

    /**
     * Set value for specified object
     * \param object object on which the value is to be set
     * \param value value to set
     */
    void setValueFor(const Place& place, ValueType value) {
        auto found = std::find(places.begin(), places.end(), place);
        if (found != places.end()) values[found-places.begin()] = value;
        else {
            places.push_front(place);
            values.push_front(value);
        }
        this->fireChanged();
    }

    /**
     * Get value from specified object
     * \param object object on which the value is to be gotten
     * \param hints optional hints specifying particular object instances
     */
    ValueType getValueFrom(const Place& place) const {
        auto found = std::find(places.begin(), places.end(), place);
        if (found != places.end()) return values[found-places.begin()];
        return default_value;
    }

    /**
     * Remove value from specified object
     * \param object object on which the value is to be removed
     * \param hints optional hints specifying particular object instances
     */
    void removeValueFrom(const Place& place) {
        auto found = std::find(places.begin(), places.end(), place);
        if (found != places.end()) { places.erase(found); values.erase(values.begin() + (found-places.begin())); this->fireChanged(); }
    }

    /// Clear all the values
    void clear() {
        places.clear();
        values.clear();
        this->fireChanged();
    }

    /// \return root geometry object
    weak_ptr<const GeometryObjectD<SpaceT::DIMS>> getRoot() const { return root_geometry; }

    /**
     * Set root geometry
     * \param root new root geometry
     */
    void setRoot(weak_ptr<const GeometryObjectD<SpaceT::DIMS>> root) {
        root_geometry = root;
        this->fireChanged();
    }
};

}   // namespace plask

#endif // CONST_PROVIDERS_H
