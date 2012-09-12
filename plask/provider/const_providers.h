#ifndef CONST_PROVIDERS_H
#define CONST_PROVIDERS_H

#include "../geometry/element.h"
#include "../data.h"
#include "../geometry/path.h"

namespace plask {

template <typename PropertyT, typename SpaceT> struct ProviderFor;

/**
 * Helper class which allow to easy implementation of providers which allows to define value in each geometry place pointed as geometry element.
 */
template <typename PropertyT, typename SpaceT>
struct ConstByPlaceProviderImpl: public ProviderFor<PropertyT, SpaceT> {

    enum  { DIMS = SpaceT::DIMS };
    typedef typename PropertyT::ValueType ValueType;

    struct Place {
        /// Element for which we specify the value
        weak_ptr<GeometryElementD<DIMS>> element;

        /// Hints specifying pointed element
        PathHints hints;

        Place(weak_ptr<GeometryElementD<DIMS>> element, const PathHints& hints=PathHints())
            : element(element), hints(hints) {}

        bool operator<(const Place& other) const {
            return (element < other.element) || ( !(other.element < element) && (hints < other.hints) );
        }
    };

  protected:

    /// Element for which coordinates we specify the values
    weak_ptr<const GeometryElementD<DIMS>> root_geometry;

    /// Values for places.
    std::map<Place, ValueType> values;


    /// Default value, provided for places where there is no other value
    ValueType default_value;


    ConstByPlaceProviderImpl(weak_ptr<const GeometryElementD<DIMS>> root=weak_ptr<const GeometryElementD<DIMS>>(), const ValueType& default_value=ValueType())
        : root_geometry(root), default_value(default_value) {}

    /**
     * Get values in places showed by @p dst_mesh.
     * @param dst_mesh mesh
     */
    DataVector<ValueType> get(const plask::MeshD<DIMS>& dst_mesh) const {
        shared_ptr<const GeometryElementD<DIMS>> geometry = root_geometry.lock();
        if (!geometry) return DataVector<ValueType>(dst_mesh.size(), default_value);

        DataVector<ValueType> result(dst_mesh.size());

        size_t i = 0;
        for (Vec<DIMS, double> point: dst_mesh) {
            bool assigned = false;
            for (auto& place: values) {
                auto element = place.first.element.lock();
                if (!element) continue;
                auto regions = geometry->getElementInThisCoordinates(element, place.first.hints);
                for (const auto& region: regions) {
                    if (region && region->includes(point)) {
                        result[i] = place.second;
                        assigned = true;
                        break;
                    }
                }
            }
            if (!assigned) result[i] = default_value;
            ++i;
        }

        return result;
    }

  public:

    /**
     * Set value for specified element
     * \param element element on which the value is to be set
     * \param value value to set
     */
    void setValueFor(const Place& place, ValueType value) {
        this->values[place] = value;
        this->fireChanged();
    }

    /**
     * Get value from specified element
     * \param element element on which the value is to be gotten
     * \param hints optional hints specifying particular element instantions
     */
    ValueType getValueFrom(const Place& place) const {
        auto item = this->values.find(place);
        if (item != this->values.end()) return item->second;
        return default_value;
    }

    /**
     * Remove value from specified element
     * \param element element on which the value is to be removed
     * \param hints optional hints specifying particular element instantions
     */
    void removeValueFrom(const Place& place) {
        auto item = this->values.find(place);
        if (item != this->values.end()) { this->values.erase(item); this->fireChanged(); }
    }

    /// Clear all the values
    void clear() {
        this->values.clear();
        this->fireChanged();
    }

    /// \return root geometry element
    weak_ptr<const GeometryElementD<SpaceT::DIMS>> getRoot() const { return this->root_geometry; }

    /**
     * Set root geometry
     * \param root new root geometry
     */
    void setRoot(weak_ptr<const GeometryElementD<SpaceT::DIMS>> root) {
        this->root_geometry = root;
        this->fireChanged();
    }
};

}   // namespace plask

#endif // CONST_PROVIDERS_H
