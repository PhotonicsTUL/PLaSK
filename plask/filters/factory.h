#ifndef PLASK__FILTERS_FACTORY_H
#define PLASK__FILTERS_FACTORY_H

#include "filter.h"

#include <map>
#include <string>
#include <functional>
#include "../utils/xml/reader.h"
#include "../manager.h"

namespace plask {

struct FiltersFactory
{

    typedef std::function<shared_ptr<Solver>(XMLReader& reader, Manager& manager)> FilterCreator;

private:
    /// property name -> constructor of filters for this property
    std::map<std::string, FilterCreator> filterCreators;

public:

    /**
     * Get default register of filters factories.
     * @return default register of filters factories
     */
    static FiltersFactory& getDefault();

    /**
     * Helper which calls getDefault().add(typeName, filterCreator) in constructor.
     *
     * Creating global objects of this type allow to fill the default filters database.
     */
    struct Register {
        Register(const std::string typeName, FilterCreator filterCreator) { getDefault().add(typeName, filterCreator); }
    };

    /**
     * Helper which calls getDefault().addStandard<PropertyTag>() in constructor.
     *
     * Creating global objects of this type allow to fill the default filters database.
     */
    template <typename PropertyTag>
    struct RegisterStandard {
        RegisterStandard() { getDefault().addStandard<PropertyTag>(); }
    };

    /// Standard filter factory.
    template <typename PropertyTag>
    static shared_ptr<Solver> standard(XMLReader& reader, Manager& manager);

    /**
     * Try to get filter from tag which is pointed by @p reader.
     *
     * Throw exception if reader points to filter tag, but this tag can't be parsed.
     * @param reader source of filter configuration, should point to begin of tag
     * @param manager
     * @return One of:
     * - filter - in such case reader point to end of filter tag,
     * - nullptr - if reader doesn't point to filter tag, in such case, reader is not changed.
     */
    shared_ptr<Solver> get(XMLReader& reader, Manager& manager);

    void add(const std::string typeName, FilterCreator filterCreator);

    template <typename PropertyTag>
    void addStandard() { add(type_name<PropertyTag>(), FiltersFactory::standard<PropertyTag>); }

};

template <typename PropertyTag>
inline shared_ptr<Solver> FiltersFactory::standard(XMLReader& reader, Manager& manager) {
    shared_ptr<GeometryObject> out = manager.requireGeometryObject(reader.requireAttribute("geometry"));
    reader.requireTagEnd();

    if (shared_ptr<Geometry3D> out_as_geom3D = dynamic_pointer_cast<Geometry3D>(out))
        return shared_ptr<Solver>(new Filter<PropertyTag, Geometry3D>(out_as_geom3D));

    if (shared_ptr<Geometry2DCartesian> out_as_geom2D = dynamic_pointer_cast<Geometry2DCartesian>(out))
        return shared_ptr<Solver>(new Filter<PropertyTag, Geometry2DCartesian>(out_as_geom2D));

    if (shared_ptr<Geometry2DCylindrical> out_as_geomCyl = dynamic_pointer_cast<Geometry2DCylindrical>(out))
        return shared_ptr<Solver>(new Filter<PropertyTag, Geometry2DCylindrical>(out_as_geomCyl));

    throw NotImplemented("standard filter (for given configuration), geometry must be of type: Geometry3D, Geometry2DCartesian or Geometry2DCylindrical");
}

}   // namespace plask

#endif // PLASK__FILTERS_FACTORY_H
