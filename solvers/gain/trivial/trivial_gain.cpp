#include "trivial_gain.h"

namespace plask { namespace solvers { namespace gain_trivial {

template<> std::string StepProfileGain<Geometry2DCartesian>::getClassName() const { return "StepProfileGain2D"; }
template<> std::string StepProfileGain<Geometry2DCylindrical>::getClassName() const { return "StepProfileGainCyl"; }
template<> std::string StepProfileGain<Geometry3D>::getClassName() const { return "StepProfileGain3D"; }

template <typename GeometryT>
void StepProfileGain<GeometryT>::loadParam(const std::string& param, XMLReader& reader, Manager& manager)
{
    if (param == "element") {
        element = manager.requireGeometryElement<GeometryElementD<GeometryT::DIMS>>(reader.requireAttribute("ref"));
        auto path = reader.getAttribute("path");
        if (path) {
            hints = manager.requirePathHints(*path);
        }
    } else
        throw XMLUnexpectedElementException(reader, "<gometry> or <element>", param);
    reader.requireTagEnd();
}


template <typename GeometryT>
void StepProfileGain<GeometryT>::setElement(const weak_ptr<const GeometryElementD<GeometryT::DIMS>>& element, const PathHints& path)
{
    this->element = element;
    this->hints = path;
}


template <typename GeometryT>
const DataVector<double> StepProfileGain<GeometryT>::getGain(const plask::MeshD<GeometryT::DIMS>& dst_mesh, double wavelength, plask::InterpolationMethod method)
{
}


template class StepProfileGain<Geometry2DCartesian>;
template class StepProfileGain<Geometry2DCylindrical>;
template class StepProfileGain<Geometry3D>;

}}} // namespace
