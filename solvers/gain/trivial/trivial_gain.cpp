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
    this->outGain.fireChanged();
}


template <typename GeometryT>
const DataVector<double> StepProfileGain<GeometryT>::getGainProfile(const plask::MeshD<GeometryT::DIMS>& dst_mesh, double wavelength, plask::InterpolationMethod method)
{
    if (!this->geometry) throw NoGeometryException(this->getId());

    DataVector<double> result(dst_mesh.size());

    auto el = this->element.lock();

    if (el) {
        auto positions = this->geometry->getElementPositions(el, hints);
        size_t i = 0;
        for (auto point: dst_mesh) {
            result[i++] = NAN;
            for (auto pos: positions)
                if (el->includes(point-pos))
                    result[i-1] = this->gain;
        }
    } else {
        std::fill(result.begin(), result.end(), NAN);
    }

    return result;
}


template class StepProfileGain<Geometry2DCartesian>;
template class StepProfileGain<Geometry2DCylindrical>;
template class StepProfileGain<Geometry3D>;

}}} // namespace
