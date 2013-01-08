#include "fermi.h"

namespace plask { namespace solvers { namespace fermi {

template <typename GeometryType>
FermiGainSolver<GeometryType>::FermiGainSolver(const std::string& name): SolverOver<GeometryType>(name),
    inTemperature(this), inCarriersConcentration(this),
    outGain(this, &FermiGainSolver<GeometryType>::getGain) // getDelegated will be called whether provider value is requested
{
    inTemperature = 300.; // temperature receiver has some sensible value
}


template <typename GeometryType>
void FermiGainSolver<GeometryType>::loadConfiguration(XMLReader& reader, Manager& manager) {
    // Load a configuration parameter from XML.
    // Below you have an example
    while (reader.requireTagOrEnd()) {
//         std::string param = reader.getNodeName();
//         if (param == "newton") {
//             newton.tolx = reader.getAttribute<double>("tolx", newton.tolx);
//             newton.tolf = reader.getAttribute<double>("tolf", newton.tolf);
//             newton.maxstep = reader.getAttribute<double>("maxstep", newton.maxstep);
//             reader.requireTagEnd();
//         } else if (param == "wavelength") {
//             std::string = reader.requireTextUntilEnd();
//             inWavelength.setValue(boost::lexical_cast<double>(wavelength));
//         } else
//             parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <newton>, or <wavelength>");
    }
}


template <typename GeometryType>
void FermiGainSolver<GeometryType>::onInitialize() // In this function check if geometry and mesh are set
{
    if (!this->geometry) throw NoGeometryException(this->getId());

    //TODO

    outGain.fireChanged();
}


template <typename GeometryType>
void FermiGainSolver<GeometryType>::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
{
    //TODO (if needed)
}


template <typename GeometryType>
void FermiGainSolver<GeometryType>::detectActiveRegions()
{
    shared_ptr<RectilinearMesh2D> mesh = RectilinearMesh2DSimpleGenerator()(this->geometry->getChild());
    shared_ptr<RectilinearMesh2D> points = mesh->getMidpointsMesh();

    size_t ileft = 0, iright = points->axis0.size();
    shared_ptr<Material> prev_material;
    bool in_active = false;

    for (size_t r = 0; r < points->axis1.size(); ++r)
    {
        bool had_active = false; // indicates if we had active region in this layer
        shared_ptr<Material> layer_material;
        bool layer_QW = false;

        for (size_t c = 0; c < points->axis0.size(); ++c) { // In the (possible) active region
            auto point = points->at(c,r);
            auto tags = this->geometry->getRolesAt(point);
            bool active = tags.find("active") != tags.end();
            bool QW = tags.find("QW") != tags.end()/* || tags.find("QD") != tags.end()*/;

            if (QW && !active)
                throw Exception("%1%: All marked quantum wells must belong to marked active region.", this->getId());

            if (c < ileft) {
                if (active)
                    throw Exception("%1%: Left edge of the active region not aligned.", this->getId());
            } else if (c >= iright) {
                if (active)
                    throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
            } else {
                // Here we are inside potential active region
                if (active) {
                    if (!in_active) { // active region is starting set-up new region info
                        regions.emplace_back(mesh->at(r,c));
                        if (r > 0) regions.back().bottom = this->geometry->getMaterial(points->at(c,r-1));
                        ileft = c;
                    }
                    if (!had_active) {
                        layer_material = this->geometry->getMaterial(point);
                        layer_QW = QW;
                    } else {
                        if (layer_material != this->geometry->getMaterial(point))
                            throw Exception("%1%: Non-uniform active region layer.", this->getId());
                        if (layer_QW != QW)
                            throw Exception("%1%: Quantum-well role of the active region layer not consistent.", this->getId());
                    }
                } else {
                    if (!in_active)
                        iright = c;
                    else
                        throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
                }
                had_active |= active;
            }
        }
        in_active = had_active;

        // Now fill-in the layer info
        double h = mesh->axis1[r+1] - mesh->axis1[r];
        double w = mesh->axis1[iright] - mesh->axis1[ileft];
        ActiveRegionInfo* region = regions.empty()? nullptr : &regions.back();
        shared_ptr<Block<2>> last;
        if (region) {
            size_t n = region->layers->getChildrenCount();
            if (n > 0) last = dynamic_pointer_cast<Block<2>>(region->layers->getChildNo(n-1));
        }
        if (region && last && layer_material == last->material && layer_QW == region->areQW.back()) {
            assert(last->size.c0 == w);
            last->setSize(w, last->size.c1 + h);
        } else {
            region->layers->push_back(make_shared<Block<2>>(Vec<2>(w, h), layer_material));
            region->areQW.push_back(layer_QW);
        }
        if (!in_active && region && !region->top)
            region->top = layer_material;
    }
}


template <typename GeometryType>
double FermiGainSolver<GeometryType>::computeGain(const Vec<2>& point, double wavelenght)
{
    this->initCalculation(); // This must be called before any calculation!
}


template <typename GeometryType>
const DataVector<const double> FermiGainSolver<GeometryType>::getGain(const MeshD<2>& dst_mesh, double wavelength, InterpolationMethod) {

    //TODO


}


template <> std::string FermiGainSolver<Geometry2DCartesian>::getClassName() const { return "gain.Fermi2D"; }
template <> std::string FermiGainSolver<Geometry2DCylindrical>::getClassName() const { return "gain.FermiCyl"; }

template struct FermiGainSolver<Geometry2DCartesian>;
template struct FermiGainSolver<Geometry2DCylindrical>;

}}} // namespace
