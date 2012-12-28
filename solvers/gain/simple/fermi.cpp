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

    outGain->fireChanged();
}


template <typename GeometryType>
void FermiGainSolver<GeometryType>::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
{
    //TODO (if needed)
}


template <typename GeometryType>
void FermiGainSolver<GeometryType>::detectQuantumWells()
{
    shared_ptr<RectilinearMesh2D> mesh = RectilinearMesh2DSimpleGenerator()(this->geometry->getChild());
    shared_ptr<RectilinearMesh2D> points = mesh->getMidpointsMesh();

    std::vector<Box2D> results;

    shared_ptr<Material> QW_material; //TODO

    // Now compact each row (it can contain only one QW and each must start and end in the same point)
    double left = 0., right = 0.;
    bool foundQW = false;
    for (int j = 0; j < points->axis1.size(); ++j)
    {
        bool inQW = false;
        for (int i = 0; i < points->axis0.size(); ++i)
        {
            auto point = points->at(i,j);
            auto tags = this->geometry->getRolesAt(point);
            bool QW = tags.find("QW") != tags.end() || tags.find("QD") != tags.end();
            if (QW && !inQW)
            { // QW start
                if (foundQW)
                {
                    if (left != mesh->axis0[i])
                        throw Exception("This solver can only handle quantum wells of identical size located exactly one above another");
                    if (this->geometry->getMaterial(point) != QW_material)
                        throw Exception("In this solver all quantum wells must be constructed of a single material");
                }
                else
                {
                    QW_material = this->geometry->getMaterial(point);
                }
                left = mesh->axis0[i];
                inQW = true;
            }
            if (!QW && inQW)
            { // QW end
                if (foundQW && right != mesh->axis0[i])
                    throw Exception("This solver can only handle quantum wells of identical size located exactly one above another");
                right = mesh->axis0[i];
                results.push_back(Box2D(left, mesh->axis1[j], right, mesh->axis1[j+1]));
                foundQW = true;
                inQW = false;
            }
        }
        if (inQW)
        { // handle situation when QW spans to the end of the structure
            if (foundQW && right != mesh->axis0[points->axis0.size()])
                throw Exception("This solver can only handle quantum wells of identical size located exactly one above another");
            right = mesh->axis0[points->axis0.size()];
            results.push_back(Box2D(left, mesh->axis1[j], right, mesh->axis1[j+1]));
            foundQW = true;
            inQW = false;
        }
    }

    // Compact results in vertical direction
    //TODO

    //return results; TODO
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


template <>
std::string FermiGainSolver<Geometry2DCartesian>::getClassName() const { return "gain.Fermi2D"; }


}}} // namespace
