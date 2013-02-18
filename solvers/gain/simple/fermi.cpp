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
void FermiGainSolver<GeometryType>::loadConfiguration(XMLReader& reader, Manager& manager)
{
    // Load a configuration parameter from XML.
    // Below you have an example
    while (reader.requireTagOrEnd())
    {
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

    detectActiveRegions();

    //TODO

    outGain.fireChanged();
}


template <typename GeometryType>
void FermiGainSolver<GeometryType>::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
{
    //TODO (if needed)
}

//template <typename GeometryType>
//void FermiGainSolver<GeometryType>::compute()
//{
//    this->initCalculation(); // This must be called before any calculation!
//}




template <typename GeometryType>
void FermiGainSolver<GeometryType>::detectActiveRegions()
{
    shared_ptr<RectilinearMesh2D> mesh = makeGeometryGrid(this->geometry->getChild());
    shared_ptr<RectilinearMesh2D> points = mesh->getMidpointsMesh();

    size_t ileft = 0, iright = points->axis0.size();
    bool in_active = false;

    for (size_t r = 0; r < points->axis1.size(); ++r)
    {
        bool had_active = false; // indicates if we had active region in this layer
        shared_ptr<Material> layer_material;
        bool layer_QW = false;

        for (size_t c = 0; c < points->axis0.size(); ++c)
        { // In the (possible) active region
            auto point = points->at(c,r);
            auto tags = this->geometry->getRolesAt(point);
            bool active = tags.find("active") != tags.end();
            bool QW = tags.find("QW") != tags.end()/* || tags.find("QD") != tags.end()*/;

            if (QW && !active)
                throw Exception("%1%: All marked quantum wells must belong to marked active region.", this->getId());

            if (c < ileft)
            {
                if (active)
                    throw Exception("%1%: Left edge of the active region not aligned.", this->getId());
            }
            else if (c >= iright)
            {
                if (active)
                    throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
            }
            else
            {
                // Here we are inside potential active region
                if (active)
                {
                    if (!had_active)
                    {
                        if (!in_active)
                        { // active region is starting set-up new region info
                            regions.emplace_back(mesh->at(c,r));
                            ileft = c;
                        }
                        layer_material = this->geometry->getMaterial(point);
                        layer_QW = QW;
                    }
                    else
                    {
                        if (*layer_material != *this->geometry->getMaterial(point))
                            throw Exception("%1%: Non-uniform active region layer.", this->getId());
                        if (layer_QW != QW)
                            throw Exception("%1%: Quantum-well role of the active region layer not consistent.", this->getId());
                    }
                }
                else if (had_active)
                {
                    if (!in_active)
                    {
                        iright = c;
                        if (layer_QW)
                        { // quantum well is at the egde of the active region, add one row below it
                            if (r == 0)
                                throw Exception("%1%: Quantum-well at the edge of the structure.", this->getId());
                            auto bottom_material = this->geometry->getMaterial(points->at(ileft,r-1));
                            for (size_t cc = ileft; cc < iright; ++cc)
                                if (*this->geometry->getMaterial(points->at(cc,r-1)) != *bottom_material)
                                    throw Exception("%1%: Material below quantum well not uniform.", this->getId());
                            auto& region = regions.back();
                            double w = mesh->axis0[iright] - mesh->axis0[ileft];
                            double h = mesh->axis1[r] - mesh->axis1[r-1];
                            region.origin += Vec<2>(0., -h);
                            region.layers->push_back(make_shared<Block<2>>(Vec<2>(w, h), bottom_material));
                        }
                    }
                    else
                        throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
                }
                had_active |= active;
            }
        }
        in_active = had_active;

        // Now fill-in the layer info
        ActiveRegionInfo* region = regions.empty()? nullptr : &regions.back();
        if (region)
        {
            double h = mesh->axis1[r+1] - mesh->axis1[r];
            double w = mesh->axis0[iright] - mesh->axis0[ileft];
            if (in_active)
            {
                size_t n = region->layers->getChildrenCount();
                shared_ptr<Block<2>> last;
                if (n > 0) last = static_pointer_cast<Block<2>>(static_pointer_cast<Translation<2>>(region->layers->getChildNo(n-1))->getChild());
                assert(!last || last->size.c0 == w);
                if (last && layer_material == last->material && layer_QW == region->isQW(region->size()-1))
                {
                    last->setSize(w, last->size.c1 + h);
                }
                else
                {
                    auto layer = make_shared<Block<2>>(Vec<2>(w,h), layer_material);
                    if (layer_QW) layer->addRole("QW");
                    region->layers->push_back(layer);
                }
            }
            else
            {
                if (region->isQW(region->size()-1))
                { // top layer of the active region is quantum well, add the next layer
                    auto top_material = this->geometry->getMaterial(points->at(ileft,r));
                    for (size_t cc = ileft; cc < iright; ++cc)
                        if (*this->geometry->getMaterial(points->at(cc,r)) != *top_material)
                            throw Exception("%1%: Material above quantum well not uniform.", this->getId());
                    region->layers->push_back(make_shared<Block<2>>(Vec<2>(w,h), top_material));
                }
                ileft = 0;
                iright = points->axis0.size();
            }
        }
    }
    if (!regions.empty() && regions.back().isQW(regions.back().size()-1))
        throw Exception("%1%: Quantum-well at the edge of the structure.", this->getId());
}


template <typename GeometryType>
const DataVector<const double> FermiGainSolver<GeometryType>::getGain(const MeshD<2>& dst_mesh, double wavelength, InterpolationMethod)
{
    this->initCalculation(); // This must be called before any calculation!

    DataVector<const double> nOnMesh = inCarriersConcentration(dst_mesh); // carriers concentration on the mesh
    DataVector<const double> TOnMesh = inTemperature(dst_mesh); // temperature on the mesh
    DataVector<double> gainOnMesh(dst_mesh.size(), NAN);

    for (int act=0; act<regions.size(); act++)
    {
        for (int i=0; i<dst_mesh.size(); i++)
        {
            setParameters(wavelength, TOnMesh[i], nOnMesh[i], regions[act]);
            gainOnMesh[i] = gainModule.Get_gain_at(wavelength);
            writelog(LOG_DEBUG, "point = %1%", "gain = %2%", dst_mesh[i], gainOnMesh[i]);
        }
    }

    return gainOnMesh;
}


template <typename GeometryType>
void FermiGainSolver<GeometryType>::setParameters(double wavelength, double T, double n, ActiveRegionInfo active)
{
    /// Material data base
    plask::shared_ptr<plask::Material> QW_material, Bar_material;
    plask::Box2D QWBox, BarBox;

    for (int i=0; i<active.size(); i++)
    {
        if (active.isQW(i) == true)
        {
            if (!QW_material)
            {
                QW_material = active.getLayerMaterial(i);
                QWBox = active.getLayerBox(i);
            }
            else if (QW_material != active.getLayerMaterial(i))
            {
                throw Exception("%1%: Multiple quantum wells materials in active region.", this->getId());
                break;
            }

            if (!Bar_material)
            {
                Bar_material == active.getLayerMaterial(i-1);
                BarBox = active.getLayerBox(i-1);
            }
            else if (Bar_material != active.getLayerMaterial(i+1))
            {
                throw Exception("%1%: Multiple barriers materials in active region.", this->getId());
                break;
            }

        }
    }

    gainModule.Set_temperature(T);
    gainModule.Set_koncentr(n);

    gainModule.Set_refr_index(QW_material->nr(wavelength, T));

    gainModule.Set_electron_mass_in_plain(QW_material->Me(T,'G').c11);
    gainModule.Set_electron_mass_transverse(QW_material->Me(T, 'G').c00);
    gainModule.Set_heavy_hole_mass_in_plain(QW_material->Mhh(T, 'G').c11);
    gainModule.Set_heavy_hole_mass_transverse(QW_material->Mhh(T, 'G').c00);
    gainModule.Set_light_hole_mass_in_plain(QW_material->Mlh(T, 'G').c11);
    gainModule.Set_light_hole_mass_transverse(QW_material->Mlh(T, 'G').c00);

    gainModule.Set_electron_mass_in_barrier(Bar_material->Me(T, 'G').c11);
    gainModule.Set_heavy_hole_mass_in_barrier(Bar_material->Mhh(T, 'G').c11);
    gainModule.Set_light_hole_mass_in_barrier(Bar_material->Mlh(T, 'G').c11);
//    gainModule.Set_barrier_width(15);

    gainModule.Set_well_width(determineBoxWidth(QWBox));
    gainModule.Set_waveguide_width(determineBoxWidth(BarBox));

    gainModule.Set_split_off(QW_material->Dso(T));
    gainModule.Set_bandgap(QW_material->Eg(T, 'G'));
    gainModule.Set_conduction_depth(QW_material->CBO(T, 'G'));
    gainModule.Set_valence_depth(QW_material->CBO(T, 'G'));

    gainModule.Set_cond_waveguide_depth(0.1);
    gainModule.Set_vale_waveguide_depth(0.1);

    gainModule.Set_lifetime(0.5);
    gainModule.Set_momentum_matrix_element(10.0);
//    gainModule.Set_momentum_matrix_element(gainModule.element());

//    gainModule.Set_first_point(1.19);
//    gainModule.Set_last_point(1.4);
//    gainModule.Set_step(.001);
}


template<typename GeometryType>
double FermiGainSolver<GeometryType>::determineBoxWidth(plask::Box2D materialBox)
{
    return  materialBox.upper[1] - materialBox.lower[1];
}


template <> std::string FermiGainSolver<Geometry2DCartesian>::getClassName() const { return "gain.Fermi2D"; }
template <> std::string FermiGainSolver<Geometry2DCylindrical>::getClassName() const { return "gain.FermiCyl"; }

template struct FermiGainSolver<Geometry2DCartesian>;
template struct FermiGainSolver<Geometry2DCylindrical>;

}}} // namespace
