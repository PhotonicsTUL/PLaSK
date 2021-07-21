#ifndef PLASK__SOLVERS__OPTICAL__FDTD_H
#define PLASK__SOLVERS__OPTICAL__FDTD_H

#include <meep.hpp>
#include <plask/plask.hpp>
#include "meep_data_wrappers.hpp"
#include "plask_material_function.hpp"

namespace plask { namespace solvers { namespace optical_fdtd {

class PLASK_SOLVER_API FDTDSolver : public plask::SolverOver<plask::Geometry2DCartesian> {
  public:
    struct FieldData {
        FDTDSolver* solver;
        DataVector<plask::Vec<3, dcomplex>> field;
        std::array<meep::component, 3> components;

        FieldData(FDTDSolver* solver, std::array<meep::component, 3> components)
            : solver(solver), field(solver->fieldSize()), components(components) {}
    };

    // MEEP components
    meep::grid_volume compBounds;               // Use meep::vol2d later
    shared_ptr<meep::structure> compStructure;  // Material parameters etc
    shared_ptr<meep::fields> compFields;        // Fields chunks
    std::vector<SourceMEEP> sourceContainer;    // Used for storing the information about the sources

    plask::shared_ptr<RectangularMesh2D> fieldMesh;  // Mesh used for the extracted field;
    int resolution;                                  // geometry resolution
    double layerPML;                                 // PML layer thickness
    double shiftPML;                                 // Value used for spacing of the PML later
    double courantFactor;                            // Used for recalculation in order to obtain proper time-step
    double wavelength;                               // Wavelength used for material properties
    double temperature;                              // Temperature used for material properites
    double dt;                                       // Single time-step in MEEP time units
    double elapsedTime;                              // Time elapsed in the simulation (in SI)
    const double baseTimeUnit;                       // One time unit in SI (fs)

    // Providers
    ProviderFor<LightH, Geometry2DCartesian>::Delegate outLightH;
    ProviderFor<LightE, Geometry2DCartesian>::Delegate outLightE;

    FDTDSolver(const std::string& name);
    ~FDTDSolver() = default;
    virtual std::string getClassName() const;

    void compute(double time);
    void step();

    int fieldSize() const;

    void chunkloop(meep::fields_chunk* fc,
                   int ichunk,
                   meep::component cgrid,
                   meep::ivec is,
                   meep::ivec ie,
                   meep::vec s0,
                   meep::vec s1,
                   meep::vec e0,
                   meep::vec e1,
                   double dV0,
                   double dV1,
                   meep::ivec shift,
                   std::complex<double> shift_phase,
                   const meep::symmetry& S,
                   int sn,
                   FieldData* chunkloop_data);
    static void solver_chunkloop(meep::fields_chunk* fc,
                                 int ichunk,
                                 meep::component cgrid,
                                 meep::ivec is,
                                 meep::ivec ie,
                                 meep::vec s0,
                                 meep::vec s1,
                                 meep::vec e0,
                                 meep::vec e1,
                                 double dV0,
                                 double dV1,
                                 meep::ivec shift,
                                 std::complex<double> shift_phase,
                                 const meep::symmetry& S,
                                 int sn,
                                 void* chunkloop_data);
    void outputHDF5(meep::component component);

    meep::component getXmlComponent(const XMLReader& reader, const char* attr) const;
    void loadConfiguration(XMLReader& reader, Manager& manager);

    double eps(const meep::vec& r);
    double conductivity(const meep::vec& r);

    // DFT Fluxes
    FluxDFT addFluxDFT(plask::Vec<2, double> p1, plask::Vec<2, double> p2, double wl_min, double wl_max, int Nlengths);

    // DFT Fields
    FieldsDFT addFieldDFT(const std::vector<meep::component>& fields,
                          plask::Vec<2, double> p1,
                          plask::Vec<2, double> p2,
                          double wl);

    // Harminv
    shared_ptr<HarminvResults> doHarminv(meep::component component,
                                         plask::Vec<2, double> point,
                                         double wavelength,
                                         double dl,
                                         double time = 200.,
                                         const unsigned int NBands = 100);

    // Setters
    void setGeometry();
    void setSource();
    void setTimestep();
    void setMesh();

  protected:
    void onInitialize() override;
    LazyData<plask::Vec<3, dcomplex>> getLightE(const plask::shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod int_method);
    LazyData<plask::Vec<3, dcomplex>> getLightH(const plask::shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod int_method);

  private:
    long long int timeToSteps(double time);
    double fsToTimeMEEP(double time);
};

}}}  // namespace plask::solvers::optical_fdtd

#endif  // PLASK__SOLVERS__OPTICAL_FDTD_H
