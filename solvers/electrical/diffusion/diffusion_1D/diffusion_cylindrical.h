#include <plask/plask.hpp>

namespace plask { namespace electrical { namespace diffusion_cylindrical {

template<typename Geometry2DType>
class PLASK_SOLVER_API FiniteElementMethodDiffusion2DSolver: public SolverWithMesh<Geometry2DType, RegularMesh1D>
{
    public:
        enum FemMethod {
            FEM_LINEAR,
            FEM_PARABOLIC
        };

        enum ComputationType {
            COMPUTATION_INITIAL,
            COMPUTATION_THRESHOLD,
            COMPUTATION_OVERTHRESHOLD
        };

        ReceiverFor<CurrentDensity, Geometry2DType> inCurrentDensity;
        ReceiverFor<Temperature, Geometry2DType> inTemperature;
        ReceiverFor<Gain, Geometry2DType> inGain;
        ReceiverFor<Wavelength> inWavelength;
        ReceiverFor<LightE, Geometry2DType> inLightE;

        typename ProviderFor<CarriersConcentration, Geometry2DType>::Delegate outCarriersConcentration;

        InterpolationMethod interpolation_method;   ///< Selected interpolation method
        double relative_accuracy;                   ///< Relative accuracy
        int max_mesh_changes;                       ///< Maximum number of mesh refinemenst
        int max_iterations;                         ///< Maximum number of diffusion iterations for sigle mesh size
        FemMethod fem_method;                       ///< Finite element method (linear or parabolic)
        double minor_concentration;
        bool do_initial;                            ///< Should we start from initial computations

        FiniteElementMethodDiffusion2DSolver<Geometry2DType>(const std::string& name=""):
            SolverWithMesh<Geometry2DType,RegularMesh1D>(name),
            outCarriersConcentration(this, &FiniteElementMethodDiffusion2DSolver<Geometry2DType>::getConcentration),
            interpolation_method(INTERPOLATION_SPLINE),
            relative_accuracy(0.01),
            max_mesh_changes(5),
            max_iterations(20),
            fem_method(FEM_PARABOLIC),
            minor_concentration(5.0e+15),
            do_initial(false),
            mesh2(new RectangularMesh<2>())
        {
            inTemperature = 300.;
        }

        virtual ~FiniteElementMethodDiffusion2DSolver<Geometry2DType>()
        {
        }

        virtual std::string getClassName() const override;

        virtual void loadConfiguration(XMLReader&, Manager&) override;

        void compute(ComputationType type);
        void compute_initial();
        void compute_threshold();
        void compute_overthreshold();

        shared_ptr<MeshAxis> current_mesh_ptr()
        {
            return mesh2->axis[0];
        }

        RegularAxis& current_mesh()
        {
            return *static_cast<RegularAxis*>(mesh2->axis[0].get());
        }

        double burning_integral(void);  // całka strat nadprogu

        std::vector<double> modesP;                     // Integral for overthreshold computations summed for each mode

    protected:

        shared_ptr<RectangularMesh<2>> mesh2;         ///< Computational mesh

        static constexpr double hk = phys::h_J/plask::PI;      // stala plancka/2pi

        shared_ptr<Material> QW_material;

        double z;                  // z coordinate of active region

        bool initial_computation;
        bool threshold_computation;
        bool overthreshold_computation;

        double global_QW_width;                   // sumaryczna grubosc studni kwantowych [cm];
        int iterations;

        double jacobian(double r);

        std::vector<Box2D> detected_QW;

        LazyData<Vec<2>> j_on_the_mesh;  // current density vector provided by inCurrentDensity reciever
        LazyData<double> T_on_the_mesh;  // temperature vector provided by inTemperature reciever

        DataVector<double> PM;                   // Factor for overthreshold computations summed for all modes
        DataVector<double> overthreshold_dgdn;   // Factor for overthreshold computations summed for all modes

        DataVector<double> n_previous;           // concentration computed in n-1 -th step vector
        DataVector<double> n_present;            // concentration computed in n -th step vector

       /**********************************************************************/

        // Methods for solving equation
        // K*n" -  E*n = -F

        void createMatrices(DataVector<double> A_matrix, DataVector<double> RHS_vector);

        double K(int i);
//        double KInitial(size_t i, double T, double n0);	// K dla rozkladu poczatkowego
//        double KThreshold(size_t i, double T, double n0);    // K postaci D(T)
        double E(int i);        // E dla rozkladu poczatkowego i progowego
        double F(int i);        // F dla rozkladu poczatkowego i progowego

//        double Enprog(size_t i, double T, double n0);   // E dla rozkladu nadprogowego
//        double Fnprog(size_t i, double T, double n0);	// F dla rozkladu nadprogowego

        double leftSide(std::size_t i);		// lewa strona rownania dla rozkladu poczatkowego
//        double leftSideInitial(size_t i, double T, double n);		// lewa strona rownania dla rozkladu poczatkowego
//        double leftSideThreshold(size_t i, double T, double n);		// lewa strona rownania dla rozkladu progowego
//        double Lnprog(size_t i, double T, double n);	// lewa strona rownania dla rozkladu nadprogowego
        double rightSide(std::size_t i);         // prawa strona rownania dla rozkladu poczatkowego i progowego
        double nSecondDeriv(std::size_t i);                          // druga pochodna n po r

        bool MatrixFEM();
        void determineQwWidth();

        std::vector<Box2D> detectQuantumWells();
        double getZQWCoordinate();
        std::vector<double> getZQWCoordinates();

        DataVector<const Tensor2<double>> averageLi(LazyData<Vec<3,dcomplex>> initLi, const RectangularMesh<2>& mesh_Li);

        virtual void onInitialize() override;
        virtual void onInvalidate() override;

        struct ConcentrationDataImpl: public LazyDataImpl<double>
        {
            const FiniteElementMethodDiffusion2DSolver* solver;
            shared_ptr<const MeshD<2>> destination_mesh;
            InterpolationFlags interpolationFlags;
            LazyData<double> concentration;
            ConcentrationDataImpl(const FiniteElementMethodDiffusion2DSolver* solver,
                                  shared_ptr<const MeshD<2>> dest_mesh,
                                  InterpolationMethod interp);
            double at(size_t i) const override;
            size_t size() const override { return destination_mesh->size(); }
        };

        /// Provide concentration from inside to the provider (outConcentration).
        const LazyData<double> getConcentration(CarriersConcentration::EnumType what, shared_ptr<const MeshD<2>> dest_mesh,
                                                InterpolationMethod interpolation=INTERPOLATION_DEFAULT ) const;

}; // class FiniteElementMethodDiffusion2DSolver

template <> inline
double FiniteElementMethodDiffusion2DSolver<Geometry2DCartesian>::jacobian(double) {
    return 1;
}

template <> inline
double FiniteElementMethodDiffusion2DSolver<Geometry2DCylindrical>::jacobian(double r) {
    return 2*plask::PI * r;
} // 2*PI from integral over full angle,

}}} //namespace electrical::diffusion_cylindrical
