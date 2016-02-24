#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace diffusion_cylindrical {

template<typename Geometry2DType>
class PLASK_SOLVER_API FiniteElementMethodDiffusion2DSolver: public plask::SolverWithMesh<Geometry2DType, plask::RegularMesh1D>
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

        plask::ReceiverFor<plask::CurrentDensity, Geometry2DType> inCurrentDensity;
        plask::ReceiverFor<plask::Temperature, Geometry2DType> inTemperature;
        plask::ReceiverFor<plask::Gain, Geometry2DType> inGain;
        plask::ReceiverFor<plask::Wavelength> inWavelength;
        plask::ReceiverFor<plask::LightMagnitude, Geometry2DType> inLightMagnitude;

        typename ProviderFor<plask::CarriersConcentration, Geometry2DType>::Delegate outCarriersConcentration;

        double relative_accuracy;                   ///< Relative accuracy
        InterpolationMethod interpolation_method;   ///< Selected interpolation method
        int max_mesh_changes;                  // maksymalna liczba zmian dr
        int max_iterations;              // maksymalna liczba petli dyfuzji dla jednego dr
        FemMethod fem_method;           // metoda obliczen MES ("linear" - elementy pierwszego rzedu lub "parabolic" - -||- drugiego rzedu)
        double minor_concentration;
        bool do_initial;                            ///< Should we start from initial computations

        FiniteElementMethodDiffusion2DSolver<Geometry2DType>(const std::string& name=""):
            plask::SolverWithMesh<Geometry2DType,plask::RegularMesh1D>(name),
            outCarriersConcentration(this, &FiniteElementMethodDiffusion2DSolver<Geometry2DType>::getConcentration),
            interpolation_method(INTERPOLATION_SPLINE),
            do_initial(false),
            mesh2(new plask::RectangularMesh<2>())
        {
            relative_accuracy = 0.01;
            max_mesh_changes = 5;
            max_iterations = 20;
            minor_concentration = 5.0e+15;
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

        plask::shared_ptr<plask::RectangularAxis> current_mesh_ptr()
        {
            return mesh2->axis0;
        }

        plask::RegularAxis& current_mesh()
        {
            return *static_cast<plask::RegularAxis*>(mesh2->axis0.get());
        }

        double burning_integral(void);  // całka strat nadprogu

        std::vector<double> modesP;                     // Integral for overthreshold computations summed for each mode
        
    protected:

        shared_ptr<plask::RectangularMesh<2>> mesh2;         ///< Computational mesh

        static constexpr double hk = plask::phys::h_J/M_PI;      // stala plancka/2pi

        plask::shared_ptr<plask::Material> QW_material;

        double z;                  // z coordinate of active region

        bool initial_computation;
        bool threshold_computation;
        bool overthreshold_computation;

        double global_QW_width;                   // sumaryczna grubosc studni kwantowych [cm];
        int iterations;

        double jacobian(double r);
        
        std::vector<Box2D> detected_QW;

        plask::LazyData<Vec<2>> j_on_the_mesh;  // current density vector provided by inCurrentDensity reciever
        plask::LazyData<double> T_on_the_mesh;  // temperature vector provided by inTemperature reciever

        plask::DataVector<double> PM;                   // Factor for overthreshold computations summed for all modes
        plask::DataVector<double> overthreshold_dgdn;   // Factor for overthreshold computations summed for all modes

        plask::DataVector<double> n_previous;           // concentration computed in n-1 -th step vector
        plask::DataVector<double> n_present;            // concentration computed in n -th step vector

       /**********************************************************************/

        // Methods for solving equation
        // K*n" -  E*n = -F

        void createMatrices(plask::DataVector<double> A_matrix, plask::DataVector<double> RHS_vector);

        double K(int i);
//        double KInitial(size_t i, double T, double n0);	// K dla rozkladu poczatkowego
//        double KThreshold(size_t i, double T, double n0);    // K postaci D(T)
        double E(int i);        // E dla rozkladu poczatkowego i progowego
        double F(int i);        // F dla rozkladu poczatkowego i progowego

//        double Enprog(size_t i, double T, double n0);   // E dla rozkladu nadprogowego
//        double Fnprog(size_t i, double T, double n0);	// F dla rozkladu nadprogowego

        double leftSide(int i);		// lewa strona rownania dla rozkladu poczatkowego
//        double leftSideInitial(size_t i, double T, double n);		// lewa strona rownania dla rozkladu poczatkowego
//        double leftSideThreshold(size_t i, double T, double n);		// lewa strona rownania dla rozkladu progowego
//        double Lnprog(size_t i, double T, double n);	// lewa strona rownania dla rozkladu nadprogowego
        double rightSide(int i);         // prawa strona rownania dla rozkladu poczatkowego i progowego
        double nSecondDeriv(int i);                          // druga pochodna n po r

        bool MatrixFEM();
        void determineQwWidth();

        std::vector<Box2D> detectQuantumWells();
        double getZQWCoordinate();
        std::vector<double> getZQWCoordinates();

        plask::DataVector<const double> averageLi(plask::LazyData<double> initLi, const plask::RectangularMesh<2>& mesh_Li);

        virtual void onInitialize() override;
        virtual void onInvalidate() override;

        struct ConcentrationDataImpl: public LazyDataImpl<double>
        {
            const FiniteElementMethodDiffusion2DSolver* solver;
            shared_ptr<const MeshD<2>> destination_mesh;
            InterpolationFlags interpolationFlags;
            LazyData<double> concentration;
            ConcentrationDataImpl(const FiniteElementMethodDiffusion2DSolver* solver,
                                  shared_ptr<const plask::MeshD<2>> dest_mesh,
                                  InterpolationMethod interp);
            double at(size_t i) const;
            size_t size() const override { return destination_mesh->size(); }
        };

        /// Provide concentration from inside to the provider (outConcentration).
        const LazyData<double> getConcentration(shared_ptr<const plask::MeshD<2>> dest_mesh,
                                                InterpolationMethod interpolation=INTERPOLATION_DEFAULT ) const;

}; // class FiniteElementMethodDiffusion2DSolver

template <>
double FiniteElementMethodDiffusion2DSolver<Geometry2DCartesian>::jacobian(double) {
    return 1;
}

template <>
double FiniteElementMethodDiffusion2DSolver<Geometry2DCylindrical>::jacobian(double r) {
    return 2*M_PI * r;
} // 2*M_PI from integral over full angle, 

}}} //namespace plask::solvers::diffusion_cylindrical
