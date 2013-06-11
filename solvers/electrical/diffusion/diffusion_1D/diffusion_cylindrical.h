#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace diffusion_cylindrical {

template<typename Geometry2DType>
class FiniteElementMethodDiffusion2DSolver: public plask::SolverOver < Geometry2DType > //plask::Geometry2DCylindrical
{
    public:
        enum FemMethod {
            FEM_LINEAR,
            FEM_PARABOLIC
        };

        plask::ReceiverFor<plask::CurrentDensity, Geometry2DType> inCurrentDensity;
        plask::ReceiverFor<plask::Temperature, Geometry2DType> inTemperature;

        typename ProviderFor<plask::CarriersConcentration, Geometry2DType>::Delegate outCarriersConcentration;

        plask::RegularMesh1D mesh;                  ///< Radial mesh

        double relative_accuracy;                   ///< Relative accuracy
        InterpolationMethod interpolation_method;   ///< Selected interpolation method
        int max_mesh_changes;                  // maksymalna liczba zmian dr
        int max_iterations;              // maksymalna liczba petli dyfuzji dla jednego dr
        FemMethod fem_method;           // metoda obliczen MES ("linear" - elementy pierwszego rzedu lub "parabolic" - -||- drugiego rzedu)
        double minor_concentration;

        FiniteElementMethodDiffusion2DSolver<Geometry2DType>(const std::string& name=""):
            plask::SolverOver<Geometry2DType> (name),
            outCarriersConcentration(this, &FiniteElementMethodDiffusion2DSolver<Geometry2DType>::getConcentration),
            interpolation_method(INTERPOLATION_SPLINE)
            {}

        virtual std::string getClassName() const;

        virtual void loadConfiguration(XMLReader&, Manager&);

        void compute(bool initial, bool threshold);


    protected:
//        plask::DataVector<double> ?; // some internal vector used in calculations

/************************************************************************************************/

        static constexpr double hk = plask::phys::h_J/M_PI;      // stala plancka/2pi

        plask::shared_ptr<plask::Material> QW_material;

        double z;                  // z coordinate of active region

        bool initial_computation;
        bool threshold_computation;

        double global_QW_width;                   // sumaryczna grubosc studni kwantowych [m];
        int iterations;

        std::vector<Box2D> detected_QW;

        plask::DataVector<const Vec<2>> j_on_the_mesh;    // current density vector provided by inCurrentDensity reciever
        plask::DataVector<const double> T_on_the_mesh;    // temperature vector provided by inTemperature reciever

        plask::DataVector<double> n_previous;       // concentration computed in n-1 -th step vector
        plask::DataVector<double> n_present;        // concentration computed in n -th step vector
//        plask::DataVector<double> j_on_the_mesh;    // current density on internal computation mesh
//        plask::DataVector<double> T_on_the_mesh;    // temperature on internal computation mesh
//        bool daneWczytane;        // Czy dane zostaly wczytane
//        std::vector<std::string> wektorObliczen;    // przechowuje informacje o kolejnosci wykonywanych obliczen
//
//

//
//        std::string rodzajObliczen;       // rodzaj wykonywanych obliczen

//
//        /**********************************************************************/
//
        // Methods for solving equation
        // K*n" -  E*n = -F

        void createMatrices(plask::DataVector<double> A_matrix, plask::DataVector<double> RHS_vector);

        double K(double T);
//        double KInitial(size_t i, double T, double n0);	// K dla rozkladu poczatkowego
//        double KThreshold(size_t i, double T, double n0);    // K postaci D(T)
        double E(double T, double n0);        // E dla rozkladu poczatkowego i progowego
        double F(int i, double T, double n0);        // F dla rozkladu poczatkowego i progowego

//        double Enprog(size_t i, double T, double n0);   // E dla rozkladu nadprogowego
//        double Fnprog(size_t i, double T, double n0);	// F dla rozkladu nadprogowego

        double leftSide(int i, double T, double n);		// lewa strona rownania dla rozkladu poczatkowego
//        double leftSideInitial(size_t i, double T, double n);		// lewa strona rownania dla rozkladu poczatkowego
//        double leftSideThreshold(size_t i, double T, double n);		// lewa strona rownania dla rozkladu progowego
//        double Lnprog(size_t i, double T, double n);	// lewa strona rownania dla rozkladu nadprogowego
        double rightSide(int i);         // prawa strona rownania dla rozkladu poczatkowego i progowego
        double nSecondDeriv(int i);                          // druga pochodna n po r

        bool MatrixFEM();
        void determineQwWidth();

        std::vector<Box2D> detectQuantumWells();
        double getZQWCoordinate();
        virtual void onInitialize();
        //virtual void onInvalidate();

        const DataVector<double> getConcentration(const plask::MeshD<2>& dest_mesh, plask::InterpolationMethod interpolation=DEFAULT_INTERPOLATION ); // method providing concentration from inside to the provider (outConcentration)

}; // class FiniteElementMethodDiffusion2DSolver


}}} //namespace plask::solvers::diffusion_cylindrical
