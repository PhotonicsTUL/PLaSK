#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace diffusion_cylindrical {

class DiffusionCylindricalSolver: public plask::SolverOver < plask::Geometry2DCylindrical >
{
    public:
        plask::ReceiverFor<plask::CurrentDensity2D, plask::Geometry2DCylindrical> inCurrentDensity;
        plask::ReceiverFor<plask::Temperature, plask::Geometry2DCylindrical> inTemperature;

        plask::ProviderFor<plask::CarrierConcentration, plask::Geometry2DCylindrical>::Delegate outCarrierConcentration;

        DiffusionCylindricalSolver ( const std::string& name="" ):
            plask::SolverOver< plask::Geometry2DCylindrical > (name),
            outCarrierConcentration( this, &DiffusionCylindricalSolver::getConcentration ){}

        virtual std::string getClassName() const { return "DiffusionCylindrical1D"; }
        void Compute();
        virtual void loadConfiguration(XMLReader&, Manager&);

    private:
//        plask::DataVector<double> ?; // some internal vector used in calculations

/************************************************************************************************/

		static constexpr double hk = plask::phys::h_J/M_PI;      // stala plancka/2pi

		plask::shared_ptr< plask::Material> QW_material;

		double r_min;              // maximum radius value (right boundary value)
        double r_max;              // maximum radius value (right boundary value)
        double no_points;          // number of mesh points
        double z;                  // z coordinate of active region

        bool initial_computation;
        bool threshold_computation;

        std::string symmetry_type;                 // VCSEL or EEL

        plask::RegularMesh1D mesh;                  // radius vector (computation mesh)
        plask::DataVector<const Vec<2>> j_input;    // current density vector provided by inCurrentDensity reciever
        plask::DataVector<const double> T_input;    // temperature vector provided by inTemperature reciever

        plask::DataVector<double> n_previous;       // concentration computed in n-1 -th step vector
        plask::DataVector<double> n_present;        // concentration computed in n -th step vector
        plask::DataVector<double> j_on_the_mesh;    // current density on internal computation mesh
        plask::DataVector<double> T_on_the_mesh;    // temperature on internal computation mesh
//        bool daneWczytane;        // Czy dane zostaly wczytane
//        std::vector<std::string> wektorObliczen;    // przechowuje informacje o kolejnosci wykonywanych obliczen
//
//
        std::string mes_method;           // metoda obliczen MES ("linear" - elementy pierwszego rzedu lub "parabolic" - -||- drugiego rzedu)
//
//        std::string rodzajObliczen;       // rodzaj wykonywanych obliczen
        double global_QW_width;                   // sumaryczna grubosc studni kwantowych [m];
        double relative_accuracy;                   // dokladnosc wzgledna
        std::string interpolation_method;         // metoda interpolacji
        int max_mesh_change;                  // maksymalna liczba zmian dr
        int max_iterations;              // maksymalna liczba petli dyfuzji dla jednego dr
//
//        /**********************************************************************/
//
        //FUNKCJIE POTRZEBNE DO ROZWIAZANIA ROWNANIA
        // K*n" -  E*n = -F

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

        bool CylindricalMES();
//
//        /********** KONFIGURACJA *********/
//
//        std::string doubleToString(const double x);
//            // konwersja double na string
//
//
//
///*!!!!*/std::vector<std::vector<double> >  dyfuzja_1D();
///*!!!!*/bool rozklad(std::string rodzObliczen);     // wyznacza rozklad koncentracji nosnikow
//
//
///*!!!!*/bool wczytajDane(std::string dataFile, std::string typ = "text");
//            //funkcja wczytujaca dane z pliku i umieszczajaca je w tablicy dwuwymiarowej
//
//        std::string getSymmetryType();
//
//        void setCalculationType(std::string ctype);     // rodzaj wykonywanych obliczen
//        std::string getCalculationType();
//
//        void setTotalQWsWidth(const double qw);     // ustawianie calkowitej szerokosci QWs w ob. czynnym
//        double getTotalQWsWidth();
//
//        void setPrecision(const double prec);     // ustawianie wzglednej dokladnosci obliczen
//        double getPrecision();
//
//        void setdr(const double _dr);     // ustawianie stalej podzialu obszaru czynnego
//        double getdr();
//
//        void setInterpolationMethod(std::string _interpMetoda);     // wybor metody interpolacji
//        std::string getInterpolationMethod();
//
//        void setMaxIterations(const int iter);    // maksymalna liczba iteracji
//        int getMaxIterations();
//
//        void setMaxdrReductions(const int red);   // maksymalna liczba zmian kroku podzialu ob. czynnego
//        int getMaxdrReductions();
//
//        void setrMin();     // ustawianie rMin
//        double getrMin();
//
//        void setrMax();     // ustawianie rMin
//        double getrMax();
//
//        void setMesMethod(std::string ftype);         // metoda obliczen MES
//        std::string getMesMethod();

/************************************************************************************************/

    protected:
        std::deque<Box2D> detectQuantumWells();
        virtual void onInitialize();
        virtual void onInvalidate();

        const DataVector<double> getConcentration(const plask::MeshD<2>&, plask::InterpolationMethod ); // method providing concentration from inside to the provider (outConcentration)

}; // class DiffusionCylindricalSolver
}}} //namespace plask::solvers::diffusion_cylindrical
