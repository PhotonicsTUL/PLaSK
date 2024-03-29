/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include <plask/plask.hpp>

namespace plask { namespace electrical { namespace diffusion1d {

template<typename Geometry2DType>
class PLASK_SOLVER_API DiffusionFem2DSolver: public SolverWithMesh<Geometry2DType, RegularMesh1D>
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
        ReceiverFor<ModeWavelength> inWavelength;
        ReceiverFor<ModeLightE, Geometry2DType> inLightE;

        typename ProviderFor<CarriersConcentration, Geometry2DType>::Delegate outCarriersConcentration;

        InterpolationMethod interpolation_method;   ///< Selected interpolation method
        double relative_accuracy;                   ///< Relative accuracy
        int max_mesh_changes;                       ///< Maximum number of mesh refinemenst
        int max_iterations;                         ///< Maximum number of diffusion iterations for sigle mesh size
        FemMethod fem_method;                       ///< Finite element method (linear or parabolic)
        double minor_concentration;
        bool do_initial;                            ///< Should we start from initial computations

        DiffusionFem2DSolver<Geometry2DType>(const std::string& name=""):
            SolverWithMesh<Geometry2DType,RegularMesh1D>(name),
            outCarriersConcentration(this, &DiffusionFem2DSolver<Geometry2DType>::getConcentration),
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

        virtual ~DiffusionFem2DSolver<Geometry2DType>()
        {
        }

        std::string getClassName() const override;

        void loadConfiguration(XMLReader&, Manager&) override;

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

        void createMatrices(DataVector<double> A_matrix, DataVector<double> B_vector);

        double K(int i);
        double E(int i);        // E dla rozkladu poczatkowego i progowego
        double F(int i);        // F dla rozkladu poczatkowego i progowego

        double leftSide(std::size_t i);		// lewa strona rownania dla rozkladu poczatkowego
        double rightSide(std::size_t i);         // prawa strona rownania dla rozkladu poczatkowego i progowego
        double nSecondDeriv(std::size_t i);                          // druga pochodna n po r

        bool MatrixFEM();
        void determineQwWidth();

        std::vector<Box2D> detectQuantumWells();
        double getZQWCoordinate();
        std::vector<double> getZQWCoordinates();

        DataVector<const Tensor2<double>> averageLi(LazyData<Vec<3,dcomplex>> initLi, const RectangularMesh<2>& mesh_Li);

        void onInitialize() override;
        void onInvalidate() override;

        struct ConcentrationDataImpl: public LazyDataImpl<double>
        {
            const DiffusionFem2DSolver* solver;
            shared_ptr<const MeshD<2>> destination_mesh;
            InterpolationFlags interpolationFlags;
            LazyData<double> concentration;
            ConcentrationDataImpl(const DiffusionFem2DSolver* solver,
                                  shared_ptr<const MeshD<2>> dest_mesh,
                                  InterpolationMethod interp);
            double at(size_t i) const override;
            size_t size() const override { return destination_mesh->size(); }
        };

        /// Provide concentration from inside to the provider (outConcentration).
        const LazyData<double> getConcentration(CarriersConcentration::EnumType what, shared_ptr<const MeshD<2>> dest_mesh,
                                                InterpolationMethod interpolation=INTERPOLATION_DEFAULT ) const;

}; // class DiffusionFem2DSolver

template <> inline
double DiffusionFem2DSolver<Geometry2DCartesian>::jacobian(double) {
    return 1;
}

template <> inline
double DiffusionFem2DSolver<Geometry2DCylindrical>::jacobian(double r) {
    return 2*plask::PI * r;
} // 2*PI from integral over full angle,

}}} //namespace electrical::diffusion1d
