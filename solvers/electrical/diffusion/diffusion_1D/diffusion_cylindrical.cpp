#include "diffusion_cylindrical.h"

#define dpbtrf_ F77_GLOBAL(dpbtrf,DPBTRF)
#define dpbtrs_ F77_GLOBAL(dpbtrs,DPBTRS)

F77SUB dpbtrf_(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, int& info);
F77SUB dpbtrs_(const char& uplo, const int& n, const int& kd, const int& nrhs, double* ab, const int& ldab, double* b, const int& ldb, int& info);

namespace plask { namespace solvers { namespace diffusion_cylindrical {

template<typename Geometry2DType> void FiniteElementMethodDiffusion2DSolver<Geometry2DType>::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd())
    {
        const std::string& param = reader.getNodeName();

        if (param == "config")
        {
            fem_method = reader.enumAttribute<FemMethod>("fem-method").value("linear", FEM_LINEAR).value("parabolic", FEM_PARABOLIC).get(fem_method);
            relative_accuracy = reader.getAttribute<double>("accuracy", relative_accuracy);
            minor_concentration = reader.getAttribute<double>("abs-accuracy", minor_concentration);
            interpolation_method = reader.getAttribute<InterpolationMethod>("interpolation", interpolation_method);
            max_mesh_changes = reader.getAttribute<int>("maxrefines", max_mesh_changes);
            max_iterations = reader.getAttribute<int>("maxiters", max_iterations);

            reader.requireTagEnd();
        }
        else if (param == "mesh")
        {
            double r_min = reader.getAttribute<double>("start", mesh.first());
            double r_max = reader.getAttribute<double>("stop", mesh.last());
            size_t no_points = reader.getAttribute<size_t>("num", mesh.size());
            mesh.reset(r_min, r_max, no_points);
            reader.requireTagEnd();
        }
        else
            this->parseStandardConfiguration(reader, manager);
    }

}


template<typename Geometry2DType> void FiniteElementMethodDiffusion2DSolver<Geometry2DType>::onInitialize()
{
    relative_accuracy = 0.01;
    max_mesh_changes = 5;
    max_iterations = 20;
    global_QW_width = 0.0;
    minor_concentration = 5.0e+15;
    iterations = 0;

    detected_QW = detectQuantumWells();
    z = getZQWCoordinate();

}

//virtual void DiffusionCylindricalSolver::onInvalidate()
//{
//    // body
//}

template<typename Geometry2DType> void FiniteElementMethodDiffusion2DSolver<Geometry2DType>::compute(bool initial, bool threshold)
{
    initial_computation = initial;
    threshold_computation = threshold;

    this->initCalculation();
    determineQwWidth();

    this->writelog(LOG_INFO, "Computing lateral carriers diffusion using %1% FEM method", fem_method==FEM_LINEAR?"linear":"parabolic");

    if (mesh.size() % 2 == 0) mesh.reset(mesh.first(), mesh.last(), mesh.size()+1);

    plask::RegularMesh2D mesh2(mesh, plask::RegularMesh1D(z, z, 1));

    T_on_the_mesh = inTemperature(mesh2, interpolation_method);      // data temperature vector provided by inTemperature reciever
    j_on_the_mesh = inCurrentDensity(mesh2, interpolation_method);   // data current density vector provided by inCurrentDensity reciever

    n_present.reset(mesh.size(), 0.0);
    n_previous.reset(mesh.size(), 0.0);

    int mesh_changes = 0;
    bool convergence = true;

    do
    {
        if(!convergence)
        {
            mesh_changes += 1;
            if (mesh_changes > max_mesh_changes)
                throw ComputationError(this->getId(), "Maximum number of mesh refinements (%1%) reached", max_mesh_changes);
            size_t new_size = 2.0 * mesh.size() + 1;
            writelog(LOG_DETAIL, "Refining mesh (new mesh size: %1%)", new_size);
            mesh.reset(mesh.first(), mesh.last(), new_size);
            mesh2.axis0 = mesh;
            T_on_the_mesh = inTemperature(mesh2, interpolation_method);      // data temperature vector provided by inTemperature reciever
            j_on_the_mesh = inCurrentDensity(mesh2, interpolation_method);   // data current density vector provided by inCurrentDensity reciever
            n_present.reset(mesh.size(), 0.0);
            n_previous.reset(mesh.size(), 0.0);
        }
        if (initial_computation)
        {
            this->writelog(LOG_DETAIL, "Conducting initial computations");
            convergence = MatrixFEM();
            if (convergence) initial_computation = false;
        }
        if (threshold_computation)
        {
            this->writelog(LOG_DETAIL, "Conducting threshold computations");
            convergence = MatrixFEM();
            if (convergence) threshold_computation = false;
        }
    }
    while(initial_computation || threshold_computation);

    this->writelog(LOG_DETAIL, "Converged after %1% mesh refinements and %2% computational loops", mesh_changes, iterations);

    outCarriersConcentration.fireChanged();
}

template<typename Geometry2DType> bool FiniteElementMethodDiffusion2DSolver<Geometry2DType>::MatrixFEM()
{
//    Computation of K*n" - E*n = -F
    bool _convergence;
    iterations = 0;

    // LAPACK factorization (dpbtrf) and equation solver (dpbtrs) info variables:
    int info_f = 0;
    int info_s = 0;

    double T = 0.0;
    double n0 = 0.0;

    double A = 0.0;
    double B = 0.0;
    double C = 0.0;

    double L = 0.0;
    double R = 0.0;

    // equation AX = RHS

    DataVector<double> A_matrix;
    DataVector<double> RHS_vector;
    DataVector<double> X_vector;

    do
    {
        if (fem_method == FEM_LINEAR)
        {
            A_matrix.reset();
            A_matrix.reset(2*mesh.size(), 0.0);
        }
        else if (fem_method == FEM_PARABOLIC)
        {
            A_matrix.reset();
            A_matrix.reset(3*mesh.size(), 0.0);
        }

        RHS_vector.reset(mesh.size(), 0.0);
        X_vector.reset(mesh.size(), 0.0);

        n_previous = n_present.copy();

        if (initial_computation)
        {
            for (int i = 0; i < mesh.size(); i++)
            {

                T = T_on_the_mesh[i];

                A = this->QW_material->A(T);
                B = this->QW_material->B(T);
                C = this->QW_material->C(T);

                double RS = rightSide(i); // instead of rightSide(i, T, 0) ?
X_vector[i]=pow((sqrt(27*C*C*RS*RS+(4*B*B*B-18*A*B*C)*RS+4*A*A*A*C-A*A*B*B)/(2*pow(3.0,3./2.)*C*C)+(-27*C*C*RS+9*A*B*C-2*B*B*B)/
(54*C*C*C)),1./3.)-(3*A*C-B*B)/pow(9*C*C*(sqrt(27*C*C*RS*RS+(4*B*B*B-18*A*B*C)*RS+4*A*A*A*C-A*A*B*B)/
(2*pow(3.0,3./2.)*C*C)+(-27*C*C*RS+9*A*B*C-2*B*B*B)/(54*C*C*C)),1./3.)-B/(3*C);
//                X_vector[i] = -B/(3.0*C) - (pow(2.0,1.0/3.0)*(- B*B + 3.0*A*C))/(3.0*C*pow(-2.0*B*B*B +
//						9.0*A*B*C + 27.0*C*C*RS + sqrt(4.0*pow(- B*B + 3.0*A*C,3.0) +
//						pow(-2.0*B*B*B + 9.0*A*B*C + 27.0*C*C*RS,2.0)),1.0/3.0)) +
//						pow(-2.0*B*B*B + 9.0*A*B*C + 27.0*C*C*RS + sqrt(4.0*pow(-B*B +
//						3.0*A*C,3.0) + pow(-2.0*B*B*B + 9.0*A*B*C +
//						27.0*C*C*RS,2.0)),1.0/3.0)/(3.0*pow(2.0,1.0/3.0)*C);
//                double X_part = 9*C*sqrt(27*C*C*rightSide(i)*rightSide(i) + (4*B*B*B-18*A*B*C)*rightSide(i) + 4*A*A*A*C - A*A*B*B);
//                X_vector[i] = (pow(X_part, 2./3.) - pow(2,1./3.)*pow(3,1./6.)*B*pow(X_part, 1./3.) - pow(2,2./3.)*pow(3,4./3.)*A*C + pow(2,2./3.)*pow(3,1./3.)*B*B)/(pow(2,1./3.)*pow(3,7./6.)*C*pow(X_part, 1./3.));
//                double X_part = pow(sqrt(27*C*C*rightSide(i)*rightSide(i) + (4*B*B*B-18*A*B*C)*rightSide(i) + 4*A*A*A*C - A*A*B*B)/(2*pow(3,3/2)*C*C) + (-27*C*C*rightSide(i) + 9*A*B*C - 2*B*B*B)/(54*C*C*C),1/3);
//                X_vector[i] = X_part - (3*A*C - B*B)/X_part - B/(3*C);
            }
            _convergence = true;
            n_present = X_vector.copy();
        }
        else
        {
            const char uplo = 'U';
            int lapack_n = 0;
            int lapack_kd = 0;
            int lapack_ldab = 0;
            int lapack_nrhs = 0;
            int lapack_ldb = 0;

            if (fem_method == FEM_LINEAR)  // 02.10.2012 Marcin Gebski
            {
                FiniteElementMethodDiffusion2DSolver<Geometry2DType>::createMatrices(A_matrix, RHS_vector);

                lapack_n = lapack_ldb = (int)mesh.size();
                lapack_kd = 1;
                lapack_ldab = 2;
                lapack_nrhs = 1;

                dpbtrf_(uplo, lapack_n, lapack_kd, A_matrix.begin(), lapack_ldab, info_f);    // faktoryzacja macierzy A
                dpbtrs_(uplo, lapack_n, lapack_kd, lapack_nrhs, A_matrix.begin(), lapack_ldab, RHS_vector.begin(), lapack_ldb, info_s);    // rozwi�zywanie Ax = B

                X_vector = RHS_vector.copy();
            }
            else if (fem_method == FEM_PARABOLIC)  // 02.10.2012 Marcin Gebski
            {
                FiniteElementMethodDiffusion2DSolver<Geometry2DType>::createMatrices(A_matrix, RHS_vector);

                lapack_n = lapack_ldb = (int)mesh.size();
                lapack_kd = 2;
                lapack_ldab = 3;
                lapack_nrhs = 1;

                dpbtrf_(uplo, lapack_n, lapack_kd, A_matrix.begin(), lapack_ldab, info_f);                                              // factorize A
                dpbtrs_(uplo, lapack_n, lapack_kd, lapack_nrhs, A_matrix.begin(), lapack_ldab, RHS_vector.begin(), lapack_ldb, info_s); // solve Ax = B

                X_vector = RHS_vector.copy();
            }

            n_present = X_vector.copy();

            double absolute_error = 0.0;
            double relative_error = 0.0;
            double absolute_concentration_error =  abs(this->QW_material->A(T) * minor_concentration + this->QW_material->B(T) * minor_concentration*minor_concentration+ this->QW_material->C(T) * minor_concentration*minor_concentration*minor_concentration);

            /****************** RPSFEM: ******************/
            // double tolerance = 5e+15;
            // double n_tolerance = this->QW_material->A(300)*tolerance + this->QW_material->B(300)*tolerance*tolerance
            //                    + this->QW_material->C(300)*tolerance*tolerance*tolerance;
            /****************** end RPSFEM: ******************/

            if (fem_method == FEM_LINEAR)  // 02.10.2012 Marcin Gebski
            {
                _convergence = true;
                for (int i = 0; i < mesh.size(); i++)
                {
                    n0 = n_present[i];
                    T = T_on_the_mesh[i];
                    L = leftSide(i, T, n0);
                    R = rightSide(i);

                    absolute_error = L - R;
                    relative_error = absolute_error/R;
                    if ( relative_accuracy < relative_error )
                        _convergence = false;
                }
            }
            else if (fem_method == FEM_PARABOLIC)
            {
//                double max_error_absolute = 0.0;
                double max_error_relative = 0.0;
//                int max_error_point = 0.0;
//                double max_error_R = 0.0;

                _convergence = true;
                for (int i = 0; i < (mesh.size() - 1)/2 ; i++)
                {
                    n0 = n_present[2*i + 1];
                    T = T_on_the_mesh[2*i + 1];
                    L = leftSide(2*i + 1, T, n0);
                    R = rightSide(2*i + 1);

                    absolute_error = abs(L - R);
                    relative_error = abs(absolute_error/R);

                    if ( max_error_relative < relative_error )
                        max_error_relative = relative_error;
//                        max_error_absolute = absolute_error;
//                        max_error_point = mesh[2*i + 1];
//                        max_error_R = R;

                    if ( (relative_accuracy < relative_error) && (absolute_concentration_error < absolute_error) ) // (n_tolerance < absolute_error)
                    {
                        _convergence = false;
                        break;
                    }
                }
            }
            iterations += 1;
        }

    } while ( !_convergence && (iterations < max_iterations));

    return _convergence;
}

template<>
void FiniteElementMethodDiffusion2DSolver<Geometry2DCartesian>::createMatrices(DataVector<double> A_matrix, DataVector<double> RHS_vector)
{
    // linear FEM elements variables:

    double r1 = 0.0, r2 = 0.0;
    double k11e = 0, k12e = 0, k22e = 0;
    double p1e = 0, p2e = 0;

    // parabolic FEM elements variables:

    double T = 0.0;
    double n0 = 0.0;

    double K = 0.0;
    double E = 0.0;
    double F = 0.0;

    if (fem_method == FEM_LINEAR)  // 02.10.2012 Marcin Gebski
    {
        double j1 = 0.0;
        double j2 = 0.0;

        for (int i = 0; i < mesh.size() - 1; i++) // loop over all elements
        {

            T = T_on_the_mesh[i+1];
            n0 = n_previous[i+1];

            r1 = mesh[i]*1e-4;
            r2 = mesh[i+1]*1e-4;

            j1 = abs(j_on_the_mesh[i][1]*1e+3);
            j2 = abs(j_on_the_mesh[i+1][1]*1e+3);

            K = this->K(T);
            F = this->F(i, T, n0);
            E = this->E(T, n0);

            k11e = K/(r2-r1) + E*(r2-r1)/3;
            k12e = -K/(r2-r1) + E*(r2-r1)/6;
            k22e = K/(r2-r1) + E*(r2-r1)/3;

            p1e = ((r2-r1)/2)*(F + (2*j1+j2)/(3*plask::phys::qe*global_QW_width));
            p2e = ((r2-r1)/2)*(F + (2*j2+j1)/(3*plask::phys::qe*global_QW_width));

            A_matrix[2*i + 1] += k11e;
            A_matrix[2*i + 2] += k12e;
            A_matrix[2*i + 3] += k22e;

            RHS_vector[i] += p1e;
            RHS_vector[i+1] += p2e;

        } // end loop over all elements
    }
    else if (fem_method == FEM_PARABOLIC)  // 02.10.2012 Marcin Gebski
    {
        double r3 = 0.0;
        double k13e = 0.0, k23e = 0.0, k33e = 0.0;  // local stiffness matrix elements
        double p3e = 0.0;

        for (int i = 0; i < (mesh.size() - 1)/2; i++) // loop over all elements
        {
            T = T_on_the_mesh[2*i + 1];              // value in the middle node
            n0 = n_previous[2*i + 1];                // value in the middle node

            r1 = mesh[2*i]*1e-4;
            r3 = mesh[2*i + 2]*1e-4;

            K = this->K(T);
            F = this->F(2*i + 1, T, n0);
            E = this->E(T, n0);

            K = K/((r3-r1)*(r3-r1));
            double Cnst = (r3-r1)/30;

            k11e = Cnst*(70*K + 4*E);
            k12e = Cnst*(-80*K + 2*E);         // = k21e
            k13e = Cnst*(10*K - E);            // = k31e
            k22e = Cnst*(160*K + 16*E);
            k23e = k12e;                       // = k32e
            k33e = Cnst*(70*K + 16*E);

            p1e = F*(r3-r1)/6;
            p2e = 2*F*(r3-r1)/3;
            p3e = p1e;

            //  Fill matrix A_matrix columnwise: //29.06.2012 r. Marcin Gebski

            A_matrix[6*i + 2] += k11e;
            A_matrix[6*i + 4] += k12e;
            A_matrix[6*i + 6] += k13e;
            A_matrix[6*i + 5] += k22e;
            A_matrix[6*i + 7] += k23e;
            A_matrix[6*i + 8] += k33e;
            A_matrix[6*i + 3] += 0;                         // k24 = 0 - fill top band

            RHS_vector[2*i] += p1e;
            RHS_vector[2*i+1] += p2e;
            RHS_vector[2*i+2] += p3e;

        } // end loop over all elements
    }
}

template<>
void FiniteElementMethodDiffusion2DSolver<Geometry2DCylindrical>::createMatrices(DataVector<double> A_matrix, DataVector<double> RHS_vector)
{
    // linear FEM elements variables:

    double r1 = 0.0, r2 = 0.0;
    double k11e = 0, k12e = 0, k22e = 0;
    double p1e = 0, p2e = 0;

    // parabolic FEM elements variables:

    double T = 0.0;
    double n0 = 0.0;

    double K = 0.0;
    double E = 0.0;
    double F = 0.0;

    if (fem_method == FEM_LINEAR)  // 02.10.2012 Marcin Gebski
    {
        double j1 = 0.0;
        double j2 = 0.0;

        for (int i = 0; i < mesh.size() - 1; i++) // loop over all elements
        {

            T = T_on_the_mesh[i+1];
            n0 = n_previous[i+1];

            r1 = mesh[i]*1e-4;
            r2 = mesh[i+1]*1e-4;

            j1 = abs(j_on_the_mesh[i][1]*1e+3);
            j2 = abs(j_on_the_mesh[i+1][1]*1e+3);

            K = this->K(T);
            F = this->F(i, T, n0);
            E = this->E(T, n0);

            K = 4*K/((r2-r1)*(r2-r1));

            k11e = M_PI*(r2-r1)/4*(( K+E)*(r1+r2) + E*(3*r1-r2)/3);
            k12e = M_PI*(r2-r1)/4*((-K+E)*(r1+r2) - E*(  r1+r2)/3);
            k22e = M_PI*(r2-r1)/4*(( K+E)*(r1+r2) + E*(3*r2-r1)/3);

            p1e  = M_PI*(r2-r1)*(F/3*(2*r1+r2) + (1/(6*plask::phys::qe*global_QW_width))*(3*j1*r1+j1*r2+j2*r1+r2*j2));
            p2e  = M_PI*(r2-r1)*(F/3*(r1+2*r2) + (1/(6*plask::phys::qe*global_QW_width))*(3*j2*r2+j1*r2+j2*r1+r1*j1));

            A_matrix[2*i + 1] += k11e;
            A_matrix[2*i + 2] += k12e;
            A_matrix[2*i + 3] += k22e;

            RHS_vector[i] += p1e;
            RHS_vector[i+1] += p2e;

        } // end loop over all elements
    }
    else if (fem_method == FEM_PARABOLIC)  // 02.10.2012 Marcin Gebski
    {
        double r3 = 0.0;
        double k13e = 0.0, k23e = 0.0, k33e = 0.0;  // local stiffness matrix elements
        double p3e = 0.0;

        for (int i = 0; i < (mesh.size() - 1)/2; i++) // loop over all elements
        {
            T = T_on_the_mesh[2*i + 1];              // value in the middle node
            n0 = n_previous[2*i + 1];                // value in the middle node

            r1 = mesh[2*i]*1e-4;
            r3 = mesh[2*i + 2]*1e-4;

            K = this->K(T);
            F = this->F(2*i + 1, T, n0);
            E = this->E(T, n0);


            double Cnst = M_PI*(r3-r1)/30;

            k11e = Cnst*(10*K*(11*r1+3*r3)/((r3-r1)*(r3-r1)) + E*(7*r1+r3));
            k12e = Cnst*(-40*K*(3*r1+r3)/((r3-r1)*(r3-r1)) + 4*E*r1);            // = k21e
            k13e = Cnst*(10*K*(r1+r3)/((r3-r1)*(r3-r1)) - E*(r1+r3));            // = k31e
            k22e = Cnst*(160*K*(r1+r3)/((r3-r1)*(r3-r1)) + 16*E*(r1+r3));
            k23e = Cnst*(-40*K*(r1+3*r3)/((r3-r1)*(r3-r1)) + 4*E*r3);            // = k32e
            k33e = Cnst*(10*K*(3*r1+11*r3)/((r3-r1)*(r3-r1)) + E*(r1+7*r3));

            p1e = Cnst*10*F*r1;
            p2e = Cnst*20*F*(r1+r3);
            p3e = Cnst*10*F*r3;

            //  Fill matrix A_matrix columnwise: //29.06.2012 r. Marcin Gebski

            A_matrix[6*i + 2] += k11e;
            A_matrix[6*i + 4] += k12e;
            A_matrix[6*i + 6] += k13e;
            A_matrix[6*i + 5] += k22e;
            A_matrix[6*i + 7] += k23e;
            A_matrix[6*i + 8] += k33e;
            A_matrix[6*i + 3] += 0;                         // k24 = 0 - fill top band

            RHS_vector[2*i] += p1e;
            RHS_vector[2*i+1] += p2e;
            RHS_vector[2*i+2] += p3e;

        } // end loop over all elements
    }
}

template<typename Geometry2DType> const DataVector<double> FiniteElementMethodDiffusion2DSolver<Geometry2DType>::getConcentration(const plask::MeshD<2>& destination_mesh, plask::InterpolationMethod interpolation_method)
{
    RegularMesh2D mesh2(mesh, plask::RegularMesh1D(z, z, 1));
    auto concentration = interpolate(mesh2, n_present, destination_mesh, defInterpolation<INTERPOLATION_LINEAR>(interpolation_method));
    // Make sure we have concentration only in the quantum wells
    //TODO maybe more optimal approach would be reasonable?
    size_t i = 0;
    for (auto point: destination_mesh)
    {
        bool inqw = false;
        for (auto QW: detected_QW)
            if (QW.includes(point))
            {
                inqw = true;
                break;
            }
        if (!inqw) concentration[i] = NAN;
        ++i;
    }
    return concentration;
}

template<typename Geometry2DType> double FiniteElementMethodDiffusion2DSolver<Geometry2DType>::K(double T)
{
    double product = 0.0;
    if (threshold_computation)
        product += this->QW_material->D(T);
    return product;        // for initial distribution there is no diffusion
}

template<typename Geometry2DType> double FiniteElementMethodDiffusion2DSolver<Geometry2DType>::E(double T, double n0)
{
    return (this->QW_material->A(T) + 2*this->QW_material->B(T)*n0 + 3*this->QW_material->C(T)*n0*n0);
}


template<typename Geometry2DType> double FiniteElementMethodDiffusion2DSolver<Geometry2DType>::F(int i, double T, double n0)
{
    if (fem_method == FEM_PARABOLIC)  // 02.10.2012 Marcin Gebski
        return (+ abs(j_on_the_mesh[i][1]*1e+3)/(plask::phys::qe*global_QW_width) + this->QW_material->B(T)*n0*n0 + 2*this->QW_material->C(T)*n0*n0*n0);
    else if (fem_method == FEM_LINEAR)  // 02.10.2012 Marcin Gebski
        return (this->QW_material->B(T)*n0*n0 + 2*this->QW_material->C(T)*n0*n0*n0);
    else
        throw Exception("Wrong diffusion equation RHS!");
}


template<typename Geometry2DType> double FiniteElementMethodDiffusion2DSolver<Geometry2DType>::nSecondDeriv(int i)
{
    double n_second_deriv = 0.0;     // second derivative with respect to r
    double dr = 0.0;

    if (fem_method != FEM_PARABOLIC)  // 02.10.2012 Marcin Gebski
    {
        dr = (mesh.last() - mesh.first())*1e-4/(double)mesh.size();
        double n_right = 0, n_left = 0, n_central = 0;  // n values for derivative: right-side, left-side, central

        if ( (i > 0) && (i <  mesh.size() - 1) )     // middle of the range
        {
            n_right = n_present[i+1];
            n_left = n_present[i-1];
            n_central = n_present[i];

            n_second_deriv = (n_right - 2*n_central + n_left)/(dr*dr); // + 1.0/(mesh[i]*1e-4);

            if (std::is_same<Geometry2DType, Geometry2DCylindrical>::value)
                n_second_deriv = (n_right - 2*n_central + n_left)/(dr*dr) + 1.0/(mesh[i]*1e-4) * (n_right - n_left) / (2*dr);
        }
        else if (i == 0)     // punkt r = 0
        {
            n_right = n_present[i+1];
            n_left = n_present[i+1];    //w r = 0 jest maksimum(warunki brzegowe), zalozylem nP = nL w blisko maksimum
            n_central = n_present[i];

            n_second_deriv = 2*(n_right - 2*n_central + n_left)/(dr*dr);	//wymyslone przez Wasiak 2011.02.07
        }
        else  // prawy brzeg
        {
            n_right = n_present[i-1];
            n_left = n_present[i-1];    // podobnie jak we wczesniejszym warunku
            n_central = n_present[i];

            n_second_deriv = (n_right - 2*n_central + n_left)/(dr*dr); // + 1.0/(mesh[i]*1e-4);

            if (std::is_same<Geometry2DType, Geometry2DCylindrical>::value)
                n_second_deriv = (n_right - 2*n_central + n_left)/(dr*dr) + 1.0/(mesh[i]*1e-4) * (n_right - n_left) / (2*dr);
        }
    }
    else if (fem_method == FEM_PARABOLIC)  // 02.10.2012 Marcin Gebski
    {
        dr = (mesh[i+1] - mesh[i-1])*1e-4;
        n_second_deriv = (n_present[i-1] + n_present[i+1] - 2.0*n_present[i]) * (4.0/(dr*dr));

        if (std::is_same<Geometry2DType, Geometry2DCylindrical>::value)
            n_second_deriv += (1.0/(mesh[i]*1e-4)) * (1.0/dr) * (n_present[i+1] - n_present[i-1]); // adding cylindrical component of laplace operator
    }

    return n_second_deriv;
}


template<typename Geometry2DType> double FiniteElementMethodDiffusion2DSolver<Geometry2DType>::leftSide(int i, double T, double n)
{
    double product = -( this->QW_material->A(T) * n + this->QW_material->B(T) * n*n + this->QW_material->C(T) * n*n*n);

    if (threshold_computation)
        product += this->QW_material->D(T)*nSecondDeriv(i);

    return product;
}

template<typename Geometry2DType> double FiniteElementMethodDiffusion2DSolver<Geometry2DType>::rightSide(int i)
{
    return -abs(j_on_the_mesh[i][1])*1e+3/(plask::phys::qe*global_QW_width);
}

template<typename Geometry2DType> std::vector<Box2D> FiniteElementMethodDiffusion2DSolver<Geometry2DType>::detectQuantumWells()
{
    shared_ptr<RectilinearMesh2D> mesh = RectilinearMesh2DSimpleGenerator()(this->geometry->getChild());
    shared_ptr<RectilinearMesh2D> points = mesh->getMidpointsMesh();

    std::vector<Box2D> results;

    // Compact each row (it can contain only one QW and each must start and end in the same point)
    double left = 0., right = 0.;
    bool foundQW = false;
    bool had_active = false, after_active = false;
    for (int r = 0; r < points->axis1.size(); ++r)
    {
        bool inQW = false;
        bool active_row = false;
        for (int c = 0; c < points->axis0.size(); ++c)
        {
            auto point = points->at(c,r);
            auto tags = this->geometry->getRolesAt(point);
            bool QW = tags.find("QW") != tags.end() || tags.find("QD") != tags.end();
            bool active = tags.find("active") != tags.end();
            if (QW && !active)
                throw Exception("%1%: All marked quantum wells must belong to marked active region.", this->getId());
            if (QW && !inQW)        // QW start
            {
                if (foundQW)
                {
                    if (left != mesh->axis0[c])
                        throw Exception("%1%: Left edge of quantum wells not vertically aligned.", this->getId());
                    if (*this->geometry->getMaterial(point) != *QW_material)
                        throw Exception("%1%: Quantum wells of multiple materials not supported.", this->getId());
                }
                else
                {
                    QW_material = this->geometry->getMaterial(point);
                }
                left = mesh->axis0[c];
                inQW = true;
            }
            if (!QW && inQW)        // QW end
            {
                if (foundQW && right != mesh->axis0[c])
                    throw Exception("%1%: Right edge of quantum wells not vertically aligned.", this->getId());
                right = mesh->axis0[c];
                results.push_back(Box2D(left, mesh->axis1[r], right, mesh->axis1[r+1]));
                foundQW = true;
                inQW = false;
            }
            if (active) {
                active_row = had_active = true;
                if (after_active)  throw Exception("%1%: Multiple active regions not supported.", this->getId());
            }
        }
        if (inQW)
        { // handle situation when QW spans to the end of the structure
            if (foundQW && right != mesh->axis0[points->axis0.size()])
                throw Exception("%1%: Right edge of quantum wells not vertically aligned.", this->getId());
            right = mesh->axis0[points->axis0.size()];
            results.push_back(Box2D(left, mesh->axis1[r], right, mesh->axis1[r+1]));
            foundQW = true;
            inQW = false;
        }
        if (!active_row && had_active) after_active = true;
    }

    // Compact results in vertical direction
    //TODO

    return results;
}

template<typename Geometry2DType> double FiniteElementMethodDiffusion2DSolver<Geometry2DType>::getZQWCoordinate()
{
    double coordinate = 0.0;
    int no_QW = detected_QW.size();
    int no_Box = 0;

    if ((no_QW%2 == 0) && (no_QW > 0))
    {
        no_Box = no_QW/2 -1;
        coordinate = (detected_QW[no_Box].lower[1] + detected_QW[no_Box].upper[1]) / 2.0;
    }
    else if ((no_QW%2 == 1) && (no_QW > 0))
    {
        no_Box = (no_QW - 1)/2;
        coordinate = (detected_QW[no_Box].lower[1] + detected_QW[no_Box].upper[1]) / 2.0;
    }
    else
        throw Exception("No quantum wells defined");

    return coordinate;
}

template<typename Geometry2DType> void FiniteElementMethodDiffusion2DSolver<Geometry2DType>::determineQwWidth()
{
    for (int i = 0; i< detected_QW.size(); i++)
    {
        global_QW_width += ( this->detected_QW[i].upper[1] - this->detected_QW[i].lower[1] );
    }
    global_QW_width *= 1e-4;
}

template<> std::string FiniteElementMethodDiffusion2DSolver<Geometry2DCartesian>::getClassName() const { return "Diffusion2D"; }
template<> std::string FiniteElementMethodDiffusion2DSolver<Geometry2DCylindrical>::getClassName() const { return "DiffusionCyl"; }

template class FiniteElementMethodDiffusion2DSolver<Geometry2DCartesian>;
template class FiniteElementMethodDiffusion2DSolver<Geometry2DCylindrical>;

}}} //namespaces
