#include "diffusion_cylindrical.h"

#define dpbtrf_ F77_GLOBAL(dpbtrf,DPBTRF)
#define dpbtrs_ F77_GLOBAL(dpbtrs,DPBTRS)

extern "C" int dpbtrf_(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, int& info);
extern "C" int dpbtrs_(const char& uplo, const int& n, const int& kd, const int& nrhs, double* ab, const int& ldab, double* b, const int& ldb, int& info);

namespace plask { namespace solvers { namespace diffusion_cylindrical {

void DiffusionCylindricalSolver::onInitialize()
{
    initial_computation = true;
    threshold_computation = true;

    detected_QW = detectQuantumWells();
    z = getZQWCoordinate();
}

//virtual void DiffusionCylindricalSolver::onInvalidate()
//{
//    // body
//}

void DiffusionCylindricalSolver::Compute()
{
    initCalculation();

    int points = no_points;

    int mesh_change = 0;

    bool convergence = true;

    mesh = plask::RegularMesh1D( r_min, r_max, no_points );
    RegularMesh2D mesh2( mesh, plask::RegularMesh1D( z, z, 1 ) );

    T_on_the_mesh = inTemperature(mesh2);      // data temperature vector provided by inTemperature reciever
    j_on_the_mesh = inCurrentDensity(mesh2);   // data current density vector provided by inCurrentDensity reciever

    n_present.reset(mesh.size(), 0.0);
    n_previous.reset(mesh.size(), 0.0);

    do
    {
        if(!convergence && (mesh_change < max_mesh_change))
        {
            points *= 2.0;

            if (points%2 == 0)
                points += 1;

            mesh.reset( r_min, r_max, points );
//            mesh2.reset( mesh, plask::RegularMesh1D( z, z, 1 ) )

            T_on_the_mesh = inTemperature(mesh2);      // data temperature vector provided by inTemperature reciever
            j_on_the_mesh = inCurrentDensity(mesh2);   // data current density vector provided by inCurrentDensity reciever
//            n_present.reset(mesh.size(), 0.0);
//            n_previous.reset(mesh.size(), 0.0);
            mesh_change += 1;
        }
        if (initial_computation)
        {

            std::cout << "Initial computation: ";

            convergence = CylindricalMES();

            if (convergence)
            {
                std::cout << "done" << std::endl;
                initial_computation = false;
            }
        }
        else
        {
            std::cout << "Threshold computation: ";

            convergence = CylindricalMES();

            if (convergence)
            {
                std::cout << "done" << std::endl;
                threshold_computation = false;
            }

        }
    }
    while( !(convergence && !initial_computation && !threshold_computation) );



/* calculation... */

//    outConcentration = ? // computed concentration
    outCarrierConcentration.fireChanged();
}

bool DiffusionCylindricalSolver::CylindricalMES()
{
//    Computation of K*n" - E*n = -F
    bool _convergence;
    int iterations = 0;

    // LAPACK factorization (dpbtrf) and equation solver (dpbtrs) info variables:
    int info_f = 0;
    int info_s = 0;

    // linear MES elements variables:

    double r1 = 0.0, r2 = 0.0;
    double k11e = 0, k12e = 0, k22e = 0;
    double p1e = 0, p2e = 0;

    // parabolic MES elements variables:

    double r3 = 0.0;
    double k13e = 0.0, k23e = 0.0, k33e = 0.0;  // elemnty lokalnej macierzy sztywnosci
    double p3e = 0.0;

    double T = 0.0;
    double n0 = 0.0;

    double A = 0.0;
    double B = 0.0;
    double C = 0.0;

    double K = 0.0;
    double E = 0.0;
    double F = 0.0;

    double L = 0.0;
    double R = 0.0;

//    double (DiffusionCylindricalSolver::*KPointer)(size_t,double,double);
//    double (DiffusionCylindricalSolver::*EPointer)(size_t,double,double);
//    double (DiffusionCylindricalSolver::*FPointer)(size_t,double,double);

//    double (DiffusionCylindricalSolver::*LPointer)(size_t,double,double);
//    double (DiffusionCylindricalSolver::*RPointer)(size_t,double,double);

    // equation AX = RHS

    DataVector<double> A_matrix;
    DataVector<double> RHS_vector;
    DataVector<double> X_vector;

    do
    {
        if (mes_method == "linear")
        {
            A_matrix.reset();
            A_matrix.reset(2*mesh.size());
        }
        else if (mes_method == "parabolic")
        {
            A_matrix.reset();
            A_matrix.reset(3*mesh.size());
        }

        RHS_vector.reset();
        RHS_vector.reset(mesh.size());

        X_vector.reset();
        X_vector.reset(mesh.size());

        n_previous = n_present;

        if (initial_computation)
        {
//            KPointer = &DiffusionCylindricalSolver::KInitial;
//            EPointer = &DiffusionCylindricalSolver::E;
//            FPointer = &DiffusionCylindricalSolver::F;

//            LPointer = &DiffusionCylindricalSolver::leftSideInitial;
//            RPointer = &DiffusionCylindricalSolver::rightSide;

            for (int i = 0; i < mesh.size(); i++)
            {
                T = T_on_the_mesh[i];

                A = this->QW_material->A(T);
                B = this->QW_material->B(T);
                C = this->QW_material->C(T);

//                RS = rightSide(i, T, n_previous[i]); // instead of rightSide(i, T, 0) ?

                X_vector[i] = -B/(3.0*C) - (pow(2.0,1.0/3.0)*(- B*B + 3.0*A*C))/(3.0*C*pow(-2.0*B*B*B +
						9.0*A*B*C + 27.0*C*C*rightSide(i) + sqrt(4.0*pow(- B*B + 3.0*A*C,3.0) +
						pow(-2.0*B*B*B + 9.0*A*B*C + 27.0*C*C*rightSide(i),2.0)),1.0/3.0)) +
						pow(-2.0*B*B*B + 9.0*A*B*C + 27.0*C*C*rightSide(i) + sqrt(4.0*pow(-B*B +
						3.0*A*C,3.0) + pow(-2.0*B*B*B + 9.0*A*B*C +
						27.0*C*C*rightSide(i),2.0)),1.0/3.0)/(3.0*pow(2.0,1.0/3.0)*C);
            }
        }
        else
        {
//            KPointer = &DiffusionCylindricalSolver::KThreshold;
//            EPointer = &DiffusionCylindricalSolver::E;
//            FPointer = &DiffusionCylindricalSolver::F;

//            LPointer = &DiffusionCylindricalSolver::leftSideThreshold;
//            RPointer = &DiffusionCylindricalSolver::rightSide;

            const char uplo = 'U';
            int lapack_n = 0;
            int lapack_kd = 0;
            int lapack_ldab = 0;
            int lapack_nrhs = 0;
            int lapack_ldb = 0;

            if (mes_method == "linear")  // 02.10.2012 Marcin Gebski
            {
                for (int i = 0; i < mesh.size() - 1; i++) // petla po wszystkich elementach
                {

                    T = T_on_the_mesh[i+1];
                    n0 = n_previous[i+1];

                    r1 = mesh[i];
                    r2 = mesh[i+1];

                    double j1 = j_on_the_mesh[0][i];
                    double j2 = j_on_the_mesh[0][i+1];

                    K = DiffusionCylindricalSolver::K(T);
//                    K = (this->*KPointer)(i, T, n0);
                    F = DiffusionCylindricalSolver::F(i, T, n0);
                    E = DiffusionCylindricalSolver::E(T, n0);

                    if ( symmetry_type == "VCSEL")
                    {
                        K = 4*K/((r2-r1)*(r2-r1));

                        k11e = M_PI*(r2-r1)/4*(( K+E)*(r1+r2) + E*(3*r1-r2)/3);
                        k12e = M_PI*(r2-r1)/4*((-K+E)*(r1+r2) - E*(  r1+r2)/3);
                        k22e = M_PI*(r2-r1)/4*(( K+E)*(r1+r2) + E*(3*r2-r1)/3);

                        p1e  = M_PI*(r2-r1)*(F/3*(2*r1+r2) + (1/(6*plask::phys::qe*global_QW_width))*(3*j1*r1+j1*r2+j2*r1+r2*j2));
                        p2e  = M_PI*(r2-r1)*(F/3*(r1+2*r2) + (1/(6*plask::phys::qe*global_QW_width))*(3*j2*r2+j1*r2+j2*r1+r1*j1));
                    }
                    else
                    {
                        k11e =  K/(r2-r1) + E*(r2-r1)/3;
                        k12e = -K/(r2-r1) + E*(r2-r1)/6;
                        k22e =  K/(r2-r1) + E*(r2-r1)/3;
                        p1e  =  F*(r2-r1)/2;
                        p2e  =  F*(r2-r1)/2;
                    }

                        A_matrix[2*i + 1] += k11e;
                        A_matrix[2*i + 2] += k12e;
                        A_matrix[2*i + 3] += k22e;

                        RHS_vector[i] += p1e;
                        RHS_vector[i+1] += p2e;

                } // koniec petli po wszystkich elemntach

                lapack_n = lapack_ldb = (int)mesh.size();
                lapack_kd = 1;
                lapack_ldab = 2;
                lapack_nrhs = 1;
//                lapack_ldb = (int)mesh.size();

                dpbtrf_(uplo, lapack_n, lapack_kd, A_matrix.begin(), lapack_ldab, info_f);    // faktoryzacja macierzy A
                dpbtrs_(uplo, lapack_n, lapack_kd, lapack_nrhs, A_matrix.begin(), lapack_ldab, RHS_vector.begin(), lapack_ldb, info_s);    // rozwi¹zywanie Ax = B

                X_vector = RHS_vector;
            }
            else if (mes_method == "parabolic")  // 02.10.2012 Marcin Gebski
            {
                for (int i = 0; i < (mesh.size() - 1)/2; i++) // petla po wszystkich elementach
                {

                    T = T_on_the_mesh[2*i + 1];                 // wartoœæ w wêŸle œrodkowym elementu
                    n0 = n_previous[2*i + 1];                // wartoœæ w wêŸle œrodkowym elementu

                    r1 = mesh[2*i];
                    r3 = mesh[2*i + 2];

                    K = DiffusionCylindricalSolver::K(T);
//                    K = (this->*KPointer)(2*i + 1, T, n0);      // wartoœæ w wêŸle œrodkowym elementu
//                    F = (this->*FPointer)(2*i + 1, T, n0);      // wartoœæ w wêŸle œrodkowym elementu
//                    E = (this->*EPointer)(2*i + 1, T, n0);      // wartoœæ w wêŸle œrodkowym elementu
                    F = DiffusionCylindricalSolver::F(2*i + 1, T, n0);
                    E = DiffusionCylindricalSolver::E(T, n0);


                    if ( symmetry_type == "VCSEL") //VCSEL
                    {

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

                    }

//  Wypelnianie wektora macA kolumnami: //29.06.2012 r. Marcin Gebski

                    A_matrix[6*i + 2] += k11e;
                    A_matrix[6*i + 4] += k12e;
                    A_matrix[6*i + 6] += k13e;
                    A_matrix[6*i + 5]+= k22e;
                    A_matrix[6*i + 7] += k23e;
                    A_matrix[6*i + 8]+= k33e;
                    A_matrix[6*i + 3] += 0;                         // k24 = 0 - dope³nienie górnej wstêgi

                    RHS_vector[2*i] += p1e;
                    RHS_vector[2*i+1] += p2e;
                    RHS_vector[2*i+2] += p3e;

                } // koniec petli po wszystkich elemntach

                lapack_n = lapack_ldb = (int)mesh.size();
                lapack_kd = 2;
                lapack_ldab = 3;
                lapack_nrhs = 1;
//                lapack_ldb = (int)mesh.size();

                dpbtrf_(uplo, lapack_n, lapack_kd, A_matrix.begin(), lapack_ldab, info_f);    // faktoryzacja macierzy A
                dpbtrs_(uplo, lapack_n, lapack_kd, lapack_nrhs, A_matrix.begin(), lapack_ldab, RHS_vector.begin(), lapack_ldb, info_s);    // rozwi¹zywanie Ax = B

                X_vector = RHS_vector;
            }
        }
        n_present = X_vector;

//        double tolerance = 5e21;
//        double absolute_accuracy = -leftSideInitial(0, 300.0, tolerance);
        double absolute_error = 0.0;
        double relative_error = 0.0;

        if (mes_method == "linear")  // 02.10.2012 Marcin Gebski
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
//                if ( (absolute_accuracy < fabs( absolute_error ) ) && ( relative_accuracy < relative_error ) )
                if ( relative_accuracy < relative_error )
                    _convergence = false;
            }
        }
        else if (mes_method == "parabolic")
        {
            _convergence = true;
            for (int i = 0; i < (mesh.size() - 1)/2 ; i++)
            {
                n0 = n_present[2*i + 1];
                T = T_on_the_mesh[2*i + 1];
                L = leftSide(2*i + 1, T, n0);
                R = rightSide(2*i + 1);

                absolute_error = L - R;
                relative_error = absolute_error/R;
//                if ( (absolute_accuracy < fabs( absolute_error ) ) && ( relative_accuracy < relative_error ) )
                if ( relative_accuracy < relative_error )
                    _convergence = false;
            }
        }
        iterations += 1;
    }

    while (!_convergence || (iterations < max_iterations));

    return _convergence;
}

//const DataVector<double> DiffusionCylindricalSolver::getConcentration(const plask::MeshD<2>& destination_mesh, plask::InterpolationMethod interpolation_method )
//{
//    // ...
//
//    return interpolate(*mesh, computed_concentration, destination_mesh, defInterpolation<INTERPOLATION_LINEAR>(interpolation_method));
//}

void DiffusionCylindricalSolver::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd())
    {
        const std::string& param = reader.getNodeName();

        if (param == "configuration")
        {
            symmetry_type = reader.getAttribute<std::string>("symmetry_type", symmetry_type);
            mes_method = reader.getAttribute<std::string>("mes_method", mes_method);
            global_QW_width = reader.getAttribute<double>("global_QW_width", global_QW_width);
            relative_accuracy = reader.getAttribute<double>("relative_accuracy", relative_accuracy);
            interpolation_method = reader.getAttribute<std::string>("interpolation_method", interpolation_method);
            max_mesh_change = reader.getAttribute<int>("max_mesh_change", max_mesh_change);
            max_iterations = reader.getAttribute<int>("max_iterations", max_iterations);

            reader.requireTagEnd();
        }
        else if (param == "mesh")
        {
            r_min = reader.getAttribute<double>("r_min", r_min);
            r_max = reader.getAttribute<double>("r_max", r_max);
            no_points = reader.getAttribute<double>("no_points", no_points);

            reader.requireTagEnd();
        }
        else
            parseStandardConfiguration(reader, manager);
    }

}

double DiffusionCylindricalSolver::K(double T)
{
    double product = 0.0;

    if (threshold_computation)
        product += this->QW_material->D(T);
    return product;        // dla rozkladu poczatkowego nie zachodzi dyfuzja
}

//double DiffusionCylindricalSolver::KInitial(int i, double T, double n0)
//{
//    return 0;        // dla rozkladu poczatkowego nie zachodzi dyfuzja
//}
//
//
//double DiffusionCylindricalSolver::KThreshold(int i, double T, double n0)
//{
//    return D(T);
//}


double DiffusionCylindricalSolver::E(double T, double n0)
{
    return (this->QW_material->A(T) + 2*this->QW_material->B(T)*n0 + 3*this->QW_material->C(T)*n0*n0);
}


double DiffusionCylindricalSolver::F(int i, double T, double n0)
{
    if ((mes_method == "old_linear") || (mes_method == "parabolic"))  // 02.10.2012 Marcin Gebski
    return (+ j_on_the_mesh[0][i]/(plask::phys::qe*global_QW_width) + this->QW_material->B(T)*n0*n0 + 2*this->QW_material->C(T)*n0*n0*n0);

    else if (mes_method == "linear")  // 02.10.2012 Marcin Gebski
    return (this->QW_material->B(T)*n0*n0 + 2*this->QW_material->C(T)*n0*n0*n0);
    else
        throw Exception("Wrong diffusion equation RHS!");
}

//double DiffusionCylindricalSolver::leftSideInitial(int i, double T, double n)
//{
//    return -( A(T) * n + B(T) * n*n + C(T) * n*n*n);
//}


double DiffusionCylindricalSolver::nSecondDeriv(int i)
{
    double n_second_deriv;     // druga pochodna n po r
    double dr = (r_max - r_min)/(double)no_points;

    if (mes_method != "parabolic")  // 02.10.2012 Marcin Gebski
    {
        double n_right = 0, n_left = 0, n_central = 0;  // wartosci n do pochodnej: prawostronna, lewostronna, centralna

        if ( (i > 0) && (i <  mesh.size() - 1) )     //srodek przedzialu
        {
            n_right = n_present[i+1];
            n_left = n_present[i-1];
            n_central = n_present[i];

            if (symmetry_type == "VCSEL")
                n_second_deriv = (n_right - 2*n_central + n_left)/(dr*dr) + 1.0/mesh[i] * (n_right - n_left) / (2*dr);
            else if (symmetry_type == "EEL")
                n_second_deriv = (n_right - 2*n_central + n_left)/(dr*dr) + 1.0/mesh[i];
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

            if (symmetry_type == "VCSEL")
                n_second_deriv = (n_right - 2*n_central + n_left)/(dr*dr) + 1.0/mesh[i] * (n_right - n_left) / (2*dr);
            else if (symmetry_type == "EEL")
                n_second_deriv = (n_right - 2*n_central + n_left)/(dr*dr) + 1.0/mesh[i];
        }
    }
    else if (mes_method == "parabolic")  // 02.10.2012 Marcin Gebski
    {
        if (symmetry_type == "VCSEL")
            n_second_deriv = (n_present[i-1] + n_present[i+1] - 2.0*n_present[i]) * (4.0/pow(mesh[i+1] - mesh[i-1],2)) + (1.0/mesh[i]) * (1.0/(mesh[i+1]-mesh[i-1])) * (n_present[i+1] - n_present[i-1]);
        else if (symmetry_type == "EEL")
            n_second_deriv = (n_present[i-1] + n_present[i+1] - 2.0*n_present[i]) * (4.0/pow(mesh[i+1] - mesh[i-1],2));
    }

    return n_second_deriv;
}

//double DiffusionCylindricalSolver::leftSideThreshold(int i, double T, double n)
//{
//    return D(T)*nSecondDeriv(i) -( A(T) * n + B(T) * n*n + C(T) * n*n*n);
//}

double DiffusionCylindricalSolver::leftSide(int i, double T, double n)
{
    double product = -( this->QW_material->A(T) * n + this->QW_material->B(T) * n*n + this->QW_material->C(T) * n*n*n);

    if (threshold_computation)
        product += this->QW_material->D(T)*nSecondDeriv(i);

    return product;
}

double DiffusionCylindricalSolver::rightSide(int i)
{
  return -j_on_the_mesh[0][i]/(plask::phys::qe*global_QW_width);
}

std::vector<Box2D> DiffusionCylindricalSolver::detectQuantumWells()
{
    shared_ptr<RectilinearMesh2D> mesh = RectilinearMesh2DSimpleGenerator()(geometry->getChild());
    shared_ptr<RectilinearMesh2D> points = mesh->getMidpointsMesh();

    std::vector<Box2D> results;

    // Now compact each row (it can contain only one QW and each must start and end in the same point)
    double left, right;
    bool foundQW = false;
    for (int j = 0; j < points->axis1.size(); ++j) {
        bool inQW = false;
        for (int i = 0; i < points->axis0.size(); ++i) {
            auto point = points->at(i,j);
            auto tags = geometry->getRolesAt(point);
            bool QW = tags.find("QW") != tags.end() || tags.find("QD") != tags.end();
            if (QW && !inQW) { // QW start
                if (foundQW && left != mesh->axis0[i])
                    throw Exception("This solver can only handle quantum wells of identical size located exactly one above another");
                left = mesh->axis0[i];
                inQW = true;
            }
            if (!QW && inQW) { // QW end
                if (foundQW && right != mesh->axis0[i])
                    throw Exception("This solver can only handle quantum wells of identical size located exactly one above another");
                right = mesh->axis0[i];
                results.push_back(Box2D(left, mesh->axis1[j], right, mesh->axis1[j+1]));
                foundQW = true;
                inQW = false;
            }
        }
        if (inQW) { // handle situation when QW spans to the end of the structure
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

    return results;
}

double DiffusionCylindricalSolver::getZQWCoordinate()
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
        throw Exception("Active region error: no_QW = 0!");

    return coordinate;
}

}}} //namespaces
