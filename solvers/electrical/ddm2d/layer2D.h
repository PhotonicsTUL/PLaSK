/**************************************************************************************************
									PREPROCESSOR DIRECTIVES
**************************************************************************************************/
#ifndef LAYER2D_H
#define LAYER2D_H

/**************************************************************************************************
										INCLUDES
**************************************************************************************************/
#include "../../../plask/phys/constants.h"
#include <iostream>
#include <string>
#include <cmath>

/**************************************************************************************************
										CLASS Layer2D
**************************************************************************************************/
class Layer2D
{
public:
	Layer2D(int iNo, double iThicknessX, double iThicknessY, std::string iMaterial, double iX, double iY, double iZ, std::string iDopant, double iDopConc, std::string iID);
	Layer2D();
	~Layer2D();

	// getters
	void getInfo(); /// all info about the layer
	int getNo();
	double getThicknessX(); /// not scaled thickness X
	double getThicknessY(); /// not scaled thickness Y
	std::string getMaterial();
	double getX();
	double getY();
	double getZ();
	std::string getDopant();
	double getDopConc(); /// not scaled dopant concentration
	std::string getID();
	double getNd();
	double getNdNorm();
	double getNa();
	double getNaNorm();

    double getK(); /// get thermal conductivity
    double getKNorm(); /// get normalized thermal conductivity
    double getEpsR(); /// get dielectric constant
    double getEpsRNorm(); /// get normalized dielectric constant
    double getEg(); /// get energy gap
    double getEgNorm(); /// get normalized energy gap
    double getEc0(); /// get conduction band edge
    double getEc0Norm(); /// get normalized conduction band edge
    double getEv0(); /// get valence band edge
    double getEv0Norm(); /// get normalized valence band edge
    double getMe(); /// get electron effective mass
    double getMh(); /// get hole effective mass
    double getNi(); /// get intrinsic carrier concentration
    double getNiNorm(); /// get normalized intrinsic carrier concentration
    double getNc(); /// get electron density of states
    double getNcNorm(); /// get normalized electron density of states
    double getNv(); /// get hole density of states
    double getNvNorm(); /// get normalized hole density of states
    double getEd(); /// get donor ionization energy
    double getEdNorm(); /// get normalized donor ionization energy
    double getEa(); /// get acceptor ionization energy
    double getEaNorm(); /// get normalized acceptor ionization energy
    double getMiN(); /// get electron mobility
    double getMiNNorm(); /// get normalized electron mobility
    double getMiP(); /// get hole mobility
    double getMiPNorm(); /// get normalized hole mobility
    double getTn(); /// get SRH recombination lifetime for electrons
    double getTnNorm(); /// get normalized SRH recombination lifetime for electrons
    double getTp(); /// get SRH recombination lifetime for holes
    double getTpNorm(); /// get normalized SRH recombination lifetime for holes
    double getB(); /// get radiative recombination coefficient
    double getBNorm(); /// get normalized radiative recombination coefficient
    double getCn(); /// get Auger recombination coefficient for electrons
    double getCnNorm(); /// get normalized Auger recombination coefficient for electrons
    double getCp(); /// get Auger recombination coefficient for holes
    double getCpNorm(); /// get normalized Auger recombination coefficient for holes
    double getH(double iT); /// get heat source
    double getHNorm(double iT); /// get normalized heat source
    double getPsp(); /// get spontaneous polarization
    double getAlc(); /// get lattice constant
    double gete13(); /// get piezoelectric constant
    double gete33(); /// get piezoelectric constant
    double getC13(); /// get elastic constant
    double getC33(); /// get elastic constant

private:
	// members
	int mNo; // layer number
	double mThicknessX; // layer thickness
	double mThicknessY; // layer thickness
	std::string mMaterial; // material
	double mX; // material content
	double mY; // material content
	double mZ; // material content
	std::string mDopant; // dopant
	double mDopConc; // dopant concentration
	std::string mID; // layer ID
    double scaleE, scaleX, scaleT, scaleN, scaleEpsR, scaleK, scaleMi, scaleR, scalet, scaleB, scaleC, scaleH; // scaling parameters for energy (in [eV]), position (in [um]), temperature (in [K]),
        // concentration (in [1/cm^3]), dielectric constant (no unit), thermal conductivity (in [W/(m*K)]), mobility (in [cm^2/(V*s)]), recombination rate (in [1/(cm^3*s)], lifetime (in [s]),
        // radiative recomb. coefficient (in [cm^3/s]), Auger recomb. coefficient (in [cm^6/s]), heat source density (in [W/m^3])

};

/**************************************************************************************************
									PREPROCESSOR DIRECTIVE
**************************************************************************************************/
#endif // Layer2D_H
