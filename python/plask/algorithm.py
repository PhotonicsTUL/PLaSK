# -*- coding: utf-8 -*-
'''
This module constains the set of high-level PLaSK algorithm typically used
in computations of semiconductor lasers.

TODO doc

'''
import plask


class ThermoElectric(object):
    '''Algorithm for thermo-electric calculations without the optical part.

    This algorithm performs under-threshold thermo-electrical computations.
    It computes electric current flow and tempereture distribution in a self-
    consistent loop until desired convergence is reached.

    ThermoElectric(geometry, mesh, js, beta, thermal, electrical,
                   geometry_thermal, geometry_electrical,
                   temperature, heatflux, convection, radiation,
                   voltages, mesh_thermal, mesh_electrical,
                   err, terr, verr, relerr, tfreq)

    Parameters
    ----------
    geometry : Cartesian2D geometry, optional
        Geometry of the structure analyzed in the algorithm. Not required
        if configured solvers are specified in the `thermal` and `electrical`
        parameters.
    mesh : Rectilinear2D mesh or its generator, optional
        Mesh used for computations. Not required if configured solvers are
        specified in the `thermal` and `electrical` parameters.

    TODO


    '''

    def __init__(self, geometry=None, mesh=None, js=None, beta=None,
                 thermal=None, electrical=None,
                 geometry_thermal=None, geometry_electrical=None,
                 temperature=[], heatflux=[], convection=[], radiation=[],
                 voltages=[], mesh_thermal=None, mesh_electrical=None,
                 err=1e-3, terr=None, verr=None, relerr=True,
                 tfreq=6):

        if thermal is None:
            import thermal as thermals
            if geometry is None or mesh is None:
                raise ValueError("Geometry and mesh must be specified if no " +
                                 "preconfigured solvers are provided")
            if type(geometry) == plask.geometry.Cartesian2D:
                thermal = thermals.Static2D("thermal")
            elif type(geometry) == plask.geometry.Cylindrical2D:
                thermal = thermals.StaticCyl("thermal")
            elif type(geometry) == plask.geometry.Cartesian2D:
                thermal = thermals.Static3D("thermal")
            else:
                raise TypeError("wrong geometry type")
            thermal.geometry = geometry_thermal if geometry_thermal else geometry
            thermal.mesh = mesh_thermal if mesh else mesh
            thermal.corrlim = terr if terr is not None else err

        if electrical is None:
            import electrical as electricals
            if geometry is None or mesh is None:
                raise ValueError("Geometry and mesh must be specified if no " +
                                 "preconfigured solvers are provided")
            if type(geometry) == plask.geometry.Cartesian2D:
                electrical = electricals.Static2D("electrical")
            elif type(geometry) == plask.geometry.Cylindrical2D:
                electrical = electricals.StaticCyl("electrical")
            elif type(geometry) == plask.geometry.Cartesian2D:
                electrical = electricals.Static3D("electrical")
            else:
                raise TypeError("wrong geometry type")
            electrical.geometry = geometry_electrical if geometry_electrical else geometry
            electrical.mesh = mesh_electrical if mesh else mesh
            electrical.corrlim = verr if verr is not None else err
            electrical.beta = beta
            electrical.js = js

        electrical.inTemperature = thermal.outTemperature
        thermal.inHeat = electrical.outHeat

        self.thermal = thermal
        self.electrical = electrical

        self.tfreq = tfreq
