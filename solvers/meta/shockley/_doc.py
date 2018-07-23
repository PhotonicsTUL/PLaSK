# coding: utf8
# Copyright (C) 2014 Photonics Group, Lodz University of Technology

outLightMagnitude = """\
Provider of the computed optical field magnitude [W/m²].


outLightMagnitude(n=0, mesh, interpolation='default')

:param int n: Computed mode number.
:param mesh mesh: Target mesh to get the field at.
:param str interpolation: Requested interpolation method.

:return: Data with the optical field magnitude on the specified mesh **[W/m²]**.

You may obtain the number of different values this provider can return by
testing its length.

Example:
   Connect the provider to a receiver in some other solver:

   >>> other_solver.inLightMagnitude = solver.outLightMagnitude

   Obtain the provided field:

   >>> solver.outLightMagnitude(0, mesh)
   <plask.Data at 0x1234567>

   Test the number of provided values:

   >>> len(solver.outLightMagnitude)
   3

See also:

   Provider class: :class:`plask.flow.LightMagnitudeProviderCyl`

   Receciver class: :class:`plask.flow.LightMagnitudeReceiverCyl`
"""

outLoss = """\
Provider of the computed modal extinction [1/cm].


outLoss(n=0)

:param int n: Value number.

:return: Value of the modal extinction **[1/cm]**.

You may obtain the number of different values this provider can return by
testing its length.

Example:
   Connect the provider to a receiver in some other solver:

   >>> other_solver.inModalLoss = solver.outLoss

   Obtain the provided value:

   >>> solver.outLoss(n=0)
   1000

   Test the number of provided values:

   >>> len(solver.outLoss)
   3

See also:

   Provider class: :class:`plask.flow.ModalLossProvider`

   Receciver class: :class:`plask.flow.ModalLossReceiver`
"""

outWavelength = """\
Provider of the computed wavelength [nm].


outWavelength(n=0)

:param int n: Computed mode number.

:return: Value of the wavelength **[nm]**.

You may obtain the number of different values this provider can return by
testing its length.

Example:
   Connect the provider to a receiver in some other solver:

   >>> other_solver.inWavelength = solver.outWavelength

   Obtain the provided value:

   >>> solver.outWavelength(n=0)
   1000

   Test the number of provided values:

   >>> len(solver.outWavelength)
   3

See also:

   Provider class: :class:`plask.flow.WavelengthProvider`

   Receciver class: :class:`plask.flow.WavelengthReceiver`
"""

outRefractiveIndex = """\
Provider of the computed refractive index [-].


outRefractiveIndex(mesh, interpolation='default')

:param mesh mesh: Target mesh to get the field at.
:param str interpolation: Requested interpolation method.

:return: Data with the refractive index on the specified mesh **[-]**.

Example:
   Connect the provider to a receiver in some other solver:

   >>> other_solver.inRefractiveIndex = solver.outRefractiveIndex

   Obtain the provided field:

   >>> solver.outRefractiveIndex(mesh)
   <plask.Data at 0x1234567>

See also:

   Provider class: :class:`plask.flow.RefractiveIndexProviderCyl`

   Receciver class: :class:`plask.flow.RefractiveIndexReceiverCyl`
"""

outLightE = """\
Provider of the computed electric field [V/m].


outLightE(n=0, mesh, interpolation='default')

:param int n: Value number.
:param mesh mesh: Target mesh to get the field at.
:param str interpolation: Requested interpolation method.

:return: Data with the electric field on the specified mesh **[V/m]**.

You may obtain the number of different values this provider can return by
testing its length.

Example:
   Connect the provider to a receiver in some other solver:

   >>> other_solver.inLightE = solver.outLightE

   Obtain the provided field:

   >>> solver.outLightE(0, mesh)
   <plask.Data at 0x1234567>

   Test the number of provided values:

   >>> len(solver.outLightE)
   3

See also:

   Provider class: :class:`plask.flow.LightEProvider2D`

   Receciver class: :class:`plask.flow.LightEReceiver2D`
"""

outNeff = """\
Provider of the computed effective index [-].


outNeff(n=0)

:param int n: Value number.

:return: Value of the effective index **[-]**.

You may obtain the number of different values this provider can return by
testing its length.

Example:
   Connect the provider to a receiver in some other solver:

   >>> other_solver.inEffectiveIndex = solver.outNeff

   Obtain the provided value:

   >>> solver.outNeff(n=0)
   1000

   Test the number of provided values:

   >>> len(solver.outNeff)
   3

See also:

   Provider class: :class:`plask.flow.EffectiveIndexProvider`

   Receciver class: :class:`plask.flow.EffectiveIndexReceiver`
"""
