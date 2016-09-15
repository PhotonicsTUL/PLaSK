try:
    from .wasiak import Fermi2D, FermiCyl
except ImportError:
    raise TypeError("Due to the request of MW 'gain.simple' solver is gone! Please use 'gain.freecarrier' instead")
