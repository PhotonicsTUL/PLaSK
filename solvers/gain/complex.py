try:
    from .wasiak import FermiNew2D, FermiNewCyl
except ImportError:
    raise TypeError("Due to the request of MW 'gain.complex' solver is gone!")
