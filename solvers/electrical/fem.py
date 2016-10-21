import plask as _plask
_plask.print_log('warning', "Electrical library 'fem' is obsolete, use 'shockley' instead!")

from shockley import Shockley2D, ShockleyCyl
