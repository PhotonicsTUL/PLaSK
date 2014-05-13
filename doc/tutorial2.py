import sys
import scipy.optimize

import optical

filename = sys.argv[1]
loadxpl(filename)

efm = optical.EffectiveFrequencyCyl("efm")
efm.geometry = GEO.main

profile = plask.StepProfile(GEO.main, default=0.)
profile[GEO.gain_region] = 500.

efm.inGain = profile.outGain

def loss_on_gain(gain):
    profile[GEO.gain_region] = gain
    mode_number = efm.find_mode(980.)
    return efm.outLoss(mode_number)

efm.lam0 = 980.

threshold_gain = scipy.optimize.brentq(loss_on_gain, 0., 2500., xtol=0.1)

profile[GEO.gain_region] = threshold_gain
mode_number = efm.find_mode(980.)
mode_wavelength = efm.outWavelength(mode_number)
print_log(LOG_INFO,
          "Threshold material gain is {:.0f}/cm with resonant wavelength {:.2f}nm"
          .format(threshold_gain, mode_wavelength))
