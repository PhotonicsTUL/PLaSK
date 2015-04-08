def loss_on_voltage(voltage):
        ELECTRICAL.invalidate()
        ELECTRICAL.voltage_boundary[0].value = voltage
        verr = ELECTRICAL.compute(1)
        terr = THERMAL.compute(1)
        iters=0
        while (terr > THERMAL.maxerr or verr > ELECTRICAL.maxerr) and iters<15:
                verr = ELECTRICAL.compute(8)
                terr = THERMAL.compute(1)
                iters+=1
        DIFFUSION.compute_threshold()
        det_lams = linspace(OPTICAL.lam0-2, OPTICAL.lam0+2, 401)+0.2j*(voltage-0.5)/1.5
        det_vals = abs(OPTICAL.get_determinant(det_lams, m=0))
        det_mins = np.r_[Fplot_geometry(GEO.GeoTE, margin=0.01)
alse, det_vals[1:] < det_vals[:-1]] & \
                   np.r_[det_vals[:-1] < det_vals[1:], False] & \
                   np.r_[det_vals[:] < 1]
        mode_number = OPTICAL.find_mode(max(det_lams[det_mins]))
        mode_loss = OPTICAL.outLoss(mode_number)
        print_log(LOG_RESULT,
            'V = {:.3f}V, I = {:.3f}mA, lam = {:.2f}nm, loss = {}/cm'
            .format(voltage, ELECTRICAL.get_total_current(), OPTICAL.outWavelength(mode_number), mode_loss))
        return mode_loss


OPTICAL.lam0 = 981.5
OPTICAL.vat = 0

threshold_voltage = scipy.optimize.brentq(loss_on_voltage, 0.5, 2., xtol=0.01)
loss_on_voltage(threshold_voltage)
threshold_current = abs(ELECTRICAL.get_total_current())
print_log(LOG_WARNING, "Vth = {:.3f}V    Ith = {:.3f}mA"
                       .format(threshold_voltage, threshold_current))

geometry_width = GEO.GeoO.bbox.upper[0]
geometry_height = GEO.GeoO.bbox.upper[1]
RR = linspace(-geometry_width, geometry_width, 200)
ZZ = linspace(0, geometry_height, 500)
intensity_mesh = mesh.Rectangular2D(RR, ZZ)

IntensityField = OPTICAL.outLightMagnitude(len(OPTICAL.outWavelength)-1, intensity_mesh)
figure()
plot_field(IntensityField, 100)
plot_geometry(GEO.GeoO, mirror=True, color="w")
gcf().canvas.set_window_title('Light Intensity Field ({0} micron aperture)'.format(GEO["aperture"].dr))
axvline(x=GEO["aperture"].dr, color='w', ls=":", linewidth=1)
axvline(x=-GEO["aperture"].dr, color='w', ls=":", linewidth=1)
xticks(append(xticks()[0], [-GEO["aperture"].dr, GEO["aperture"].dr]))
xlabel(u"r [\xb5m]")
ylabel(u"z [\xb5m]")


new_aperture = 3.
GEO["aperture"].dr = new_aperture
GEO["oxide"].dr = DEF["mesaRadius"] - new_aperture

OPTICAL.lam0=982.
threshold_voltage = scipy.optimize.brentq(loss_on_voltage, 0.5, 2., xtol=0.01)
loss_on_voltage(threshold_voltage)
threshold_current = abs(ELECTRICAL.get_total_current())
print_log(LOG_WARNING, "Vth = {:.3f}V    Ith = {:.3f}mA"
                       .format(threshold_voltage, threshold_current))

IntensityField = OPTICAL.outLightMagnitude(len(OPTICAL.outWavelength)-1, intensity_mesh)
figure()
plot_field(IntensityField, 100)
plot_geometry(GEO.GeoO, mirror=True, color="w")
gcf().canvas.set_window_title('Light Intensity Field ({0} micron aperture)'.format(GEO["aperture"].dr))
axvline(x=GEO["aperture"].dr, color='w', ls=":", linewidth=1)
axvline(x=-GEO["aperture"].dr, color='w', ls=":", linewidth=1)
show()
