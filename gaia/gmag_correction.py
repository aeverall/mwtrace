import numpy as np

def correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag):
    """
    Correct the G-band fluxes and magnitudes for the input list of Gaia EDR3 data.

    Parameters
    ----------

    bp_rp: float, numpy.ndarray
        The (BP-RP) colour listed in the Gaia EDR3 archive.
    astrometric_params_solved: int, numpy.ndarray
        The astrometric solution type listed in the Gaia EDR3 archive.
    phot_g_mean_mag: float, numpy.ndarray
        The G-band magnitude as listed in the Gaia EDR3 archive.

    Returns
    -------

    The corrected G-band magnitudes and fluxes. The corrections are only applied to
    sources with a 6-parameter astrometric solution fainter than G=13, for which a
    (BP-RP) colour is available.

    Example
    -------

    gmag_corr, gflux_corr = correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag, phot_g_mean_flux)
    """

    do_not_correct = np.isnan(bp_rp) | (phot_g_mean_mag<=13) | (astrometric_params_solved != 95)
    bright_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>13) & (phot_g_mean_mag<=16)
    faint_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>16)
    bp_rp_c = np.clip(bp_rp, 0.25, 3.0)

    correction_factor = np.ones_like(phot_g_mean_mag)
    correction_factor[faint_correct] = 1.00525 - 0.02323*bp_rp_c[faint_correct] + \
        0.01740*np.power(bp_rp_c[faint_correct],2) - 0.00253*np.power(bp_rp_c[faint_correct],3)
    correction_factor[bright_correct] = 1.00876 - 0.02540*bp_rp_c[bright_correct] + \
        0.01747*np.power(bp_rp_c[bright_correct],2) - 0.00277*np.power(bp_rp_c[bright_correct],3)

    gmag_corrected = phot_g_mean_mag - 2.5*np.log10(correction_factor)

    return gmag_corrected
