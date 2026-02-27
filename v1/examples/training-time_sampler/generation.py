"""
Simplified generation pipeline intended for the Artemisa computation cluster  (Large Datasets)
"""
import bilby.gw.prior
# Base imports
import numpy as np

from gen_utils import setup_ifos, ifo_q_transform, handle_time, getsnr, handle_snr

def create_image(theta,
                 parameter_labels,
                 target_snr_prior,
                default_prior,
                 sampling_frequency=1024,
                 waveform_generator=None,
                 ifos: tuple[str, ...] = None,
                 asds=None,
                 duration=None,
                 img_res: tuple[int, int] = (128, 128),
                 snr_tol: float = 0.5,
                 qtrans_kwargs: dict = None,

                 return_both: bool = False):
    """
    Generates image from individual sample of the parameters.

    Parameters
    ----------
    prior_sample :
        Sample of the parameters.

    sampling_frequency :
        Sampling frequency for the intermediate timeseries.

    waveform_generator :
        Waveform generator to model the signal.

    ifos :
        Collection of interferometer names.

    asds :
        Collection of ASDs for noise generation.

    duration :
        Duration for the intermediate timeseries.

    img_res :
        Image resolution (height, width) in pixels.

    snr_tol :
        Tolerance for the SNR adjusting algorithm. Accepts SNR \\in (SNR-snr_tol, SNR+snr_tol).

    qtrans_kwargs :
        Keyword arguments passed to gwpy's TimeSeries.q_transform.

    Returns
    -------
    out :
        (image, parameters)

    """

    assert waveform_generator is not None, 'Cannot inject signals without a waveform generator'
    assert duration == waveform_generator.duration, "Specified duration does not match waveform_generator.duration"

    if qtrans_kwargs is None:
        qtrans_kwargs = {}
    ts_data, params = generate_timeseries(theta,
                                          parameter_labels,
                                          target_snr_prior,
                                          default_prior,
                                          sampling_frequency=sampling_frequency,
                                          waveform_generator=waveform_generator,
                                          snr_tol=snr_tol,
                                          ifos=ifos,
                                          asds=asds)

    qt_data = [ifo_q_transform(tseries, img_res, duration, **qtrans_kwargs) for tseries in ts_data]

    if return_both:
        return ts_data, np.array(qt_data)

    return np.array(qt_data)


def generate_timeseries(theta,
                        parameter_labels,
                        target_snr_prior,
                        default_prior,
                        ifos,
                        asds,
                        sampling_frequency=1024,
                        waveform_generator=None,
                        snr_tol: float = 0.5
                        ):
    """
    Generate timeseries from parameters.

    Parameters
    ----------
    prior_sample :
        Sample of the parameters.
    ifos :
        Collection of interferometer names.
    asds :
        Collection of ASDs for noise generation.
    sampling_frequency :
        Sampling frequency for the intermediate TimeSeries.
    waveform_generator :
        Waveform generator to model the signal.
    snr_tol :
        Tolerance for the SNR adjusting algorithm. Accepts SNR \\in (SNR-snr_tol, SNR+snr_tol).

    Returns
    -------
    out :
        Timeseries with injected signal.

    """

    duration = waveform_generator.duration
    time = handle_time(theta, parameter_labels)

    prior_sample = default_prior.sample()
    if 'geocent_time' not in prior_sample.keys():
        prior_sample['geocent_time'] = time
    prior_sample.update({param: theta[i].numpy() for i, param in enumerate(parameter_labels)})

    # Set up the two interferometers.
    ifos = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifos, asds=asds,
                      time=time)

    # Save the background frequency domain strain data of the two interferometers if you always want the same noise.
    ifos_bck = [ifo.frequency_domain_strain for ifo in ifos]

    targ_snr = handle_snr(target_snr_prior)

    if targ_snr == -1:
        # Inject the signal into the noise-free background and return the resulting strain data.
        for ifo in ifos:
            ifo.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency,
                                                duration=duration,
                                                start_time=time - duration / 2)
        # prior_sample['geocent_time'] = 0.
        ifos.inject_signal(prior_sample, waveform_generator=waveform_generator)
        return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos], prior_sample
    else:
        if targ_snr == 0:  # Return as is
            ifos.inject_signal(prior_sample, waveform_generator=waveform_generator)
            return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos], prior_sample
        # Iteratively inject the signal and adjust the luminosity distance until the
        # injected signal has a SNR close to the target SNR.
        while True:
            # Reverted to old concept

            # Inject the signal into the background strain data and update the geocentric time of the signal.
            ifos_ = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifos, asds=asds)
            for i, ifo in enumerate(ifos_):
                ifo.set_strain_data_from_frequency_domain_strain(ifos_bck[i], sampling_frequency=sampling_frequency,
                                                                 duration=duration, start_time=duration / 2)
            ifos_.inject_signal(prior_sample, waveform_generator=waveform_generator)

            # Check if the SNR of the injected signal is close to the target SNR.
            new_snr = getsnr(ifos_)
            if abs(new_snr - targ_snr) < snr_tol:
                # Return the injected strain data for each interferometer.
                prior_sample['_snr'] = new_snr
                return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos_], prior_sample

            # Adjust the luminosity distance of the signal by the ratio of the current SNR to the target SNR.
            snr_ratios = np.array(getsnr(ifos_) / targ_snr)
            prior_sample['luminosity_distance'] = prior_sample['luminosity_distance'] * np.prod(snr_ratios)
