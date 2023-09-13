## Library for OCT tools
import numpy as np
from scipy.signal import hilbert, gaussian, get_window, bessel, butter, cheby1, filtfilt
from scipy.interpolate import interp1d
from typing import Iterable, Tuple


class Mzi():
    ''' A class to create an MZI
        Args:
            delta_l: length difference in air (mm)
            input_ratio: input coupler split ratio (fraction, i.e. 0.4 for 40%/60%)
            output_ratio: input coupler split ratio (fraction, i.e. 0.5 for 50%/50%)
    '''

    def __init__(self, delta_l: float = 1e-3, input_ratio: float = 0.5, output_ratio: float = 0.5):
        self.delta_l = max(delta_l, 1e-12)  # Length difference between arms. Enforce minimum of 1 pm
        self.input_ratio = input_ratio  # Split ratio at the input
        self.output_ratio = output_ratio  # Split ratio at the output
        self.fsr = 3e8 / self.delta_l
        self.time_delay = 1 / self.fsr

    def __str__(self):
        #s = f'Length Difference: {self.delta_l}\nInput Ratio: {self.input_ratio * 100}% / {self.input_ratio * 100}%\nOutput Ratio: {self.output_ratio}'
        header = ''
        rows = ''
        spacing = 15
        for field_name, field_val in self.__dict__.items():
            header += f"{field_name : ^{spacing}}"
            rows += f"{str(field_val): ^{spacing}}"

        return f"{header}\n{'-' * ((spacing) * len(self.__dict__) + 3)}\n{rows}"
        # return s


    def calculate(self, wavelength: Iterable[float], power_in: Iterable[complex]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate the output amplitudes and free spectral range of a Mach-Zehnder Interferometer.

        Parameters:
            wavelength (Iterable[float]): Iterable of input wavelengths.
            power_in (Iterable[complex]): Iterable of input power.
            
        Returns:
            tuple: A tuple containing the output amplitudes power_out_1 and power_out_2
        """
        c = 3e8
        f_light = c / np.array(wavelength)

        dphi = 2 * np.pi * f_light / self.fsr # phase shift due to length difference
        phi = 0 # NOTE: can add nominal short path length to b1 and b2

        a1 = np.sqrt(np.divide(np.array(power_in), 1e3))  # MZI input E-field

        # input coupler
        # TODO: sqrt of ratio?
        b1 = self.input_ratio * a1 # short path
        b2 = (1 - self.input_ratio) * a1 * np.exp(-1j * np.pi / 2) # long/delayed path (arbitrarily chosen to get the pi/2 phase shift)

        # add path length delays
        c1 = b1 * np.exp(-1j * phi) # short path
        c2 = b2 * np.exp(-1j * (phi + dphi)) # long/delayed path gets length difference phase shift here (dphi)

        # calculate output E-fields
        e_field_out_1 = self.output_ratio * (c1 + c2 * np.exp(-1j * np.pi / 2))
        e_field_out_2 = (1 - self.output_ratio) * (c1 * np.exp(-1j * np.pi / 2) + c2)

        power_out_1 = np.square(np.abs(e_field_out_1))
        power_out_2 = np.square(np.abs(e_field_out_2))

        return power_out_1, power_out_2

    def calculate_from_time(self, t: Iterable[float], wavelength: Iterable[float], power_in: Iterable[complex]) -> Tuple[np.ndarray, np.ndarray, float]:
            """
            Calculate the output amplitudes and free spectral range of a Mach-Zehnder Interferometer.

            Parameters:
                t (Iterable[float]): Iterable of time (sec).
                wavelength (Iterable[float]): Iterable of input wavelengths (m).
                power_in (Iterable[complex]): Iterable of input power (mW).
                
            Returns:
                tuple: A tuple containing time, the output amplitudes power_out_1 and power_out_2.
            """
            c = 3e8            

            # 2 power arrays, one time delayed for short path, one for long path
            # use dt to shift array
            # TODO: improve with interpolation instead of shifting
            _, idx = find_nearest(t, self.time_delay)
            
            npts = len(t) # store number of points in starting array to make outputs the same size
            t = t[idx:] # start time when interference begins (when long path arrives at output coupler)
            # wavelength = wavelength[:len(t)] # make wavelengths for both arms
            # don't shift power until after input coupler. apply shift to b1 and b2

            f_light = c / np.array(wavelength)
            dphi = 2 * np.pi * f_light / self.fsr # phase shift due to length difference
            phi = 0 # NOTE: can add nominal short path length to b1 and b2

            a1 = np.sqrt(np.divide(np.array(power_in), 1e3))  # MZI input E-field
            
            # input coupler
            # TODO: sqrt of ratio?
            b1 = self.input_ratio * a1 # short path
            b2 = (1 - self.input_ratio) * a1 * np.exp(-1j * np.pi / 2) # long/delayed path (arbitrarily chosen to get the pi/2 phase shift)
            # also apply time shift to long path array before applying phase shift due to length??
            # b2 = b2[idx:]
            # b1 = b1[:len(b2)] # need to trim b1 and wavelength to b2

            # add path length delays
            c1 = b1 * np.exp(-1j * phi) # short path
            c2 = b2 * np.exp(-1j * (phi + dphi)) # long/delayed path gets length difference phase shift here (dphi)
            # c2 = c2[idx:]
            # c1 = c1[:len(c2)] # need to trim b1 and wavelength to b2

            # calculate output E-fields
            e_field_out_1 = self.output_ratio * (c1 + c2 * np.exp(-1j * np.pi / 2))
            e_field_out_2 = (1 - self.output_ratio) * (c1 * np.exp(-1j * np.pi / 2) + c2)
            # NOTE: Applying time delay here gives the expected result. But is that correct???
            e_field_out_2 = e_field_out_2[idx:]
            e_field_out_1 = e_field_out_1[:len(e_field_out_2)] # need to trim b1 and wavelength to b2

            power_out_1 = np.square(np.abs(e_field_out_1))
            power_out_2 = np.square(np.abs(e_field_out_2))

            # resize
            t = resize_interpolate(t, npts)
            power_out_1 = resize_interpolate(power_out_1, npts)
            power_out_2 = resize_interpolate(power_out_2, npts)

            return t, power_out_1, power_out_2
            # return t, np.abs(c1), np.abs(c2)

class Michelson():
    ''' A class to create a Michelson Interferometer
        Args:
            delta_l: length difference in air (mm)
            input_ratio: input coupler split ratio (fraction, i.e. 0.4 for 40%/60%)
            output_ratio: input coupler split ratio (fraction, i.e. 0.5 for 50%/50%)
    '''
    '''
                                Reference Mirror
                              ---------------------
                                        |
                                        |
                                        |  ----
                                        |/      \
                                        |        | delta_l (mm)
                                        |\      /
                                        |  ----
                                        |
                                        |
                      __C1__      ______|______          ____C2____
          michIn      | 0.5|------ Circulator |----------|0.5 0.5 |-------michOut1
        ------------->|1:  |      ------------|          |   :    | 
                      | 0.5|------ Circulator |----------|0.5 0.5 |-------michOut2
                      ------      -------------          ----------
                                        |
                                        |
                                        |
                                        |
                                   ------------
                                ---            ---                      
                                   ------------
                                     |     |
                                     |     |
                                     |     |     
                                     |     |    
                                     |     |   
                          |          |     |  /
                          |          |     | /
                        |___|________|_____|/
        --------<      /|   |        |     /
                 <   /  |   |        |    /
                  >/    |   |        |   /
          Sample  |\    |   |        |  /
                  <  \  |   |        | / 
                  |>   \|___|________|/ 
        __________|     |   |        / 
                          |         /
                          |
    '''

    def __init__(self, delta_l: float, input_ratio: float = 0.5, output_ratio: float = 0.5):
        self.delta_l = delta_l  # Length difference between arms
        self.input_ratio = input_ratio  # Split ratio at the input
        self.output_ratio = output_ratio  # Split ratio at the output

    def __str__(self):
        #s = f'Length Difference: {self.delta_l}\nInput Ratio: {self.input_ratio * 100}% / {self.input_ratio * 100}%\nOutput Ratio: {self.output_ratio}'
        header = ''
        rows = ''
        spacing = 15
        for field_name, field_val in self.__dict__.items():
            header += f"{field_name : ^{spacing}}"
            rows += f"{str(field_val): ^{spacing}}"

        return f"{header}\n{'-' * ((spacing) * len(self.__dict__) + 3)}\n{rows}"
        # return s


    def calculate(self, wavelength: Iterable[float], power_in: Iterable[complex]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate the output amplitudes and free spectral range of a Mach-Zehnder Interferometer.

        Parameters:
            wavelength (Iterable[float]): Iterable of input wavelengths.
            power_in (Iterable[complex]): Iterable of input power.
            
        Returns:
            tuple: A tuple containing the output amplitudes power_out_1 and power_out_2, free spectral range (fsr), and time difference between paths(sec).
        """
        c = 3e8
        f_light = c / np.array(wavelength)

        try:
            fsr = c / self.delta_l
        except ZeroDivisionError:
            fsr = c / 0.01 # set minimum length
        dt = 1 / fsr
        dphi = 2 * np.pi * f_light / fsr # phase shift due to length difference
        phi = 0 # NOTE: can add nominal short path length to b1 and b2

        a1 = np.sqrt(np.array(power_in))  # MZI input E-field

        # input coupler
        b1 = self.input_ratio * a1 # short path
        b2 = (1 - self.input_ratio) * a1 * np.exp(-1j * np.pi / 2) # long/delayed path (arbitrarily chosen to get the pi/2 phase shift)

        # add path length delays
        c1 = b1 * np.exp(-1j * phi) # short path
        c2 = b2 * np.exp(-1j * (phi + dphi)) # long/delayed path gets length difference phase shift here (dphi)

        # calculate output E-fields
        e_field_out_1 = self.output_ratio * (c1 + c2 * np.exp(-1j * np.pi / 2))
        e_field_out_2 = (1 - self.output_ratio) * (c1 * np.exp(-1j * np.pi / 2) + c2)

        power_out_1 = np.square(np.abs(e_field_out_1))
        power_out_2 = np.square(np.abs(e_field_out_2))

        return power_out_1, power_out_2, fsr, dt
    
class Bpd():
    ''' A class to create a balanced photo-detector
        Args:
            delta_l: length difference in air (mm)
            input_ratio: input coupler split ratio (fraction, i.e. 0.4 for 40%/60%)
            output_ratio: input coupler split ratio (fraction, i.e. 0.5 for 50%/50%)
    '''

    def __init__(self, gain: float = 5000, bandwidth: float = 500e6, i_noise: float = 1e-12, filter_type: str = 'bessel', filter_order: int = 1):
        self.gain = gain  # Gain in V/A
        self.i_noise = i_noise  # Input-referred current noise (A/rtHz)
        self.bandwidth = bandwidth # Detection bandwidth
        self.filter_type = filter_type
        self.filter_order = int(filter_order)

    def __str__(self):
        #s = f'Length Difference: {self.delta_l}\nInput Ratio: {self.input_ratio * 100}% / {self.input_ratio * 100}%\nOutput Ratio: {self.output_ratio}'
        header = ''
        rows = ''
        spacing = 15
        for field_name, field_val in self.__dict__.items():
            header += f"{field_name : ^{spacing}}"
            rows += f"{str(field_val): ^{spacing}}"

        return f"{header}\n{'-' * ((spacing) * len(self.__dict__) + 3)}\n{rows}"
        # return s


    def detect(self, t: np.ndarray, pin_p: np.ndarray, pin_n: np.ndarray, resp_p: float = 0.9, resp_n: float = 0.9) -> tuple:
        """
        Calculate the detector output voltage and output RMS voltage noise for a balanced photodetector.
        
        Parameters:
            pin_p (np.ndarray): Input power vector for positive-side detector.
            pin_n (np.ndarray): Input power vector for negative-side detector.
            resp_p (float): Responsivity of detector 1.
            resp_n (float): Responsivity of detector 2.
            bandwidth (float): Bandwidth (Hz).
            input_noise_rms (float): RMS input noise level (Vrms).
            
        Returns:
            tuple: A tuple containing the detector output voltage (Vout)
        """
        i_in = resp_p * pin_p - resp_n * pin_n

        # Need to scale properly for RMS input and bandwidth
        input_noise = self.i_noise * np.random.randn(len(pin_p)) * np.sqrt(self.bandwidth)

        i_in = i_in + input_noise

        v_out = i_in * self.gain

        # Apply filter
        fs = 1 / (t[1] - t[0])
        match self.filter_type:
            case 'bessel':
                b, a = bessel(int(self.filter_order), self.bandwidth / (fs / 2), btype='low')
            case 'butterworth':
                b, a = butter(int(self.filter_order), self.bandwidth / (fs / 2), btype='low')
            case 'chebychev':
                b, a = cheby1(int(self.filter_order), 0.1, self.bandwidth / (fs / 2), btype='low')

        v_out = filtfilt(b, a, v_out)
        
        return v_out

class LightSource():
    """
    A class representing a light source.

    Attributes:
        p_ave (float): The average power of the light source in watts (default: 30e-3).
        wavelength_start (float): The starting wavelength of the light source in meters (default: 1260e-9).
        wavelength_stop (float): The stop wavelength of the light source in meters (default: 1360e-9).
        sweep_rate (float): The sweep rate of the light source in Hz (default: 100e3).
        duty_cycle (float): The duty cycle of the light source (default: 0.6).
        reflection_depth (float): The depth of reflection in the source in meters (default: 1e-3).
        reflection_strength (float): The strength of reflection in the source in dBc (default: 0).
    """

    def __init__(self, p_ave: float = 30e-3, wavelength_start: float = 1260e-9, wavelength_stop: float = 1360e-9, sweep_rate: float = 100e3, duty_cycle: float = 0.65, reflection_depth: float = 1e-3, reflection_strength: float = 0):
        self.p_ave = p_ave
        self.wavelength_start = wavelength_start
        self.wavelength_stop = wavelength_stop
        self.cwl = (wavelength_start + wavelength_stop) / 2
        self.sweep_rate = sweep_rate
        self.duty_cycle = duty_cycle
        self.reflection_depth = reflection_depth
        self.reflection_strength = reflection_strength

    def generate_vectors(self, f_samp: float = 10e9) -> np.ndarray:
        t_total = np.arange(0, 1 / self.sweep_rate, 1 / f_samp)
        on_idxs = np.where(t_total <= (self.duty_cycle / self.sweep_rate))
        t_on = t_total[on_idxs[0][0] : on_idxs[0][-1]] # on time
        len_t_diff = len(t_total) - len(t_on)
        wavelength = np.append(np.linspace(self.wavelength_start, self.wavelength_stop, len(t_on)), np.linspace(self.wavelength_stop, self.wavelength_start, len_t_diff))
        power = gaussian(len(t_on), std=len(t_on) / 5.5) + 0.1
        power = np.append(power, [0] * len_t_diff)
        power = power / max(power)
        power = power * self.p_ave / np.mean(power) / self.duty_cycle # scale envelope
        return t_total, wavelength, power

############################## FUNCTIONS #####################################

def get_crop_indices(power_envelope, threshold: float):
    ''' Get crop indices from power envelope and threshold
        Threshold in decimal, not percent
    '''
    power_envelope = np.divide(np.array(power_envelope), np.max(power_envelope)) # normalize envelope
    idx = np.where(power_envelope >= threshold) 

    return idx[0][0], idx[0][-1]

def calculate_fsr_wavelength(cwl, fsr_frequency, n=1):
    ''' Calculated FSR in wavelength from FSR in frequency

        Args:
            cwl is center wavelength (nm)
            fsr_frequency is fsr in Hz
            n is index of refraction
    '''
    velocity = 3e8 / n
    return np.mean(cwl) - velocity/ (velocity / np.mean(cwl) + fsr_frequency)

def apply_hanning(signal):
    # apply a hanning window to the data
    window = get_window('hann', len(signal))
    return signal * window

def resample(signal, reference, offset_rad=0.0):
    # resample signal along a vector derived from the phase progession of reference
    ref_phase = np.unwrap(np.angle(hilbert(reference)))
    kvector = np.linspace(ref_phase[0], ref_phase[-1], len(ref_phase))
    kvector += offset_rad
    return interp1d(ref_phase, signal, kind='linear')(kvector)

def find_nearest(array, value) -> tuple[float, int]:
    '''Finds element in array closest to value
        Returns nearest_element_value, index
    '''
    array = np.array(array)
    idx = np.abs(array - value).argmin()

    return array[idx], idx

def find_max_interference(mzi: Mzi, cwl: float) -> float:
    ''' Finds the wavelength for maximum interference power for given MZI

        Args:
            mzi object
            cwl - center wavelength (nm)

        Returns:
            wavelength - float
    '''
    # sweeps one FSR of the MZI, and finds the max power difference of the output arms
    # the found point has all power in one arm
    npts = int(1e4)
    wl = np.linspace(cwl, cwl + calculate_fsr_wavelength(cwl, mzi.fsr), npts)
    pout_p, pout_n = mzi.calculate(wl, [1] * npts)
    pout_diff = pout_p - pout_n
    _, idx = find_nearest(pout_diff, max(abs(pout_diff)))

    return wl[idx]

def find_quadrature(mzi: Mzi, cwl: float) -> float:
    ''' Finds the wavelength for quadrature for the given MZI

        Args:
            mzi object
            cwl - center wavelength (nm)

        Returns:
            wavelength - float
    '''
    # sweeps one FSR of the MZI, and finds the min power difference of the output arms
    # the found point has equal power in both arms

    # TODO: does this work if coupler ratio isn't 50/50?
    npts = int(1e4)
    wl = np.linspace(cwl, cwl + calculate_fsr_wavelength(cwl, mzi.fsr), npts)
    pout_p, pout_n = mzi.calculate(wl, [1] * npts)
    pout_diff = pout_p - pout_n
    _, idx = find_nearest(pout_diff, min(abs(pout_diff)))

    return wl[idx]

def normalize(x):
    return x / np.max(x)

def resize_interpolate(array, num_points):
    ''' Increases number of points of array with interpolation
        input array
        number of points in new array
    '''
    # Create an array of indices for the original data
    original_indices = np.arange(len(array))
    # Create an array of indices for the interpolated data
    interpolated_indices = np.linspace(0, len(array) - 1, 100_000)
    # interpolate
    return np.interp(interpolated_indices, original_indices, array)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    ls = LightSource(duty_cycle=0.65)
    t, wl, p = ls.generate_vectors()
    # plt.plot(t, p)
    plt.plot(t, wl)
    plt.show()