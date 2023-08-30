## Library for OCT tools
import numpy as np
from scipy.signal import hilbert, get_window
from scipy.interpolate import interp1d
from typing import Iterable, Tuple


class Mzi():
    ''' A class to create an MZI
        Args:
            delta_l: length difference in air (mm)
            input_ratio: input coupler split ratio (fraction, i.e. 0.4 for 40%/60%)
            output_ratio: input coupler split ratio (fraction, i.e. 0.5 for 50%/50%)
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

        return power_out_1, power_out_2, fsr, dt
    
    
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

    def __init__(self, gain: float = 5000, bandwidth: float = 500e6, i_noise: float = 1e-12):
        self.gain = gain  # Gain in V/A
        self.i_noise = i_noise  # Input-referred current noise (A/rtHz)
        self.bandwidth = bandwidth # Detection bandwidth

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


    def detect(self, pin_p: np.ndarray, pin_n: np.ndarray, resp_p: float = 0.9, resp_n: float = 0.9) -> tuple:
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
        
        return v_out


############################## FUNCTIONS #####################################

def get_crop_indices(power_envelope, threshold: float):
    ''' Get crop indices from power envelope and threshold
        Threshold in decimal, not percent
    '''
    power_envelope = np.divide(np.array(power_envelope), np.max(power_envelope)) # normalize envelope
    idx = np.where(power_envelope >= threshold) 

    return idx[0][0], idx[0][-1]

def calculate_fsr_wavelength(center_wavelength, fsr_frequency, n=1):
    ''' Calculated FSR in wavelength from FSR in frequency
    '''
    velocity = 3e8 / n
    return np.mean(center_wavelength) - velocity/ (velocity / np.mean(center_wavelength) + fsr_frequency)

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

