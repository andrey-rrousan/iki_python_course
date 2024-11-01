import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from matplotlib.colors import SymLogNorm


def polynom(en, c0, c1, c2):
    return (c0 + c1 * en + c2 * en * en) / en ** 3 / 1e24


table = np.array(
          [(0.03 ,  0.1  ,  17.3,  608.1, -2.150e+03),
           (0.1  ,  0.284,  34.6,  267.9, -4.761e+02),
           (0.284,  0.4  ,  78.1,   18.8,  4.300e+00),
           (0.4  ,  0.532,  71.4,   66.8, -5.140e+01),
           (0.532,  0.707,  95.5,  145.8, -6.110e+01),
           (0.707,  0.867, 308.9, -380.6,  2.940e+02),
           (0.867,  1.303, 120.6,  169.3, -4.770e+01),
           (1.303,  1.84 , 141.3,  146.8, -3.150e+01),
           (1.84 ,  2.471, 202.7,  104.7, -1.700e+01),
           (2.471,  3.21 , 342.7,   18.7,  0.000e+00),
           (3.21 ,  4.038, 352.2,   18.7,  0.000e+00),
           (4.038,  7.111, 433.9,   -2.4,  7.500e-01),
           (7.111,  8.331, 629. ,   30.9,  0.000e+00),
           (8.331, 10.   , 701.2,   25.2,  0.000e+00)],
dtype=[('en_min', '<f8'), ('en_max', '<f8'), ('c0', '<f8'), ('c1', '<f8'), ('c2', '<f8')]
)


def nH_cross(en):
    out = np.empty(en.shape)
    poly_idx = np.argmax(np.logical_and((en.reshape(-1, 1) >= table['en_min']), (en.reshape(-1, 1) < table['en_max'])), 1)
    for idx in range(len(out)):
        curr_idx = poly_idx[idx]
        out[idx] = polynom(en[idx], table['c0'][curr_idx], table['c1'][curr_idx], table['c2'][curr_idx])
    return out


def zphabs(en: np.ndarray, nH: float, z: float) -> np.ndarray:
    return np.clip(np.exp(-nH * 1e22 * nH_cross(en * (1 + z))), 0., 1.)


def phabs(en: np.ndarray, nH: float) -> np.ndarray:
    return zphabs(en, nH, z=0.0)


def zpowerlaw(en: np.ndarray, Gamma: float, norm: float, z: float) -> np.ndarray:
    return norm * (en * (1 + z)) ** (-Gamma)


def powerlaw(en: np.ndarray, Gamma: float, norm: float) -> np.ndarray:
    return zpowerlaw(en, Gamma, norm, z=0.0)


class Observation:
    def __init__(self, src_spec, arf, rmf, bkg_spec):
        self.src = fits.getdata(src_spec, 1)
        
        self.arf = fits.getdata(arf, 1)
        
        self.rmf = fits.getdata(rmf, 1)
        self.rmf_proxy = fits.getdata(rmf, 2)

        self.bkg = fits.getdata(bkg_spec, 1)

        src_header = fits.getheader(src_spec, 1)
        bkg_header = fits.getheader(bkg_spec, 1)
        self.exposure = src_header['EXPOSURE']
        bkg_exposure = bkg_header['EXPOSURE']
        # self.src_reg = fits.getdata(src_spec, 2)
        # self.bkg_reg = fits.getdata(bkg_spec, 2)
        # self.bkg_to_src = self._calc_src_bkg_relation()
        self.bkg_to_src = self.exposure * src_header['BACKSCAL'] / (bkg_exposure * bkg_header['BACKSCAL'])
    
    def _calc_area(self, reg):
        min_x = int(np.min(reg['X'] - reg['R'].max()))
        max_x = int(np.max(reg['X'] + reg['R'].max()))
        min_y = int(np.min(reg['y'] - reg['R'].max()))
        max_y = int(np.max(reg['y'] + reg['R'].max()))

        field = np.zeros((max_y-min_y, max_x - min_x), dtype=bool)
        x, y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
        
        for curr_region in reg:
            reg_type = curr_region['SHAPE']
            circle_x = curr_region['X']
            circle_y = curr_region['Y']
            circle_rad = curr_region['R']

            if reg_type == 'CIRCLE':
                field[((x - circle_x) ** 2 + (y - circle_y) ** 2) <= circle_rad ** 2] = True
            if reg_type == '!CIRCLE':
                field[((x - circle_x) ** 2 + (y - circle_y) ** 2) <= circle_rad ** 2] = False
            if reg_type == 'ANNULUS':
                field[((x - circle_x) ** 2 + (y - circle_y) ** 2) <= circle_rad.max() ** 2] = True
                field[((x - circle_x) ** 2 + (y - circle_y) ** 2) <= circle_rad.min() ** 2] = False

        area = np.sum(field)
        
        return area
    
    def _calc_src_bkg_relation(self):
        src_area = self._calc_area(self.src_reg)
        bkg_area = self._calc_area(self.bkg_reg)

        return src_area / bkg_area

    def group_bin(self, counts=1):

        self._recalc_cha_bins(counts)

        self.en = (self._en_max + self._en_min) / 2
        self.en_step = (self._en_max - self._en_min) / 2

        self.group_src = self._rebin_spec(self.src)
        self.group_bkg = self._rebin_spec(self.bkg)
        
        self.group_arf = self._rebin_arf()

    def _rebin_spec(self, spec_data):
        group_cts = []

        for start, stop in zip(self._cha_start, self._cha_stop):
            group_cts.append(sum(spec_data['COUNTS'][start:stop]))
        
        group_cts = np.array(group_cts)
        return group_cts 

    def _recalc_cha_bins(self, counts):
        cha_start = []
        cha_stop = []

        accum_cts = 0
        curr_cha = 0
        first_switch = True
        for cha, cts, qual in zip(self.src['CHANNEL'], self.src['COUNTS'], self.src['QUALITY']):
            if qual != 0:
                continue
            if accum_cts == 0 and first_switch:
                cha_start.append(cha)
                first_switch = False
            curr_cha = cha
            accum_cts += cts
            
            if accum_cts >= counts:
                cha_stop.append(cha+1)
                accum_cts = 0
                first_switch = True
        if cha_stop[-1] != curr_cha:
            cha_stop.append(curr_cha)

        self._cha_start = np.array(cha_start)
        self._cha_stop = np.array(cha_stop)

        self._en_min = self.rmf_proxy['E_MIN'][self._cha_start]
        self._en_max = self.rmf_proxy['E_MAX'][self._cha_stop]
    
    def _rebin_arf(self):
        arf_bins =  ((self.arf['ENERG_LO'] >= self._en_min.reshape(-1, 1)) * (self.arf['ENERG_LO'] < self._en_max.reshape(-1, 1)))
        arf_resp = []
        for curr_bin in arf_bins:
            arf_resp.append(self.arf['SPECRESP'][curr_bin].mean())
        return np.array(arf_resp)
    
    def plot_arf(self):

        plt.close()

        plt.plot(self.arf['ENERG_LO'], self.arf['SPECRESP'])

        plt.xlabel('Energy, keV')
        plt.ylabel(r'Effective area, $\rm cm^{2} \times cnts / photon$')

        plt.xscale('log')

        plt.show()

    def _create_rmf_matrix(self):
        rmf_matrix = np.zeros((len(self.rmf), self.src['CHANNEL'][-1]+1))

        for i in range(len(self.rmf)):
            chosen_channel = np.zeros(4096, dtype=bool)
            for f_chan, n_chan in zip(self.rmf['F_CHAN'][i], self.rmf['N_CHAN'][i]):
                chosen_channel[f_chan:f_chan + n_chan] = True

            rmf_matrix[i][chosen_channel] = self.rmf['MATRIX'][i]

        self._rmf_matrix = rmf_matrix

    def plot_rmf(self):
        if not hasattr(self, '_rmf_matrix'):
            self._create_rmf_matrix()

        plt.close()

        plt.pcolormesh([*self.rmf['ENERG_LO'], self.rmf['ENERG_HI'][-1]],
                       range(-1, self.src['CHANNEL'][-1]+1),
                       self._rmf_matrix.T,
                       norm=SymLogNorm(1e-4, vmin=0, vmax=0.1))
        
        plt.xlabel('Energy, keV')
        plt.ylabel('CHANNEL')

        plt.colorbar()

        plt.show()

