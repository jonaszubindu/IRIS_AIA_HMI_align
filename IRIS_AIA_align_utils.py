"""
This file contains the functions to align IRIS and AIA data. It is a collection of code from different libraries plus my own contributions. In the future we will try to modify the functions to create aligned movies and download whole HMI/AIA datasets for
further studies with solar flare forecasting. An example showing how the functions can be used is given below.

Author: Jonas Zbinden, github: @jonaszubindu, email: jonas.zbinden@unibe.ch
"""

# Import any necessary libraries

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from matplotlib.patches import Polygon
from matplotlib import colors as cs

from matplotlib.ticker import MultipleLocator

from sunpy.time import parse_time

from astropy.io import fits
import os
import subprocess
import warnings
import fnmatch


import drms
from datetime import datetime as dt_obj
from datetime import timedelta
from astropy.time import Time
from scipy.signal import correlate2d

import utils_models_MgIIk as mdls
import torch



def ASTROPY_FILE_METHOD( path ):
    """
    Astropy method to open a FITS file.
    """
    handle = fits.open( path, lazy_load_hdus=True )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        handle.verify('fix')

    return handle


def runcmd(cmd, verbose = False, *args, **kwargs):

    """
    Run terminal command from within python
    """

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def array2dict( header, data ):
    """
    Reads (key, index) pairs from the header of the extension and uses them
    to assign each row of the data array to a dictionary.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        Header with the keys to the data array
    data : numpy.ndarray
        Data array
    Returns
    -------
    list of header dictionaries
    """

    # some headers are not keys but real headers: remove them
    keys_to_remove=['XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'PCOUNT', 'GCOUNT']
    header_keys = dict( header )
    header_keys = {k: v for k, v in header_keys.items() if k not in keys_to_remove}

    # initialize dictionary list
    res = [dict() for x in range( data.shape[0] )]

    # fill dictionaries
    for i in range(0, data.shape[0]):
        res[i] = dict( zip( header_keys.keys(), data[i,list(header_keys.values())] ) )

    return res



class IRIS_SJI_cube:
    """
    Loads an IRIS SJI map from the IRIS archive and creates data, wcs, and sji_times, raster_times, and time ordered sji and raster headers objects.

    """
    def __init__(self, year, month, day, hour, minute, second, obsid, wave=1400):

        if month < 10:
            month = '0' + str(month)
        if day < 10:
            day = '0' + str(day)
        if hour < 10:
            hour = '0' + str(hour)
        if minute < 10:
            minute = '0' + str(minute)
        if second < 10:
            second = '0' + str(second)

        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
        self.obsid = obsid
        
        
        base_path = '/sml/iris/'
        path = base_path + str(year) + '/' + str(month) + '/' + str(day) + '/' + str(year) + str(month) + str(day) + '_' + str(hour) + str(minute) + str(second) + '_' + str(obsid) + '/'
        
        # Check which SJI images are avialable.
        sji_files_avail = [file for file in os.listdir(path) if 'SJI' in file]
        
        skip=False
        
        if wave == 1400:
            filename = f'iris_l2_{year}{month}{day}_{hour}{minute}{second}_{obsid}_SJI_{wave}_t000.fits'
            try:
                hdulist = ASTROPY_FILE_METHOD(path+filename)
                print(hdulist.info())
            except Exception:
                print(f"{wave} is not available, skipping")
                
                
        elif wave == 1330:
                
            filename = f'iris_l2_{year}{month}{day}_{hour}{minute}{second}_{obsid}_SJI_{wave}_t000.fits'
            try:
                hdulist = ASTROPY_FILE_METHOD(path+filename)
                print(hdulist.info())
            except Exception:
                print(f"{wave} is not available, skipping")
                
                
        elif wave == 2796:
                
            filename = f'iris_l2_{year}{month}{day}_{hour}{minute}{second}_{obsid}_SJI_{wave}_t000.fits'
            try:
                hdulist = ASTROPY_FILE_METHOD(path+filename)
                print(hdulist.info())
            except Exception:
                print(f"{wave} is not available, skipping")
                
        else:
            raise ValueError(f"{wave} is not available, stopping. Choose existing file")
            

        filename_raster_list = sorted([file for file in os.listdir(path) if fnmatch.fnmatch(file, '*_r*.fits')])
        num_of_files = len(filename_raster_list)

        num_of_ext = len(ASTROPY_FILE_METHOD(path+filename_raster_list[0]))

        hdrs_time_primary_raster = [(ASTROPY_FILE_METHOD(path+file))[0].header for file in filename_raster_list] # list of primary headers for each raster fits file
        hdrs_timespecific_raster = [(ASTROPY_FILE_METHOD(path+file))[num_of_ext-2] for file in filename_raster_list] # time specific additional headers or raster
        
        # get line number for Mg raster
        for i in range(num_of_ext-2):
            if "Mg II" in (ASTROPY_FILE_METHOD(path+filename_raster_list[0]))[0].header['TDESC'+str(i+1)]:
                line_num = i+1
                break
        
        hdrs_time_linespec_raster = [(ASTROPY_FILE_METHOD(path+file))[line_num].header for file in filename_raster_list] # list of line specific headers for each raster fits file


        # time specific raster headers
        raster_header = []

        # extract headers from each raster file and combine to single header, the line specific data is only used from the first line (whichever that is)
        for hdu_t, hdu_p, hdu_l in zip(hdrs_timespecific_raster, hdrs_time_primary_raster, hdrs_time_linespec_raster):

            # check if length of timespecific header is same as lenght of header:
            assert hdu_t.data.shape[0] == hdu_l['NAXIS3'], 'Length of time specific header is not equal to length header file'

            # add wavelnth, wavename, wavemin and wavemax (without loading primary headers)
            hdu_p['WAVELNTH'] = hdu_p['TWAVE'+str(line_num)]
            hdu_p['WAVENAME'] = hdu_p['TDESC'+str(line_num)]
            hdu_p['WAVEMIN'] =  hdu_p['TWMIN'+str(line_num)]
            hdu_p['WAVEMAX'] =  hdu_p['TWMAX'+str(line_num)]
            hdu_p['WAVEWIN'] =  hdu_p['TDET'+str(line_num)]

            hdr_tim_per_file = array2dict( hdu_t.header, hdu_t.data )
            hdr_prim_per_file = [dict(hdu_p + hdu_l)]*hdu_t.data.shape[0]
            header_per_file = []
            for hdr_p, hdr_t in zip(hdr_prim_per_file, hdr_tim_per_file):
                hdr_comb = hdr_p.copy()
                hdr_comb.update(hdr_t)
                header_per_file.append(hdr_comb)

            raster_header.extend(header_per_file)

        # add additional information to raster header
        for i in range( len(raster_header) ):
            raster_header[i]['XCEN'] = raster_header[i]['XCENIX'] 
            raster_header[i]['YCEN'] = raster_header[i]['YCENIX'] 
            raster_header[i]['CRVAL2'] = raster_header[i]['YCENIX'] 

            # set EXPTIME = EXPTIMEF in FUV and EXPTIME = EXPTIMEN in NUV 
            waveband = raster_header[i]['TDET'+str(line_num)][0]
            raster_header[i]['EXPTIME'] = raster_header[i]['EXPTIME'+waveband]

        # if only one raster file is present, the list is converted to a single header
        if len(raster_header) == 1:
            raster_header=raster_header[0]
        else:
            pass


        # time specific headers of SJI
        primary_header_sji = dict(hdulist[0].header)

        self.iris_coordinates_sji = iris_coordinates(hdulist[0].header, 'sji')
        header_coord = hdrs_time_primary_raster[0] + hdrs_time_linespec_raster[0]
        self.iris_coordinates_raster = iris_coordinates(header_coord, 'raster')
        

        # check if length of timespecific header is same as lenght of header:
        assert hdulist[1].data.shape[0] == primary_header_sji['NAXIS3'], 'Length of time specific header is not equal to length header file'

        collected_time_specific_headers_sji = array2dict( hdulist[1].header, hdulist[1].data )

        sji_header = []
        for hdr_p, hdr_t in zip([primary_header_sji]*len(collected_time_specific_headers_sji), collected_time_specific_headers_sji):
            hdr_comb = hdr_p.copy()
            hdr_comb.update(hdr_t)
            hdr_comb['DATE_OBS'] = (Time(hdr_comb['STARTOBS'])+timedelta(seconds=hdr_t['TIME'])).iso
            sji_header.append(hdr_comb)

        # add additional information to sji header
        for i in range( len(sji_header) ):
            sji_header[i]['XCEN'] = sji_header[i]['XCENIX']
            sji_header[i]['YCEN'] = sji_header[i]['YCENIX']
            sji_header[i]['PC1_1'] = sji_header[i]['PC1_1IX']
            sji_header[i]['PC1_2'] = sji_header[i]['PC1_2IX']
            sji_header[i]['PC2_1'] = sji_header[i]['PC2_1IX']
            sji_header[i]['PC2_2'] = sji_header[i]['PC2_2IX']
            sji_header[i]['CRVAL1'] = sji_header[i]['XCENIX']
            sji_header[i]['CRVAL2'] = sji_header[i]['YCENIX']
            sji_header[i]['EXPTIME'] = sji_header[i]['EXPTIMES']


        times = Time(hdulist[0].header["STARTOBS"]) + TimeDelta(
                hdulist[1].data[:, hdulist[1].header["TIME"]], format="sec")

        self.sji_header = sji_header
        self.raster_header = raster_header

        raster_time = Time(self.raster_header[0]['STARTOBS']) + TimeDelta(np.array([self.raster_header[i]['TIME'] for i in range(len(self.raster_header))]), format='sec')
        self.raster_times = raster_time
        self.n_raster_pos = raster_header[0]['NRASTERP']
        
        sji_data = hdulist[0].data
        
        self.sji_times = times
        
        self.data = sji_data
        self.filename = filename
        self.path = path

        hdulist.close()
    def get_sunpy_wcs(self, loc):
        """ Returns the WCS object for the sunpy Map object created for SJI data at a given time step."""
        setattr(self, '_sunpy_wcs', sunpy.map.Map(self.data[loc], self.sji_header[loc]).wcs)
        setattr(self, '_sunpy_observer_coordinate', sunpy.map.Map(self.data[loc], self.sji_header[loc]).observer_coordinate)


    def raster_x_coord(self, step,plot_arr, mode='iris'):
        """
        Returns n_raster slit positions to coordinates for a given SJI index.
        """
        # work in pixel coordinates before
        sji_grid = np.array([[self.get_slit_pos(step)-1]*plot_arr.shape[1],np.arange(plot_arr.shape[1])]).T

        slitcoord_primary_x = sji_grid[:,0]
        slitcoord_primary_y = sji_grid[:,1]
        slit_offset_secondary_x = self.sji_header[step]['PZTX']/self.sji_header[step]['CDELT2'] # in pix coords

        # This assumes that the PZT offsets remain constant for the entire observation
        raster_offsets_secondary_x = np.asarray([np.asarray( [ self.raster_header[i]['PZTX']/self.raster_header[i]['CDELT2'] for i in range(self.n_raster_pos) ] )]*plot_arr.shape[1]) # in pix coords
        # slit x position for each raster pos = position of primary - wedge tilt + fine scale secondary pztx
        slpos_x = np.asarray([slitcoord_primary_x - slit_offset_secondary_x + rast_off_x for rast_off_x in raster_offsets_secondary_x.T]).T

        slposypix = np.array([np.arange(plot_arr.shape[1])]*sji.n_raster_pos).T
        slposxpix = slpos_x.reshape(slposypix.shape[0],slposypix.shape[1])

        slpos_pix = np.transpose(np.dstack([slposxpix, slposypix]), (1,0,2))

        if mode == 'iris':
            slpos = np.array([self.iris_coordinates_sji.pix2coords(step,  slpos_pix[i]) for i in range(self.n_raster_pos)])
        elif mode == 'hmi':
            slpos = np.array([self.pix2coords(step,  slpos_pix[i]) for i in range(sji.n_raster_pos)])
        else:
            raise ValueError('Mode not recognized. Use iris or hmi.')

        slpos_x = slpos[:,:,0]
        slpos_y = slpos[:,:,1]

        slpos_x = slpos_x.reshape(slpos_x.shape[0]*slpos_x.shape[1]) # flatten nested list
        slpos_y = slpos_y.reshape(slpos_y.shape[0]*slpos_y.shape[1]) # flatten nested list

        return slpos_x, slpos_y


    def pix2coords( self, timestep, pixel_coordinates ):
        """
        Function to convert from camera (pixel) coordinates to solar/physical coordinates.
        Makes heavy use of astropy.wcs.

        Parameters
        ----------
        timestep : int
            time step in the image cube
        pixel_coordinates : np.array
            numpy array with x and y coordinates

        Returns
        -------
        np.array :
            Tuple of solar x and y coordinates in Arcsec (SJI) or wavelength and solar y coordinates (raster) in Arcsec / Angstrom.
        """

        # make sure pixel_coordinates is a numpy array
        pixel_coordinates = np.array( pixel_coordinates )

        # check dimensions
        ndim = pixel_coordinates.ndim
        shape = pixel_coordinates.shape
        if not ( (ndim == 1 and shape[0] == 2) or (ndim == 2 and shape[1] == 2) ):
            raise ValueError( "pixel_coordinates should be a numpy array with shape (:,2)." )

        # create a copy of the input coordinates
        pixel_coordinates = pixel_coordinates.copy()

        # generalize for single pixel pairs
        if ndim == 1:
            pixel_coordinates = np.array([pixel_coordinates])

        # transform pixels to solar coordinates
        solar_coordinates = self._sunpy_wcs.all_pix2world( pixel_coordinates , 1 )

        # convert units
        solar_coordinates *= self.iris_coordinates_sji.conversion_factor

        # return tuple if input was only one tuple
        if ndim == 1:
            return solar_coordinates[0]
        else:
            return solar_coordinates


    def coords2pix( self, timestep, solar_coordinates, round_pixels=True ):
        """
        Function to convert from solar/physical coordinates to camera (pixel) coordinates.
        Makes heavy use of astropy.wcs.

        Parameters
        ----------
        timestep : int
            time step in the image cube
        solar_coordinates : np.array
            numpy array with solar coordinates (x,y) (SJI) or solar/wavelength coordinates (lambda,y) (raster)

        Returns
        -------
        np.array :
            Tuple (x,y) of camera coordinates in pixels
        """

        # make sure solar_coordinates is a numpy array
        solar_coordinates = np.array( solar_coordinates )

        # check dimensions
        ndim = solar_coordinates.ndim
        shape = solar_coordinates.shape
        if not ( (ndim == 1 and shape[0] == 2) or (ndim == 2 and shape[1] == 2) ):
            raise ValueError( "pixel_coordinates should be a numpy array with shape (:,2)." )

        # create a copy of the input coordinates
        solar_coordinates = solar_coordinates.copy()

        # generalize for single pixel pairs
        if ndim == 1:
            solar_coordinates = np.array([solar_coordinates])

        # convert units
        solar_coordinates = solar_coordinates / self.iris_coordinates_sji.conversion_factor

        # transform solar coordinates to pixels
        pixel_coordinates = self._sunpy_wcs.all_world2pix( solar_coordinates, 1 )

        # round to nearest pixels
        if round_pixels:
            pixel_coordinates = np.round( pixel_coordinates ).astype( np.int )

        # return tuple if input was only one tuple
        if ndim == 1:
            return pixel_coordinates[0]
        else:
            return pixel_coordinates


    def get_slit_pos( self, step ):
        """
        Returns position of the slit in pixels (takes into account cropping).

        Parameters
        ----------
        step : int
            Time step in the data cube.
        Returns
        -------
        slit_position : int
            Slit position in pixels
        """

        pos = self.sji_header[step]['SLTPX1IX']

        return pos


class IRIS_coaligned_AIA_images:

    """
    Looks up or downloads coaligned AIA images to IRIS. If you want to use the original AIA data, skip this part and go to HMI_AIA_single_image. This one works the same as IRIS_SJI_cube. It reates data, wcs, and times, and header objects.
    """

    def __init__(self, year, month, day, hour, minute, second, obsid, wavelength):

        if month < 10:
            month = '0' + str(month)
        if day < 10:
            day = '0' + str(day)
        if hour < 10:
            hour = '0' + str(hour)
        if minute < 10:
            minute = '0' + str(minute)
        if second < 10:
            second = '0' + str(second)

        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
        self.obsid = obsid
        self.wavelength = wavelength

        base_path = '/sml/sdo/'
        path = 'iris_l2_' + str(year) + str(month) + str(day) + '_' + str(hour) + str(minute) + str(second) + '_' + str(obsid) + '_SDO/'
        download_file = path[:-1]
        filename = f'aia_l2_{year}{month}{day}_{hour}{minute}{second}_{obsid}_{wavelength}'

        try:
            hdulist = fits.open(base_path+path+filename)
        except Exception:

            if os.path.isfile("sdo/"+path+filename+".fits"):
                hdulist = fits.open("sdo/"+path+filename+".fits")
            else:
                runcmd(f"mkdir sdo/{path}", verbose=True)
                runcmd(f"wget -q -P sdo/{path} https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/{year}/{month}/{day}/{year}{month}{day}_{hour}{minute}{second}_{obsid}/{download_file}.tar.gz", verbose=True)

                runcmd(f"tar -xf sdo/{path}{download_file}.tar.gz -C sdo/{path}", verbose=True)
                runcmd(f"rm sdo/{path}{download_file}.tar.gz", verbose=True)
                hdulist = fits.open('sdo/'+path+filename+'.fits')

        times = Time(hdulist[0].header["STARTOBS"]) + TimeDelta(
                hdulist[1].data[:, hdulist[1].header["TIME"]], format="sec")
        wcs = WCS(hdulist[0].header)
        aia_data = hdulist[0].data

        self.header = hdulist[0].header
        self.times = times
        self.wcs = wcs
        self.data = aia_data

        hdulist.close()



class HMI_AIA_single_image:

    def __init__(self, time, dataset_name, hours=None, wavelength='171'):

        """
        This class collects AIA and HMI data and downloads them for a given time length. Always provide a wavelength informaiton as a keyword argument. Otherwise you might download huge datasets!

        Input:

            time : astropy Time object, start time

            dataset_name : str, either magnetogram, continuum, aia_image.

            hours : int, number of hours for which you want to collect HMI/AIA data. This part is still under development.

            wavelength : str, AIA filter wavelength number

        Attributes:

            wavelength : from input

            header : astropy header object

            times : astropy Time object, containing the exact time of the returned data.

            data : np.array, containing the image data.

            wcs : astropy WCS object, for coordinate transformations.


        """

        if hours>48:
            raise ValueError(f"hours is too long, choose a number below 48, currently chosen : {hours}")

        time_datetime = time.datetime
        time_iso = time.iso
        time_tai_str = parse_iso_string_to_tai(time_iso)
        time_datetime_end = time_datetime + timedelta(hours=hours) if hours else time_datetime + timedelta(hours=1)
        time_end_tai_str = parse_iso_string_to_tai(time_datetime_end.isoformat())

        c = drms.Client()
        if dataset_name == 'magnetogram':
            dataset = 'hmi.M_720s'
        elif dataset_name == 'continuum':
            dataset = 'hmi.Ic_45s'
        elif dataset_name == 'aia_image':
            dataset_name = 'image'
            dataset = 'aia.lev1_euv_12s'
        else:
            raise ValueError('Dataset not recognized')

        if dataset_name == 'image':
            name = f"{dataset}[{time_tai_str}-{time_end_tai_str}][{wavelength}]"
        else:
            name = f"{dataset}[{time_tai_str}-{time_end_tai_str}]"

        keys, segments = c.query(f"{name}", key=drms.const.all, seg=dataset_name)

        if hours:
            loc = np.arange(len(keys))
            data, header, wcs, times = download_dataset(loc, segments, dataset_name, keys)

        if dataset_name == 'image':
            loc = np.argmin(np.abs(Time(list(keys['T_OBS'].values)) - time))
        else:
            loc = np.argmin(np.abs(np.array([parse_tai_string(tstr) for tstr in list(keys['T_REC'].values)]) - time_datetime))

        data, header, wcs, times = download_dataset(loc, segments, dataset_name, keys)

        if len(segments) == 0:
            raise ValueError('No data found')

        self.wavelength = wavelength
        self.header = header
        self.times = times
        self.wcs = wcs
        self.data = data




def download_dataset(loc, segments, dataset_name, keys):
    """
    Download dataset in segments at loc and return

    data, wcs, times, header
    """

    if isinstance(loc, np.ndarray):

        time_spec_header = []
        time_spec_data = []
        times = []

        for lc in loc:
            url_hmi = 'http://jsoc.stanford.edu' + getattr(segments, dataset_name)[lc]
            hdulist = fits.open(url_hmi)

            time_spec_data.append(hdulist[-1].data)
            time_spec_header.append(hdulist[-1].header)

            times.append((hdulist[-1].header["T_OBS"]) if dataset_name == 'image' else (np.array([parse_tai_string(tstr) for tstr in list(keys['T_REC'].values)])[lc]))

            hdulist.close()

        data = np.array(time_spec_data)
        header = time_spec_header if dataset_name == 'image' else keys.iloc[loc].to_dict()

        times = Time(times)

        wcs = WCS(header)

        return data, header, wcs, times

    else:

        url_hmi = 'http://jsoc.stanford.edu' + getattr(segments, dataset_name)[loc]
        hdulist = fits.open(url_hmi)

        times = Time(hdulist[-1].header["T_OBS"]) if dataset_name == 'image' else Time(np.array([parse_tai_string(tstr) for tstr in list(keys['T_REC'].values)])[loc])
        wcs = WCS(hdulist[-1].header)

        data = hdulist[-1].data
        header = hdulist[-1].header if dataset_name == 'image' else keys.iloc[loc].to_dict()

        hdulist.close()

    return data, header, wcs, times



def get_active_region_center(ar_number, time):
    """function to get the center of an active region at a given time"""
    time_datetime = parse_tai_string(time)
    time_datetime_end = time_datetime + timedelta(hours=1)
    Time_end = Time(time_datetime_end)
    dateend, timeend = str(Time_end.tai).split(' ')
    time_end = dateend.replace('-','.')+'_'+timeend+'_TAI'

    c = drms.Client()
    name = f"hmi.sharp_cea_720s[][{time}-{time_end}][?(NOAA_AR =  {ar_number} )?]"
    keys, segments = c.query(f"{name}", key=drms.const.all, seg="continuum")

    if len(segments) == 0:
        return None

    no_ar = keys['NOAA_AR'][0]
    center_latitude = keys['LAT_FWTPOS'][0]
    center_longitude = keys['LON_FWTPOS'][0]


    return center_longitude, center_latitude, keys, segments


def parse_tai_string(tstr, datetime=True):
    """function to convert T_REC into a datetime object"""
    year   = int(tstr[:4])
    month  = int(tstr[5:7])
    day    = int(tstr[8:10])
    hour   = int(tstr[11:13])
    minute = int(tstr[14:16])
    if datetime: return dt_obj(year,month,day,hour,minute)
    else: return year,month,day,hour,minute


def parse_iso_string_to_tai(tstr):
    """function to convert an ISO string to a TAI string"""

    if isinstance(tstr, dt_obj):
        tstr = tstr.isoformat()
    year   = int(tstr[:4])
    month  = int(tstr[5:7])
    day    = int(tstr[8:10])
    hour   = int(tstr[11:13])
    minute = int(tstr[14:16])
    return f"{year}.{month:02d}.{day:02d}_{hour:02d}:{minute:02d}_TAI"


def spherical_pythagorean(ar_longitude, ar_latitude):
    # check if input is array
    if isinstance(ar_longitude, np.ndarray) and isinstance(ar_latitude, np.ndarray):
        arg_lon = np.cos(np.deg2rad(ar_longitude))
        arg_lat = np.cos(np.deg2rad(ar_latitude))
        return np.rad2deg(np.arccos(arg_lat.reshape(-1,1)*arg_lon.reshape(1,-1)))
    else:
        return np.rad2deg(np.arccos(np.cos(np.deg2rad(ar_longitude))*np.cos(np.deg2rad(ar_latitude))))


def find_best_overlap(reference_image, target_image):
    correlation = correlate2d(reference_image, target_image, mode='same', boundary='fill', fillvalue=np.nan)
    y, x = np.where(np.nanmin(correlation)==correlation)
    return y, x, correlation


def normalize_continuum_image(image):
    return image/np.nanmean(image)

def _correct_for_limb(sunpy_map):
    """
    This function takes sunpy map and removes limb darkening from it
    It transfer coordinate mesh to helioprojective coordinate (using data from header)
    Calucalates distance from sun center in units of sun radii at the time of observation
    Uses limb_dark function with given coeffitiens and divides by that value
    Parameters
    ----------
    sunpy_map: sunpy.map.Map object
                    Data object that should be corrected for limb darkening

    Returns
    -------
    sunpy.map.Map object - output data object which is corrected for limb darkening
    """
    helioproj_limb = sunpy.map.all_coordinates_from_map(sunpy_map).transform_to(
        frames.Helioprojective(observer=sunpy_map.observer_coordinate))
    rsun_hp_limb = sunpy_map.rsun_obs.value
    distance_from_limb = np.sqrt(
        helioproj_limb.Tx.value**2+helioproj_limb.Ty.value**2)/rsun_hp_limb

    koef=np.array([0.32519, 1.26432, -1.44591, 1.55723, -0.87415, 0.173333])

    if len(koef) != 6:
        raise ValueErrror("koef len should be exactly 6")
    if np.max(distance_from_limb) > 1 or np.min(distance_from_limb) < 0:
        raise ValueError("r should be in [0,1] range")
    mu = np.sqrt(1-distance_from_limb**2)  # mu = cos(theta)
    limb = koef[0]+koef[1]*mu+koef[2]*mu**2+koef[3]*mu**3+koef[4]*mu**4+koef[5]*mu**5


    limb_cor_data = sunpy_map.data / limb

    return sunpy.map.Map(limb_cor_data, sunpy_map.meta), limb



# some unit conversions
UNIT_M_NM = 1e10
UNIT_DEC_ARCSEC = 3600
XLABEL_ARCSEC = "solar x [arcsec]"
YLABEL_ARCSEC = "solar y [arcsec]"
XLABEL_ANGSTROM = r'$\lambda$ [$\AA$]'


class iris_coordinates:
    """
    A class that allows to convert pixels into coordinates and vice versa.
    Works both for spectra and rasters and makes heavy use of astropy.wcs.

    Warning: the functions in this file underwent basic tests, but more rigorous
    tests have to performed before this class can be fully trusted.

    Parameters
    ----------
    header :
        astropy HDUList header directly from FITS extension
    mode:
        whether to work in SJI ('sji') or raster ('raster') mode
    """

    def __init__( self, header, mode ):

        # initialize astropy WCS object and suppress warnings
        # set CDELTi to a tiny value if zero (otherwise wcs produces singular PC matrix)
        # see e.g. discussion at https://github.com/sunpy/irispy/issues/78
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if header['CDELT1'] == 0:
                header['CDELT1'] = 1e-10
            if header['CDELT2'] == 0:
                header['CDELT2'] = 1e-10
            if header['CDELT3'] == 0:
                header['CDELT3'] = 1e-10

            self.wcs = WCS( header )

            if header['CDELT1'] == 1e-10:
                header['CDELT1'] = 0
            if header['CDELT2'] == 1e-10:
                header['CDELT2'] = 0
            if header['CDELT3'] == 1e-10:
                header['CDELT3'] = 0

        # set mode (sji or raster) and appropriate conversions and labels
        if mode == 'sji':
            self.conversion_factor = [UNIT_DEC_ARCSEC, UNIT_DEC_ARCSEC]
            self.xlabel = XLABEL_ARCSEC
            self.ylabel = YLABEL_ARCSEC
        elif mode == 'raster':
            self.conversion_factor = [UNIT_M_NM, UNIT_DEC_ARCSEC]
            self.xlabel = XLABEL_ANGSTROM
            self.ylabel = YLABEL_ARCSEC
        else:
            raise ValueError( "mode should be either 'sji' or 'raster'" )

        self.mode = mode




    def pix2coords( self, timestep, pixel_coordinates ):
        """
        Function to convert from camera (pixel) coordinates to solar/physical coordinates.
        Makes heavy use of astropy.wcs.

        Parameters
        ----------
        timestep : int
            time step in the image cube
        pixel_coordinates : np.array
            numpy array with x and y coordinates

        Returns
        -------
        np.array :
            Tuple of solar x and y coordinates in Arcsec (SJI) or wavelength and solar y coordinates (raster) in Arcsec / Angstrom.
        """

        # make sure pixel_coordinates is a numpy array
        pixel_coordinates = np.array( pixel_coordinates )

        # check dimensions
        ndim = pixel_coordinates.ndim
        shape = pixel_coordinates.shape
        if not ( (ndim == 1 and shape[0] == 2) or (ndim == 2 and shape[1] == 2) ):
            raise ValueError( "pixel_coordinates should be a numpy array with shape (:,2)." )

        # create a copy of the input coordinates
        pixel_coordinates = pixel_coordinates.copy()

        # generalize for single pixel pairs
        if ndim == 1:
            pixel_coordinates = np.array([pixel_coordinates])

        # stack timestep to pixels
        pixel_coordinates = np.hstack( [ pixel_coordinates, pixel_coordinates.shape[0]*[[timestep]] ] )

        # transform pixels to solar coordinates
        solar_coordinates = self.wcs.all_pix2world( pixel_coordinates, 1 )[:,:2]

        # convert units
        solar_coordinates *= self.conversion_factor

        # return tuple if input was only one tuple
        if ndim == 1:
            return solar_coordinates[0]
        else:
            return solar_coordinates


    def coords2pix( self, timestep, solar_coordinates, round_pixels=True ):
        """
        Function to convert from solar/physical coordinates to camera (pixel) coordinates.
        Makes heavy use of astropy.wcs.

        Parameters
        ----------
        timestep : int
            time step in the image cube
        solar_coordinates : np.array
            numpy array with solar coordinates (x,y) (SJI) or solar/wavelength coordinates (lambda,y) (raster)

        Returns
        -------
        np.array :
            Tuple (x,y) of camera coordinates in pixels
        """

        # make sure solar_coordinates is a numpy array
        solar_coordinates = np.array( solar_coordinates )

        # check dimensions
        ndim = solar_coordinates.ndim
        shape = solar_coordinates.shape
        if not ( (ndim == 1 and shape[0] == 2) or (ndim == 2 and shape[1] == 2) ):
            raise ValueError( "pixel_coordinates should be a numpy array with shape (:,2)." )

        # create a copy of the input coordinates
        solar_coordinates = solar_coordinates.copy()

        # generalize for single pixel pairs
        if ndim == 1:
            solar_coordinates = np.array([solar_coordinates])

        # convert units
        solar_coordinates = solar_coordinates / self.conversion_factor

        # convert timestep to time coordinate (want always to reference time with timestep)
        time_coordinate = self.wcs.all_pix2world( [[0,0,timestep]], 1  )[0, 2]

        # stack timestep to pixels
        solar_coordinates = np.hstack( [ solar_coordinates, solar_coordinates.shape[0]*[[time_coordinate]] ] )

        # transform solar coordinates to pixels
        pixel_coordinates = self.wcs.all_world2pix( solar_coordinates, 1 )[:,:2]

        # round to nearest pixels
        if round_pixels:
            pixel_coordinates = np.round( pixel_coordinates ).astype( np.int )

        # return tuple if input was only one tuple
        if ndim == 1:
            return pixel_coordinates[0]
        else:
            return pixel_coordinates


def Load_data_labels(Label_set_filename, line):
    """
    Load labels to know for which model which observations were used for training and which ones for testing.

    """
    path_cleaned = f'/sml/zbindenj/MgIIk/cleaned/' #using data prepared from aggregate

    Label_set_ = np.load(path_cleaned + Label_set_filename, allow_pickle=True)['arr_0'][()]

    return Label_set_



def get_slit_yprob_data_to_overplot(obs_cls_Mg, time_):

    """
    Evaluate models on the chosen obs and create aggregate of yhat outputs. This takes a single time step and creates a data array containing the aggregated yhat outputs for that time step. The trained models will not be part of the repository.

    Input:

        obs_cls_Mg : Obs_raw_data object from IRIScast, containing all the processed spectra and data of that IRIS observation.

        time_ : astropy Time object, at which the outputs should be calculated.

    """

    num_of_models = 0

    obs_id_w_ext = obs_cls_Mg.obs_id

    time_delta_obs_cls = np.abs(obs_cls_Mg.times - time_.unix)
    time_delta_sji = np.abs(sji.sji_times.unix - time_.unix)
    rast_ind, obs_t = np.unravel_index(np.argmin(time_delta_obs_cls, axis=None), time_delta_obs_cls.shape)
    sjind = np.unravel_index(np.argmin(time_delta_sji, axis=None), time_delta_sji.shape)
    yhat_arr_probs_sum = np.array([np.zeros_like(obs_cls_Mg.norm_vals[n,obs_t,:]) for n in range(obs_cls_Mg.num_of_raster_pos)])

    X_test = np.array([obs_cls_Mg.im_arr[n,obs_t,:,:] for n in range(obs_cls_Mg.num_of_raster_pos)])
    print(np.any(X_test[-1]!=0))

    Label_set = Load_data_labels('Label_set_5_5.npz', 'aggregate') # why aggregate here?

    num_of_models = 0

    # Go over all models and take the models that did not have the observation in the training set
    for itter in range(5): # number of repetitions, needs to be adjusted
        for k in range(5): # number of splits, needs to be adjusted

            label_train = Label_set[str(k) + '_' + str(itter) + '_training']

            label_train = [lb[3:31] for lb in label_train]
            if obs_cls.obs_id_w_ext in label_train:
                pass
            else:
                # Load model with itter and k
                save_path_models = f'~/models/decision_model_{k}_{itter}'

                decision_model = mdls.ConvNet(960, 6) # Initiate network

                decision_model.load_state_dict(torch.load(f'{save_path_models}_convnet.pt', map_location=torch.device('cpu')))

                decision_model.eval()
                decision_model.cpu()

                # predict probabilities for test set
                yhat_probs = decision_model(torch.from_numpy(X_test).type(torch.FloatTensor))

                yhat_arr_probs = yhat_probs.detach().numpy().squeeze()
                yhat_arr_probs = yhat_arr_probs.reshape(yhat_arr_probs_sum.shape[0], yhat_arr_probs_sum.shape[1])

                yhat_arr_probs_sum = yhat_arr_probs_sum + yhat_arr_probs.reshape(yhat_arr_probs.shape[0], yhat_arr_probs.shape[1],1)

                num_of_models += 1

    # print(yhat_arr_probs_sum.shape)
    yhat_arr_probs_sum[np.where(np.all(X_test==0, axis=-1))] = np.full((1), fill_value=0) # Set pixel to 0 if the spectrum was set to zero
    yhat_arr_probs_ensemble = yhat_arr_probs_sum/num_of_models # Average over all models

    plot_arr = yhat_arr_probs_ensemble.reshape(X_test.shape[0], X_test.shape[1], 1)

    return plot_arr, obs_cls_Mg



def plot_sji_image(self, sjiind, plot_arr, save=False):

    """
    Plots an SJI image for a given sji index and the slit information to be overplotted.

    Input:
        self : IRIS_SJI_CUBE object

        sjiind : int, Index at which the sji image should be taken

        plot_arr : np.array, The slit information to be overplotted

        save : bool, set True if the image should be saved, adjust the path accordingly

    """

    max_intensity=1

    sj_t = sji_times[sjiind]

    rast_ind = np.arange(sji.n_raster_pos)

    plt.cla()
    plt.clf()
    plt.close()

    exptime = sji.sji_header[sjiind]['EXPTIME']

    # Generate initial figure for the animation with overplotted slit colors.
    sji_exped = (sji[sjiind,:,:].clip(min=0)/exptimes[sjiind,:,:])**0.4
    sji_exped[np.where(sji_exped == 0.)] = 5.0

    fig, ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(sji_exped, vmax=10, cmap=cm_)
    xcoords, ycoords = raster_x_coord(sjiind, plot_arr)

    date_obs = parse_time(sji_times[sjiind], format='unix').to_datetime()
    im.axes.set_title( "Flare: {}  Frame: {}  Date-Time: {}".format( Flare_num, frame, date_obs.strftime('%Y-%m-%d %H:%M:%S') ), fontsize=32, alpha=.8)
    im.axes.set_xlabel( 'camera x', fontsize=32, alpha=.8 )
    im.axes.set_ylabel( 'camera y', fontsize=32, alpha=.8 )
    cmap = plt.cm.get_cmap('jet')


    if not max_intensity:
        max_intensity = np.max(plot_arr)*.2
    colors = cmap(np.asarray([plot_arr[m,r_ind] for m, r_ind in enumerate(rast_ind)]).squeeze()/max_intensity)
    if plot_arr.shape[0] != 1:
        scat = ax.scatter(xcoords, ycoords, marker='s', s=30, c=colors.reshape(colors.shape[0]*colors.shape[1], 4), alpha=.25)
    else:
        scat = ax.scatter(xcoords, ycoords, marker='s', s=30, c=colors, alpha=.25)

    plt.text(50,300, 'Mg II h&k', fontsize=64, c='white')

    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(which='major', width=2.00)
    ax.tick_params(which='major', length=10)
    ax.tick_params(which='minor', width=2.00)
    ax.tick_params(which='minor', length=6)
    im.axes.tick_params(axis='x', labelsize=32)
    im.axes.tick_params(axis='y', labelsize=32)

    plt.tight_layout()

    if save:
        fig.savefig(f'yhats_overview_plot_Mg_sji_Si_{obs_label}.pdf', bbox_inches='tight')


###############################################################################

# SST functions:

class register_coordinates:
    """
        This is a handmade function to convert between pixel coordinates and arcseconds, relative to the center of the FOV.
    """

    def __init__(self, coordinates_data, image_shape):
        delta_x = coordinates_data[0][1][0] - coordinates_data[0][0][0]
        delta_y = coordinates_data[0][1][0] - coordinates_data[0][0][0]

        self.conversion_factor_x = delta_x/image_shape[-2]
        self.conversion_factor_y = delta_y/image_shape[-1]

class coordinates_projection(register_coordinates):

    """
    Transforms between the two coordinate systems given from register_coordinates, can be used to for instance set ticks in a matplotlib figure.
    """

    name='coordinates_projection'

    def __init__(self, coordinates_data, image_shape):
        super().__init__(coordinates_data, image_shape)

    def transform(self):
        "Transforms from pixel coordinates to world coordinates"
        xshape_arcsec = profs_map_shape[-2]*self.conversion_factor_x
        yshape_arcsec = profs_map_shape[-1]*self.conversion_factor_y
        x = np.linspace(-xshape_arcsec/2, xshape_arcsec/2, profs_map_shape[-2])
        y = np.linspace(-yshape_arcsec/2, yshape_arcsec/2, profs_map_shape[-1])
        return x, y


    def transform_inverse(self):
        "Transforms from world coordinates to pixel coordinates"
        x = np.arange(profs_map_shape[-2])
        y = np.arange(profs_map_shape[-1])
        return x, y


def downsample_SST_map(SST_map, HMI_resolution, SST_resolution_present_map):
    "Downsamples SST map to HMI resolution"
    binning_factor = int(HMI_resolution/SST_resolution_present_map)
    crop_SST = SST_map.shape[0]//binning_factor*binning_factor
    SST_map = SST_map[:crop_SST, :crop_SST]
    SST_map_downsampled = SST_map.reshape(SST_map.shape[0]//binning_factor, binning_factor, SST_map.shape[1]//binning_factor, binning_factor).mean(axis=(1,3))
    return SST_map_downsampled



# Example to understand the code:
if __name__ == "__main__":

  # IRIS type obs id to get the information to load the data, can also be constructed in some other way.
  obs_id = '20140205_153921_3860259280'

  print(obs_id)
  year = int(obs_id[:4])
  month = int(obs_id[4:6])
  day = int(obs_id[6:8])
  hour = int(obs_id[9:11])
  minute = int(obs_id[11:13])
  second = int(obs_id[13:15])
  obsid = int(obs_id[16:])

  sji = IRIS_SJI_cube(year, month, day, hour, minute, second, obsid) # Load Iris observation data


  # Choose a frame number to plot from the sji obs.
  sjiind = 95
  # Get the raster index of that point in time
  rast_ind = np.argmin(np.abs(sji.raster_times - sji.sji_times[sjiind]))

  # give the dimensions of the sji image data to a seperate variable, just for convenience later.
  dims_sji = sji.data.shape

  # Store the exact raster time step as a astropy time object for later info
  date_obs = parse_time(sji.raster_times[rast_ind], format='unix').to_datetime()

  # To evaluate the raster slit data, for instance the yhat model outputs, we use the rainbow colormap 'jet'
  cmap = plt.get_cmap('jet')

  # Here as slit data we will use in this example
  plot_arr = np.load('plot_arr_for_test.npy')

  if sji.n_raster_pos == 1: # Convert the data to colors.
      colors = cmap(plot_arr).squeeze()
  else:
      colors = cmap(plot_arr).squeeze().reshape(plot_arr.squeeze().shape[0]*plot_arr.squeeze().shape[1],4, order='A')

  print(obs_id , sji.sji_header[sjiind]['OBS_DESC'])

  dataset_name = 'continuum' # We gonna use the a continuum image and an AIA 304 image (below), this can of course be modified to any of the other options.

  sji_times = sji.sji_times # store the times from the sji image as a separate variable.

  raster_ind = sji.get_slit_pos(sjiind) # This is not the rast_ind index from before. This is the slit position at the time of sjiind
  sji.get_sunpy_wcs(sjiind) # For the coordinate transformations between IRIS and AIA/HMI we use the function get_sunpy_wcs. This creates an additional attribute to the sji instance that contains the wcs object generated by sunpy. Tests have turned out that the sunpy method does some modifications to the standard wcs object to make it more robust and more accurate. We have to use the same wcs procedure through out the entire method.

  observer = sji._sunpy_observer_coordinate # Define an observer for the later coordinate transformations. Since we want to see how the FOV and slit data from IRIS looks in the SDO frame, our initial observer is IRIS and we use the sji wcs object coordinate. This can also be replaced with any other observer if the alignment between different AIA and HMI channels would be desired.

  hmi_ = HMI_AIA_single_image(sji.raster_times[rast_ind], dataset_name) # Create the HMI/AIA data and coordinates object.
  hmi_map = sunpy.map.Map(hmi_.data, hmi_.header) # We also create a map object for HMI/AIA to optain the wcs object from sunpy.
  norm = cs.Normalize() # Get norm for the colors
  print(date_obs, hmi_.times)


  # Generate the figure and add a gridspec for the two subfigures
  fig = plt.figure(figsize=(50,20))
  gs = fig.add_gridspec(1,2)
  gs.update(wspace=0.05, hspace=0.05)
  ax = fig.add_subplot(gs[0,0], projection=hmi_map.wcs)
  im = ax.imshow(hmi_map.data, cmap='binary_r', norm=norm, interpolation='nearest', origin='lower') # plot first figure

  # Generate a frame for the FOV of IRIS and map with the sji._sunpy_wcs object to world coordinates. For plotting then convert the coordinates back into AIA/HMI pixel coordinates.
  corners21 = np.array([[0,0],[0,dims_sji[1]-1],[dims_sji[2]-1,dims_sji[1]-1],[dims_sji[2]-1,0]])
  corners211 = np.array([sji._sunpy_wcs.pixel_to_world(corn[0], corn[1]) for corn in corners21])
  corners213 = np.array([hmi_map.wcs.world_to_pixel(corn) for corn in corners211])

  # Create a Polygon and plot it over the image.
  try:
      b = Polygon(corners213, edgecolor='cyan', facecolor='none', closed=True, lw=2, alpha=1, transform=ax.get_transform(hmi_map.wcs))
      ax.add_patch(b)
  except:
      pass

  # Set some image properties, change as required
  im.axes.set_xlabel( 'Solar coordinates x', fontsize=32 )
  im.axes.set_ylabel( 'Solar coordinates y', fontsize=32 )
  ax.xaxis.set_major_locator(MultipleLocator(100))
  ax.yaxis.set_major_locator(MultipleLocator(100))
  ax.xaxis.set_minor_locator(MultipleLocator(50))
  ax.yaxis.set_minor_locator(MultipleLocator(50))
  ax.tick_params(which='major', width=2.00)
  ax.tick_params(which='major', length=10)
  # ax.tick_params(which='minor', width=2.00)
  ax.tick_params(which='minor', length=6)
  im.axes.tick_params(axis='x', labelsize=32)
  im.axes.tick_params(axis='y', labelsize=32)

  # Get the slit coordinates in world coordinates. Use mode 'hmi' here to get the right projections. If you use IRIS aligned AIA images or SJI images, then use mode 'iris'
  xcoords, ycoords = sji.raster_x_coord(sjiind, plot_arr, mode='hmi')
  xcoords, ycoords = hmi_map.wcs.world_to_pixel(SkyCoord(xcoords, ycoords, unit=u.arcsec, observer=observer, frame='helioprojective', obstime=date_obs))
  scat2 = ax.scatter(xcoords, ycoords, marker='s', s=4, c=colors) # plot the colors scattered at the coordinates of the slit.
  scat2.set_alpha(.3)

  # Set bounding box for the plot on the HMI map

  xcen = sji.sji_header[sjiind]['XCEN']
  ycen = sji.sji_header[sjiind]['YCEN']
  bounds_x = sji.sji_header[sjiind]['FOVX']*.8
  bounds_y = sji.sji_header[sjiind]['FOVY']*.8

  x_low_lim = xcen - bounds_x
  y_low_lim = ycen - bounds_y

  x_upp_lim = xcen + bounds_x
  y_upp_lim = ycen + bounds_y

  coords_low_lim = SkyCoord(x_low_lim, y_low_lim, unit='arcsec', frame='helioprojective', observer=observer, obstime=sji.raster_times[rast_ind])
  coords_upp_lim = SkyCoord(x_upp_lim, y_upp_lim, unit='arcsec', frame='helioprojective', observer=observer, obstime=sji.raster_times[rast_ind])

  lim_lower_pix = hmi_map.wcs.world_to_pixel(coords_low_lim)
  lim_upper_pix = hmi_map.wcs.world_to_pixel(coords_upp_lim)

  ax.set_xlim([lim_lower_pix[0],lim_upper_pix[0]])
  ax.set_ylim([lim_lower_pix[1],lim_upper_pix[1]])



  ##############################################################################################################################


  # Plotting different filters in AIA:
  # This part of the code works exactly the same as the part above.
  dataset_name = 'aia_image'
  wavelength = 304

  hmi_ = HMI_AIA_single_image(sji.raster_times[rast_ind], dataset_name, wavelength=wavelength)
  hmi_map = sunpy.map.Map(hmi_.data, hmi_.header)
  norm = cs.Normalize()
  print(date_obs, hmi_.times)

  ax = fig.add_subplot(gs[0,1], projection=hmi_map.wcs)
  im = ax.imshow(hmi_map.data**.25, cmap='binary_r', norm=norm, interpolation='nearest', origin='lower')

  corners21 = np.array([[0,0],[0,dims_sji[1]-1],[dims_sji[2]-1,dims_sji[1]-1],[dims_sji[2]-1,0]])
  corners211 = np.array([sji._sunpy_wcs.pixel_to_world(corn[0], corn[1]) for corn in corners21])
  corners213 = np.array([hmi_map.wcs.world_to_pixel(corn) for corn in corners211])

  try:
      b = Polygon(corners213, edgecolor='cyan', facecolor='none', closed=True, lw=2, alpha=1, transform=ax.get_transform(hmi_map.wcs))
      ax.add_patch(b)
  except:
      pass

  im.axes.set_xlabel( 'Solar coordinates x', fontsize=32 )
  im.axes.set_ylabel( 'Solar coordinates y', fontsize=32 )
  ax.xaxis.set_major_locator(MultipleLocator(100))
  ax.yaxis.set_major_locator(MultipleLocator(100))
  ax.xaxis.set_minor_locator(MultipleLocator(50))
  ax.yaxis.set_minor_locator(MultipleLocator(50))
  ax.tick_params(which='major', width=2.00)
  ax.tick_params(which='major', length=10)
  ax.tick_params(which='minor', length=6)
  im.axes.tick_params(axis='x', labelsize=32)
  im.axes.tick_params(axis='y', labelsize=32)

  xcoords, ycoords = sji.raster_x_coord(sjiind, plot_arr, mode='hmi')
  xcoords, ycoords = hmi_map.wcs.world_to_pixel(SkyCoord(xcoords, ycoords, unit=u.arcsec, observer=observer, frame='helioprojective', obstime=date_obs))
  scat2 = ax.scatter(xcoords, ycoords, marker='s', s=4, c=colors)
  scat2.set_alpha(.3)

  # Set bounding box for the plot on the HMI map

  xcen = sji.sji_header[sjiind]['XCEN']
  ycen = sji.sji_header[sjiind]['YCEN']
  bounds_x = sji.sji_header[sjiind]['FOVX']*.8
  bounds_y = sji.sji_header[sjiind]['FOVY']*.8

  x_low_lim = xcen - bounds_x
  y_low_lim = ycen - bounds_y

  x_upp_lim = xcen + bounds_x
  y_upp_lim = ycen + bounds_y

  coords_low_lim = SkyCoord(x_low_lim, y_low_lim, unit='arcsec', frame='helioprojective', observer=observer, obstime=sji.raster_times[rast_ind])

  coords_upp_lim = SkyCoord(x_upp_lim, y_upp_lim, unit='arcsec', frame='helioprojective', observer=observer, obstime=sji.raster_times[rast_ind])

  lim_lower_pix = hmi_map.wcs.world_to_pixel(coords_low_lim)
  lim_upper_pix = hmi_map.wcs.world_to_pixel(coords_upp_lim)

  ax.set_xlim([lim_lower_pix[0],lim_upper_pix[0]])
  ax.set_ylim([lim_lower_pix[1],lim_upper_pix[1]])

  gs.tight_layout(fig)
  fig.suptitle( "Flare: {}  Frame: {}  Date-Time: {} IRIS, AIA-171, HMI Continuum combined".format( obs_id, sjiind, date_obs, wavelength ), fontsize=32 )

  plt.show()


  # fig.savefig(f'Flare_{obs_id}_frame_{sjiind}_aia_continuum_combined.png', dpi=300, bbox_inches='tight')
