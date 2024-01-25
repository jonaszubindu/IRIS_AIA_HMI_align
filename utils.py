#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:07:11 2021

@author: jonaszbinden
"""
import os

from astropy import constants as c
from irisreader.utils.date import to_epoch

import astropy.units as u
from sunpy.time import parse_time

import numpy as np


from utils_features import *
import h5py
from astropy import constants as c


DN2PHOT = {'NUV':18/(u.pixel), 'FUV': 4/(u.pixel)}

#######################################################################################################################

def transform_arrays(times, image_arr=None, norm_vals_arr=None, num_of_raster_pos=0, forward=True):

    """
    transforms between image_arrays in shape (n,t,y,lambda) -> (n*t,y,lambda)
    and time arrays from shape (n,t) -> (n*t,)

    or

    the reverse operation

    """

    if not np.any(image_arr) and forward:

        image_arr = np.zeros(times.shape + (1,)+ (1,))
        norm_vals_arr = np.zeros(times.shape + (1,)+ (1,))

    elif not np.any(image_arr) and not forward:
        image_arr = np.zeros((times.shape[1],) + (1,)+ (1,))

    if forward:


        image_arr_r = image_arr.reshape(image_arr.shape[0]*image_arr.shape[1], image_arr.shape[2], image_arr.shape[3],
                                        order='F')
        times_r = times.reshape(times.shape[0]*times.shape[1], 1, order='F')
        norm_vals_r = norm_vals_arr.reshape(norm_vals_arr.shape[0]*norm_vals_arr.shape[1], norm_vals_arr.shape[2], 1, order='F')

        if np.any(times_r != sorted(times_r)):
            print(times_r[np.where(times_r != sorted(times_r))])
            if num_of_raster_pos == 1:
                sorted_inds = np.squeeze(np.argsort(times_r,axis=0))
#                 print(sorted_inds)
                times_r = times_r[sorted_inds]
                image_arr_r = image_arr_r[sorted_inds,:,:]
                norm_vals_r = norm_vals_r[sorted_inds,:,:]

            else:
                raise Warning('could not reshape times, times are not sequential and number of raster steps is: '
                              , num_of_raster_pos)

        complete_raster_step = times.shape[1]
        raster_pos = np.hstack([np.arange(0,num_of_raster_pos) for n in range(complete_raster_step)])
        times_r = np.vstack([raster_pos,times_r.T])

    else: # reverse

        if num_of_raster_pos > image_arr.shape[0]:

            start_ind = None
            stop_ind = None
            try:
                start_ind = np.where(times[0]==0)[0][0]
            except Exception:
                stop_ind = np.where(times[0]==(num_of_raster_pos-1))[0][-1]

            image_arr_new = np.zeros([num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]])
            times_new = np.zeros([num_of_raster_pos])
            norm_vals_new = np.zeros([num_of_raster_pos, image_arr.shape[1], 1])

            if start_ind:
                image_arr_new[start_ind:] = image_arr
            elif stop_ind:
                image_arr_new[:stop_ind] = image_arr
            else:
                image_arr_new[int(times[0][0]):int(times[0][-1])+1] = image_arr


            print(f"number of raster positions > timesteps, {num_of_raster_pos} > {image_arr.shape[0]}")

            times_origin = deepcopy(times)

            image_arr = image_arr_new
            times = times_new
            norm_vals_arr = norm_vals_new

            image_arr_r = image_arr.reshape(int(image_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]).transpose(1,0,2,3).reshape(num_of_raster_pos, int(image_arr.shape[0]/num_of_raster_pos), image_arr.shape[1], image_arr.shape[2])

            times_r = times.reshape(int(times.shape[0]/num_of_raster_pos), num_of_raster_pos).transpose(1,0).reshape(num_of_raster_pos, int(times.shape[0]/num_of_raster_pos))

            norm_vals_r = norm_vals_arr.reshape(int(norm_vals_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, norm_vals_arr.shape[1],1).transpose(1,0,2,3).reshape(num_of_raster_pos, int(norm_vals_arr.shape[0]/num_of_raster_pos), norm_vals_arr.shape[1],1)

        elif num_of_raster_pos == 1:

            image_arr_r = image_arr.reshape(1,image_arr.shape[0],image_arr.shape[1],image_arr.shape[2])
            times_r = times[1].reshape(1,times.shape[1])
            norm_vals_r = norm_vals_arr.reshape(1,norm_vals_arr.shape[0],norm_vals_arr.shape[1],1)

        else:


            start_ind = None
            stop_ind = None

            start_ind = np.where(times[0]==0)[0][0]
            stop_ind = np.where(times[0]==(num_of_raster_pos-1))[0][-1]

            if stop_ind < start_ind:

                try:
                    start_ind = np.where(times[0]==0)[0][0]
                except Exception:
                    stop_ind = np.where(times[0]==(num_of_raster_pos-1))[0][-1]

                image_arr_new = np.zeros([num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]])
                times_new = np.zeros([num_of_raster_pos])
                norm_vals_new = np.zeros([num_of_raster_pos, image_arr.shape[1], 1])

                if start_ind:
                    image_arr_new[start_ind:] = image_arr
                elif stop_ind:
                    image_arr_new[:stop_ind] = image_arr
                else:
                    image_arr_new[int(times[0][0]):int(times[0][-1])+1] = image_arr


                print(f"number of raster positions > timesteps, {num_of_raster_pos} > {image_arr.shape[0]}")

                times_origin = deepcopy(times)

                image_arr = image_arr_new
                times = times_new
                norm_vals_arr = norm_vals_new

                image_arr_r = image_arr.reshape(int(image_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]).transpose(1,0,2,3).reshape(num_of_raster_pos, int(image_arr.shape[0]/num_of_raster_pos), image_arr.shape[1], image_arr.shape[2])

                times_r = times.reshape(int(times.shape[0]/num_of_raster_pos), num_of_raster_pos).transpose(1,0).reshape(num_of_raster_pos, int(times.shape[0]/num_of_raster_pos))

                norm_vals_r = norm_vals_arr.reshape(int(norm_vals_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, norm_vals_arr.shape[1],1).transpose(1,0,2,3).reshape(num_of_raster_pos, int(norm_vals_arr.shape[0]/num_of_raster_pos), norm_vals_arr.shape[1],1)

            else:

                num_of_cycles = int(np.floor((stop_ind - start_ind)/(num_of_raster_pos-1)))


                if start_ind == stop_ind+1:
                    raise ValueError("cannot reshape, start_ind and stop_ind for symmetric reshaping is the same")

                image_arr_new = np.zeros([num_of_raster_pos*(num_of_cycles+2), image_arr.shape[1], image_arr.shape[2]])
                times_new = np.zeros([num_of_raster_pos*(num_of_cycles+2)])
                norm_vals_new = np.zeros([num_of_raster_pos*(num_of_cycles+2), norm_vals_arr.shape[1], 1])

                image_arr_new[int(times[0][0]):int(times.shape[1]+times[0][0])] = image_arr
                times_new[int(times[0][0]):int(times.shape[1]+times[0][0])] = times[1,:]
                norm_vals_new[int(times[0][0]):int(times.shape[1]+times[0][0])] = norm_vals_arr.reshape(norm_vals_arr.shape[0], norm_vals_arr.shape[1],1)

                times_origin = deepcopy(times)

                image_arr = image_arr_new
                times = times_new
                norm_vals_arr = norm_vals_new

                image_arr_r = image_arr.reshape(int(image_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]).transpose(1,0,2,3).reshape(num_of_raster_pos, int(image_arr.shape[0]/num_of_raster_pos), image_arr.shape[1], image_arr.shape[2])

                times_r = times.reshape(int(times.shape[0]/num_of_raster_pos), num_of_raster_pos).transpose(1,0).reshape(num_of_raster_pos, int(times.shape[0]/num_of_raster_pos))

                norm_vals_r = norm_vals_arr.reshape(int(norm_vals_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, norm_vals_arr.shape[1],1).transpose(1,0,2,3).reshape(num_of_raster_pos, int(norm_vals_arr.shape[0]/num_of_raster_pos), norm_vals_arr.shape[1],1)

    return times_r, image_arr_r, norm_vals_r



class Obs_raw_data:

    """

    Class structure to store all the important information for later training and testing of prediction models, train and clean
    with VAE's or overplot SJI images.

        Parameters:
        -------------------
        obs_id : string
            IRIS observation ID
        num_of_raster_pos : int
            Number of raster positions (not times steps)
        times_global : numpy ndarray
            first row contains raster position, second row contains time in unix
        im_arr_global : numpy ndarray
            contains all spectra like (T,y,lambda)
        norm_vals_global : numpy ndarray
            contains the normalization values for each spectrum like (T,y,1)
        n_breaks : int
            interpolation points
        lambda_min : int or float
            lower wavelength limit
        lambda_max : int or float
            upper wavelength limit
        field : string
            FUV or NUV
        line : string
            spectral line
        threshold : int
            lower limit in DN/s at which spectra were cleaned
        hdrs : pandas DataFrame
            containing all headers from raster headers
        threshold : int
            lower limit in DN/s at which spectra were cleaned

    class methods :

        __init__ : initializes a new instance of Obs_raw_data

        time_clipping : clips the observation in time: times_global, im_arr_global, norm_vals_global

            start : datetime
            end : datetime

        save_arr : saves the Obs_raw_data instance according to a specific frame, adjust path in save_arr.

            filename : filename to store the instance as
            line : spectralline
            typ : type of observation : QS, SS, AR, PF

    global methods: check each method for necessary args and kwargs

        clean_SAA_cls(obs_cls) : cleans the given instance for SAA by setting SAA parts to 0

        transform arrays : transforms array between (n,t,y,lambda) <-> (T,y,lambda)
                           CAUTION : timeclipping destroys the equivalence between the two arrays. The function automatically
                           accounts for that by using the first and last complete raster steps.

        spectra_quick_look : allows the user to have a peek at some random spectra

        load_obs_data : allows the user to load a stored Obs_raw_data instance. args/kwargs are the same as in save_arr. Adjust
                        path if necessary


    """

    def __init__(self, obs_id=None, raster=None, lambda_min=None, lambda_max=None, n_breaks=None, field=None, line=None,
                 threshold=None, load_dict=None):

        spectra_stats_single_obs = {}

        if load_dict:

            filename, line, typ = load_dict.values()
            try:
                with h5py.File(f'/sml/zbindenj/{line}/{typ}/{filename}/arrays.h5', 'r') as f:
                    im_arr_global = f["im_arr_global"][:,:,:]
                    times_global = f["times_global"][:,:]
                    norm_vals_global = f["norm_vals_global"][:,:,:]

                init_dict = np.load(f'/sml/zbindenj/{line}/{typ}/{filename}/dict.npz', allow_pickle=True)['arr_0'][()]

                for key, value in init_dict.items():

                    setattr(self, key, value)

                self.im_arr_global = im_arr_global[:,:,:]
                self.times_global = times_global[:,:]
                self.norm_vals_global = norm_vals_global[:,:]

            except Exception as exc:
                print(exc)


        else:

            raise ValueError("For new observations refer to the preprocessing repo under IRIS_FlarePrep")



    def time_clipping(self, start, end):

        """
        Clip image_array and time_array according to start datetime and end datetime, only works with global time steps

        """

        start_e = to_epoch(start)
        end_e = to_epoch(end)

        times = self.times_global[1,:]


        if not (((start_e > times[0]) and (start_e < times[-1])) or ((end_e > times[0]) and (end_e < times[-1]))):
            start_t = parse_time(self.times_global[1,0], format='unix').to_datetime()
            end_t = parse_time(self.times_global[1,-1], format='unix').to_datetime()
            raise Warning(f'in {self.obs_id}: start {start} or end {end} is outside of times: {start_t}, {end_t}')

        start = start_e
        end = end_e

        diff = times - start

        try:
            start_ind = np.argmin(np.abs(diff[diff<0]))
        except ValueError:
            start_ind = 0

        diff = times - end

        end_ind = np.argmin(np.abs(diff[diff<0])) # only take last step before end

        self.times_global = self.times_global[:,start_ind:end_ind] # first dimension contains raster position information
        self.im_arr_global = self.im_arr_global[start_ind:end_ind, :, :] # first dimension contains raster position
                                                                         # information
#         self.im_arr_global_raw = self.im_arr_global_raw[start_ind:end_ind, :, :]
        self.norm_vals_global = self.norm_vals_global[start_ind:end_ind, :]

        # Quick visualization of the selected data ###############################################
#         try:
#             nprof = self.im_arr_global*self.norm_vals_global#.reshape(self.norm_vals_global.shape + (1,))
#         except ValueError:
#             nprof = self.im_arr_global*self.norm_vals_global.reshape(self.norm_vals_global.shape + (1,))


#         nprof = nprof.reshape(nprof.shape[0]*nprof.shape[1], self.n_breaks)

#         nprof = nprof[np.where(~np.all(((nprof == 0) | (nprof == 1)), axis=-1))]

        self.spectra_stats_single_obs['remaining spectra after time clipping'] = self.im_arr_global[np.where(np.any(self.im_arr_global!=0, axis=-1))].shape

#         try:
#             spectra_quick_look(nprof, self.lambda_min, self.lambda_max, self.n_breaks)
#             spectra_quick_look(self.im_arr_global, self.lambda_min, self.lambda_max, self.n_breaks)

#             fig = plt.figure(figsize=(20,20))
#             ax = fig.add_subplot(projection='3d')
#             x = np.arange(nprof.shape[0])
#             y = np.arange(nprof.shape[1])
#             xs, ys = np.meshgrid(x, y)

#             ax.plot_surface(ys.T, xs.T, nprof, cmap=plt.cm.Blues, linewidth=1, alpha=0.9)#, vmin=-5, vmax=+10)

#             ax.axes.set_zlim3d(bottom=-1, top=2000000)
#             ax.view_init(10, -95)
#             plt.show()

#         except Exception as exc:
#             print(exc)
#             print('no data in this time-window')

        ##########################################################################################


    def save_arr(self, filename, line, typ):

        filename = filename.split('.')[0]

        try: # make directory for save file to keep it all together.
            os.mkdir(path=f'/sml/zbindenj/{line}/{typ}/{filename}/')
        except Exception:
            pass


        filename = filename.split('.')[0]

        save_dict = {'obs_id': self.obs_id,
                    'num_of_raster_pos': self.num_of_raster_pos,
                    'lambda_min': self.lambda_min,
                    'lambda_max': self.lambda_max,
                    'n_breaks': self.n_breaks,
                    'field' : self.field,
                    'line' : self.line,
                    'threshold' : self.threshold,
                    'hdrs' : self.hdrs,
                    'stats' : self.spectra_stats_single_obs
                   }

        np.savez(f'/sml/zbindenj/{line}/{typ}/{filename}/dict.npz', save_dict)

        #gc.collect()

        try:

            f = h5py.File(f'/sml/zbindenj/{line}/{typ}/{filename}/arrays.h5', 'w')
            dataset = f.create_dataset("im_arr_global", data=self.im_arr_global)
            dataset = f.create_dataset("times_global", data=self.times_global)
            dataset = f.create_dataset("norm_vals_global", data=self.norm_vals_global)
            f.close()
            del dataset

        except Exception as exc:

            print(exc)
            f.close()


##########################################################################################

def load_obs_data(filename, line, typ, only_im_arr = False):

    filename = filename.split('.')[0]

    if only_im_arr:

        try:

            f = h5py.File('/fast/zbindenj/{line}/{typ}/{filename}/array.h5', 'r')
            im_arr_global = f["im_arr_global"]
            f.close()
            return im_arr_global

        except Exception as exc:

            print(exc)
            f.close()

            return None

    else:

        load_dict = {
                     'filename' : filename,
                     'line' : line,
                     'typ' : typ
                    }

        return Obs_raw_data(load_dict=load_dict)

