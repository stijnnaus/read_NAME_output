#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:23:23 2024

@author: nauss
"""

from netCDF4 import Dataset
import numpy as np
import xesmf
import xarray

def read_mahuika_onroad():
    filename = 'emissions/onroad_hourly_500m_CO2ff_2016.nc'
    with Dataset(filename, 'r') as d:
        lons = d['longitude'][:]
        lats = d['latitude'][:]
        emi  = d['CO2ff_annual'][:] # ton/hr
        emi *= 1000                 # ton/hr to kg/hour
        emi  = np.swapaxes(emi,0,1) # The only category with dimensions (lon,lat) instead of (lat,lon)
        
        # Fill values to 0
        emi = np.array(emi)
        emi[emi>1e30] = 0.0 
        emi[np.isnan(emi)] = 0.0 
        
    # convert kg/hour to kg/m2/s
    emi /= 3600.
    area_per_gridcell = calc_area_per_gridcell(lats,lons)
    emi /= area_per_gridcell[:,:,np.newaxis,np.newaxis]
    
    return lats, lons, emi
    
def regrid_emissions(lat_in, lon_in, emis, lat_out, lon_out):
    '''
    Mass-conserving regridding of emissions.
    '''
    
    ds_in  = xarray.Dataset({"lat": (["lat"], lat_in, {"units": "degrees_north"}),
                         "lon": (["lon"], lon_in, {"units": "degrees_east"})})
    ds_out = xarray.Dataset({"lat": (["lat"], lat_out, {"units": "degrees_north"}),
                         "lon": (["lon"], lon_out, {"units": "degrees_east"})})
    regr = xesmf.Regridder(ds_in, ds_out, 'conservative')
    return regr(emis)

def calc_area_per_gridcell(lats, lons, bounds=False):
    '''
    Calculate area per grid cell in m2
    If lat, lon are grid cell bounds, then bounds=True
    If they are grid cell centers we calculate first grid cell boundaries (approx)
    '''
    
    if not bounds:
        lats = get_gridcell_bounds_from_centers(lats)
        lons = get_gridcell_bounds_from_centers(lons)
    
    dlat = np.abs(np.sin(lats[1:]*np.pi/180.) - np.sin(lats[:-1]*np.pi/180.))
    dlon = np.abs(lons[1:]-lons[:-1])*np.pi/180.
    R_EARTH = 6371e3 # [m]
    area = np.outer(dlat,dlon)*R_EARTH**2
    
    return area

def get_gridcell_bounds_from_centers(ll):
    '''
    An approximate way to get from e.g., latitude gridcell centers the boundaries.
    We assume that bounds are in the middle between grid cell centers, 
    and extrapolate the two outer boundaries by the mean grid spacing
    '''
    
    dl = np.abs(np.mean(ll[1:]-ll[:-1]))
    ll_mid = list((ll[1:] + ll[:-1])/2.)
    ll = [ll[0]-dl/2.] + ll_mid + [ll[-1]+dl/2.]
    return np.array(ll)