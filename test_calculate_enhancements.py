#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to read footprints from NAME runs, and combine with emission fields to 
calculate enhancements.

@author: nauss
"""

import numpy as np
import matplotlib.pyplot as plt
import tarfile
import os
from datetime import datetime,timedelta
import helpers

def read_footprint_NAME(flasknumber, path_base, samplelayer='40m', tarred=False, unit='ppm'):
    """
    Read NAME footprint. This can be done from a tarred output directory, or from
    an untarred directory. 
    NAME footprints are output both in ppm and in concentration, so you need to specify 
    which you want to read. PPM is easier to work with, but PBL average output is only 
    available in concentration.
    Samplelayer is the depth of the layer in which particles are still considered
    to be in contact with the sources. Options are 40m, 100m and PBL
    """
    
    # Hard-coded, this is the domain I used for the flask NAME simulations
    lonsNAME = np.linspace(172, 179.2, 533)
    latsNAME = np.linspace(-39.5, -33, 481)
    
    # Open output file
    if unit=='ppm' or unit=='mixr':
        unitlabel = 'mixr'
    elif unit=='conc':
        unitlabel = 'conc'
    else:
        raise ValueError('Unknown unit for footprint : %s'%unit)
        
    fname = 'Fields_%s_%s_C1.txt'%(unitlabel, samplelayer)
    if tarred:
        try:
            fname_tar    = 'Flask_%i.tar.gz'%(flasknumber)
            tarf         = tarfile.open('%s/%s'%(path_base, fname_tar))
            path_in_tar  = 'Flask_%i/'%flasknumber
            fname_in_tar = os.path.join(path_in_tar, fname)
            f            = tarf.extractfile(fname_in_tar)
        except:
            print('Tarfile %s'%fname_tar)
            raise
    else:
        f = open('%s/Flask_%i/%s'%(path_base, flasknumber, fname))
    
    # Read footprints from open output file
    
    # Number of timesteps within each NAME simulation - note: NAME simulation
    # actually has 26 timesteps, but the first hour is the release hour so is
    # not included
    nstepNAME = 26
    
    timesteps  = np.zeros(nstepNAME, dtype=object)
    footprints = np.zeros((nstepNAME, len(latsNAME), len(lonsNAME)))
            
    for i,line in enumerate(f.readlines()[35:]):
        if tarred:
            # Is read as binary
            line = str(line)[2:]
        
        try:
            line = line.split(',')
            itime = int(line[0])-2
            ix = int(line[1])-1
            iy = int(line[2])-1
            
            if samplelayer != 'PBL':
                # There is a Z column in there which we need to skip past
                timesteps[itime] = datetime.strptime(line[4].strip(), '%d/%m/%Y %H:%M UTC')
                footprints[itime,iy,ix] = float(line[8])
            else:
                # No Z column in PBL average file
                timesteps[itime] = datetime.strptime(line[3].strip(), '%d/%m/%Y %H:%M UTC')
                footprints[itime,iy,ix] = float(line[6])
                
        except:
            print(fname)
            print(line)
            raise
            
    f.close()
    if tarred:
        tarf.close()
            
    # Sometimes all particles leave the domain before end of simulation, which means
    # timesteps will have zeros. Here I ensure there's a full timeseries
    timesteps = np.array([timesteps[0]-timedelta(seconds=3600*i) for i in range(nstepNAME)])
    
    if unit=='conc':
        
        # Simple assumptions on unit conversion regarding e.g., pressure and temperature
        Mco2 = 44.01 # gCO2 mol-1
        R    = 8.31  # J K-1 mol-1
        P    = 1e5   # kg m-1 s-2
        T    = 290   # K
        conv = (R*T)/(Mco2*P) # m3 gCO2-1 (converting between volume (e.g., ppm) and mass CO2)
        footprints *= conv*1e6 # gCO2/m3 to ppm
                    
    return timesteps, latsNAME, lonsNAME, footprints

def calculate_enhancements(footprints, emis):
    """
    Calculate enhancements from footprints and emissions in ppm
     - footprints can have units ppm or concentration depending on which file has been read
     - emissions have to be in kg/m2/s
     
    We convert emissions to footprint grid and calculate enhancements on footprint grid.
    Reason being that then we can compare gridded enhancements from different emission inventories.
    """
    
    # Footprints to [s/m]
    area_per_gridcell = helpers.calc_area_per_gridcell(lats_fp, lons_fp)
    fp  = footprints/3600.  # Correct for amount of CO2 released
    fp *= area_per_gridcell
    
    # Emissions from [kg/m2/s] to [g/m2/s]
    emis = np.copy(emis)*1e3
    
    xco2  = fp*emis_regr      # [ppm/m2]
    xco2 *= area_per_gridcell # [ppm]
    
    return xco2
    


path_base   = '/nesi/nobackup/niwa03154/nauss/cylc-run/flask_runs/' # Location of footprint tars or subdirectories
tarred      = True      # False or True
samplelayer = '40m'     # 40m, 100m, PBL
unit        = 'mixr'    # conc or mixr
flasknumber = 12585     # I choose an AUT flask number so I know I expect some enhancement

# Read footprints
tsteps_fp, lats_fp, lons_fp, footprints = read_footprint_NAME(flasknumber, path_base, samplelayer=samplelayer, tarred=tarred, unit=unit)
tsteps_fp += timedelta(seconds=12*3600) # UTC to NZST; easier to work with

print('Footprints read %4.4f'%( footprints.sum() ))

# Read emissions
# You could now read emissions for the timesteps specifically of the NAME footprints
# Here I just read weekday onroad Mahuika to get out a quick number
lats_emis, lons_emis, emis = helpers.read_mahuika_onroad() # emis in [kg/m2/s]
emis = emis[:,:,:,1]                    # Just using weekday emissions for now
emis = np.rollaxis(emis, 2, 0)          # Move time axis to front

# Regrid emissions to NAME grid
emis_regr = helpers.regrid_emissions(lats_emis, lons_emis, emis, lats_fp, lons_fp)

# Sanity check
area_per_gridcell_emis = helpers.calc_area_per_gridcell(lats_emis, lons_emis)
area_per_gridcell_fp   = helpers.calc_area_per_gridcell(lats_fp, lons_fp)
emis_tot1 = np.sum(np.mean(emis,axis=0)*area_per_gridcell_emis*3600*24*365 /1e6)    # kton/year
emis_tot2 = np.sum(np.mean(emis_regr,axis=0)*area_per_gridcell_fp*3600*24*365 /1e6) # kton/year
print("Etot before regridding = %2.2f \nEtot after regridding = %2.2f"%(emis_tot1, emis_tot2))

# We need to match hours in emissions to hours in footprints. Emissions is now 1 daily cycle,
# footprints will be 26 hours counted backwards from the end of release
emis_regr = emis_regr[[t.hour for t in tsteps_fp]] # select correct hours

print("Emissions read")

enh = calculate_enhancements(footprints, emis_regr)

print(enh.sum())
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 







