import numpy as np
import os
import sys
import math
from math import fmod
import pandas as pd
import xarray as xr 
import yaml
# from datetime import datetime, timedelta
import datetime
from parcels import FieldSet, Field, VectorField, ParticleSet, JITParticle, ParcelsRandom, Variable

sys.path.append('/ocean/gwatts/home/analysis-grace') #Add directory where OP_Kernels is located.
# from PBDEs_OP_Kernels_V2 import * # change between files before running! Just to make sure that you are using the right Kernels :D
 
def path(local = 1):
    '''Change with your paths'''
    if local == 1:
        path = {'NEMO': '/results2/SalishSea/nowcast-green.202111/',
        'coords': '/ocean/vvalenzuela/MOAD/grid/coordinates_seagrid_SalishSea201702.nc',
        'coordsWW3': '/ocean/vvalenzuela/MOAD/grid2/WW3_grid.nc',
        'mask': '/ocean/vvalenzuela/MOAD/grid2/mesh_mask202108_TDV.nc',
        'bat': '/ocean/vvalenzuela/MOAD/grid/bathymetry_202108.nc',
        'out': '/home/vvalenzuela/MOAD/Ocean_Parcels/results/Mixing_Advection_Sinking',
        'home': '/home/vvalenzuela/MOAD/analysis-vicente/Ocean_Parcels',
        'anim': '/home/vvalenzuela/MOAD/Ocean_Parcels/animations',
        'fondo' : '/ocean/vvalenzuela/MOAD/analysis-vicente/Ocean_Parcels/drafts/PBDE_runs/timeless_mbathy.nc'}
    else:
        path = {'NEMO': '/results2/SalishSea/nowcast-green.202111/',
        'coords': '/ocean/vvalenzuela/MOAD/grid/coordinates_seagrid_SalishSea201702.nc',
        'coordsWW3': '/ocean/vvalenzuela/MOAD/grid2/WW3_grid.nc',
        'mask': '/ocean/vvalenzuela/MOAD/grid2/mesh_mask202108_TDV.nc',
        'bat': '/ocean/vvalenzuela/MOAD/grid/bathymetry_202108.nc',
        'out': '/home/vvalenzuela/MOAD/Ocean_Parcels/results/Mixing_Advection_Sinking',
        'home': '/home/vvalenzuela/MOAD/analysis-vicente/Ocean_Parcels',
        'anim': '/home/vvalenzuela/MOAD/Ocean_Parcels/animations',
        'fondo' : '/ocean/vvalenzuela/MOAD/analysis-vicente/Ocean_Parcels/drafts/PBDE_runs/timeless_mbathy.nc'}
    return path


def make_prefix(date, path, res='h'):
    """Construct path prefix for local SalishSeaCast results given date object and paths dict
    e.g., /results2/SalishSea/nowcast-green.201905/daymonthyear/SalishSea_1h_yyyymmdd_yyyymmdd
    """

    datestr = '_'.join(np.repeat(date.strftime('%Y%m%d'), 2))
    folder = date.strftime("%d%b%y").lower()
    prefix = os.path.join(path, f'{folder}/SalishSea_1{res}_{datestr}')
    
    return prefix


def output(outfile,local=1):
    '''coords,mask,ds = output(outfile,local=1) 
    outfile is the name of the output file'''
    paths = path(local)
    coords = xr.open_dataset(paths['coords'], decode_times=False)
    mask = xr.open_dataset(paths['mask'])
    ds = xr.open_dataset(outfile)
    return  coords,mask,ds
    
    
def filename_set(start,length,varlist=['U','V','W'],local=0):
    '''filename,variables,dimensions = filename_set(start,duration,varlist=['U','V','W'],local=1)
    Modify function to include more default variables
    define start as: e.g, datetime(2018, 1, 17)
    length= number of days'''
    
    duration = datetime.timedelta(days=length)
    #Build filenames
    paths = path(local)
    Rlist,Tlist,Ulist, Vlist, Wlist = [], [], [], [], []
    Waveslist = []
    Flist = []
    Biolist,MZlist = [],[]
    Biology_list = []
   #
    for day in range(duration.days):
        path_NEMO = make_prefix(start + datetime.timedelta(days=day), paths['NEMO'])
        path_NEMO_d = make_prefix(start + datetime.timedelta(days=day), paths['NEMO'],res='d')
        Ulist.append(path_NEMO + '_grid_U.nc')
        Vlist.append(path_NEMO + '_grid_V.nc')
        Wlist.append(path_NEMO + '_grid_W.nc')
        Tlist.append(path_NEMO + '_grid_T.nc')
        Biolist.append(path_NEMO_d + '_prod_T.nc')
        Biology_list.append(path_NEMO + '_biol_T.nc')
        Waveslist.append(get_WW3_path(start + datetime.timedelta(days=day)))
        Flist.append(get_Fraser_path(start + datetime.timedelta(days=day)))
        

    # Load NEMO forcing 
    filenames = {
        'U': {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Wlist[0], 'data': Ulist},
        'V': {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Wlist[0], 'data': Vlist},
        'W': {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Wlist[0], 'data': Wlist},
        'Kz': {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Wlist[0], 'data': Wlist},
        'T': {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Tlist[0], 'data': Tlist},
        'S': {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Tlist[0], 'data': Tlist},
        'ssh': {'lon': paths['coords'], 'lat': paths['coords'], 'data': Tlist},
        'R': {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Tlist[0], 'data': Tlist},
        'Bathy' : {'lon': paths['coords'], 'lat': paths['coords'], 'data': paths['bat']},
        'gdepth' : {'lon': paths['coords'], 'lat': paths['coords'],'depth': Wlist[0], 'data': paths['mask']},
        'totdepth' : {'lon': paths['coords'], 'lat': paths['coords'], 'data': paths['mask']},
        'US' : {'lon': paths['coordsWW3'], 'lat': paths['coordsWW3'], 'data': Waveslist},
        'VS' : {'lon': paths['coordsWW3'], 'lat': paths['coordsWW3'], 'data': Waveslist},
        'WL' : {'lon': paths['coordsWW3'], 'lat': paths['coordsWW3'], 'data': Waveslist},
        'FS' :  {'lon': paths['coords'], 'lat': paths['coords'],'data': Flist},
        'Diat' : {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Tlist[0], 'data': Biolist},
        'Flag' : {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Tlist[0], 'data': Biolist},
        'Vol' : {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Wlist[0],'data': paths['mask']},
        'PON' : {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Tlist[0], 'data': Biology_list},
        'DIATO' : {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Tlist[0], 'data': Biology_list},
        'FLAGE' : {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Tlist[0], 'data': Biology_list},
        'DON' : {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Tlist[0], 'data': Biology_list},
        'cell_size' : {'lon': paths['coords'], 'lat': paths['coords'], 'depth': Tlist[0], 'data': Tlist},
        'last_cell_index' : {'lon': paths['coords'], 'lat': paths['coords'], 'data': paths['fondo']}
    }
    variables = {'U': 'vozocrtx', 'V': 'vomecrty','W': 'vovecrtz','T':'votemper','S':'vosaline','R':'sigma_theta',
        'US':'uuss','VS':'vuss','WL':'lm','Bathy':'Bathymetry','FS':'rorunoff','Kz':'vert_eddy_diff',
        'MZ':'microzooplankton','Diat':'PPDIATNO3','Flag':'PPPHYNO3','ssh':'sossheig','totdepth':'totaldepth','Vol':'volume', 'PON' : 'particulate_organic_nitrogen',
        'DIATO' : 'diatoms', 'FLAGE' : 'flagellates', 'DON' : 'dissolved_organic_nitrogen', 'cell_size' : 'e3t', 'last_cell_index' : 'mbathy'}
        
    file2,var2 = {},{}
    for var in varlist:
        file2[var]=filenames[var]
        var2[var]=variables[var]
    return file2,var2

def p_deploy(N,n,dmin,dd,r = 1000):
    #r is radius of particle cloud [m]
    deg2m = 111000 * np.cos(50 * np.pi / 180)
    var = (r / (deg2m * 3))**2
    x_offset, y_offset = np.random.multivariate_normal([0, 0], [[var, 0], [0, var]], [n,N]).T
    if isinstance(dmin,int):
        zvals1 = dmin + np.random.random_sample([n,N]).T*(dd)
    else:
        zvals = []
        zvals1 = []
        for dept in dmin:
            zvals.append(dept + np.random.random_sample([n]).T*(dd))
        for i in range(len(zvals)):   
            zvals1=np.concatenate((zvals1[:],zvals[i]))
    return x_offset, y_offset, zvals1


def get_WW3_path(date):
    """Construct WW3 results path given the date
    e.g., /opp/wwatch3/nowcast/SoG_ww3_fields_YYYYMMDD_YYYYMMDD.nc
    :arg date: date of WW3 record
    :type date: :py:class:`datetime.datetime`
    :returns: WW3 path
    :rtype: str
    """
    # Make WW3 path
    path = '/opp/wwatch3/hindcast'
    path2 = '/opp/wwatch3/nowcast'
    datestr = [date.strftime(fmt) for fmt in ('%d%b%y', '%Y%m%d_%Y%m%d')]
    path = os.path.join(path, datestr[0].lower(), f'SoG_ww3_fields_{datestr[1]}.nc')
    if not os.path.exists(path):
        path = os.path.join(path2, datestr[0].lower(), f'SoG_ww3_fields_{datestr[1]}.nc')
        if not os.path.exists(path):    
            raise ValueError(f"No WW3 record found for the specified date {date.strftime('%Y-%b-%d')}")

    return path

def homodist(Ni):
    ff = '/home/jvalenti/MOAD/analysis-jose/Source/'
    Tlat = {1:'clat.txt',2:'clat2.txt'}
    Tlon = {1:'clon.txt',2:'clon2.txt'}
    with open(ff+Tlat[Ni]) as f:
        clat = f.read()
        clat= clat[1:-1]
        clat0 = clat.split(",")
        f.close()
    with open(ff+Tlon[Ni]) as f:
        clon = f.read()
        clon=clon[1:-1]
        clon0 = clon.split(",")
        f.close()
    clat,clon=[],[]
    for i in range(len(clat0)):
        clat.append(float(clat0[i]))
        clon.append(float(clon0[i]))
    return clat, clon

def dist_coord(LAT,LON):
    la1,la2 = LAT[0],LAT[-1]
    lo1,lo2 = LON[0],LON[-1]
    R = 6378137
    PI=math.pi
    lat1 = la1 * PI/180 
    lat2 = la2 * PI/180
    dlat = (lat2-lat1)
    dlon = (lo1-lo2) * PI/180;
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d/1e3

def transect_deploy(llat,llon,N):
    '''use decimal coordinates, only works without coordinate sign changes'''
    dxlat = abs(llat[0]-llat[1])/(N-1)
    dxlon = abs(llon[0]-llon[1])/(N-1)
    clat = []
    clon = []
    for i in range(N):
        clat.append(llat[0]+i*dxlat)
        clon.append(llon[0]+i*dxlon)
    return clat, clon

def p_unidist(dy,dx):
    latbat = np.array(pd.read_csv("/home/jvalenti/MOAD/analysis-jose/Source/lat_bat.csv").drop('Unnamed: 0', axis=1))
    lonbat = np.array(pd.read_csv("/home/jvalenti/MOAD/analysis-jose/Source/lon_bat.csv").drop('Unnamed: 0', axis=1))
    yi = np.arange(0,latbat.shape[0],dy)
    xi = np.arange(0,latbat.shape[1],dx)
    plat0,plon0 = latbat[yi,:],lonbat[yi,:]
    plat1,plon1 = plat0[:,xi],plon0[:,xi]
    plon,plat = [],[]
    for i in range(plat1.shape[0]):
        for j in range(plat1.shape[1]):
            if plat1[i,j]>1e-5:
                plat.append(plat1[i,j])
                plon.append(plon1[i,j])
    return plat,plon

def load_config1(config_yaml):
   with open(config_yaml) as f:
       config = yaml.safe_load(f)
   return config

def load_config(config_yaml):
   with open(config_yaml[0]) as f:
       config = yaml.safe_load(f)
   return config


def get_Fraser_path(date):
    """Construct Fraser river outflow path given the date
    e.g., /results/forcing/rivers/#R201702DFraCElse_yYYYYmMMdDD.nc
    :arg date: date of fraser outflow record 
    :type date: :py:class:`datetime.datetime`
    :returns: nc path
    :rtype: str
    """

    # Make Fraser path
    path = '/results/forcing/rivers/'
    datestr = [date.strftime(fmt) for fmt in ('%d%b%y', 'y%Ym%md%d')]
    path = os.path.join(path, f'R201702DFraCElse_{datestr[1]}.nc')
    if not os.path.exists(path):
        raise ValueError(f"No file found for the specified date {date.strftime('%Y-%b-%d')}")
    return path

def particle_maker(config):
    #Define particle properties 
    class MPParticle(JITParticle):
        if 'ro' in config['particle']:        
            ro = Variable('ro', initial = config['particle']['ro'])  # config['particle']['ro']
        if 'diameter' in config['particle']:           
            diameter = Variable('diameter', initial = config['particle']['diameter'])
        if 'SDD' in config['particle']:  
            SDD = Variable('SDD', initial = config['particle']['SDD'])
        if 'SDL' in config['particle']:           
            SDL = Variable('SDL', initial = config['particle']['SDL'])
        if 'length' in config['particle']:  
            length = Variable('length', initial = config['particle']['length'])
        if 'Lb' in config['particle']:  
            Lb = Variable('Lb', initial = config['particle']['Lb'])  #days needed in days for particle to have 67% probability of beaching if in beaching zone (500m)
        if 'Db' in config['particle']:  
            Db = Variable('Db', initial = config['particle']['Db']) #Distance at which particles can randomly beach.
        if 'Ub' in config['particle']:  
            Ub = Variable('Ub', initial = config['particle']['Ub'])  #days to have 67% probability of unbeaching
        if 'beached' in config['particle']:  
            beached = Variable('beached', initial = 0)
        if 'surf' in config['particle']:  
            surf = Variable('surf', initial =  0)   
        if 'fact' in config['particle']:  
            fact = Variable('fact', initial =  1)    
        if 'ws' in config['particle']:  
            ws = Variable('ws', initial =  0) 
        if 'tau' in config['particle']:  
            tau = Variable('tau', initial =  config['particle']['tau']) # track number of particles
        if 'fratio' in config['particle']:  
            fratio = Variable('fratio', initial =  config['particle']['fratio']) # track number of particles
        if 'Nbac' in config['particle']:  
            Nbac = Variable('Nbac', initial =  0) # number of bacteria attached
        if 'Nflag' in config['particle']:  
            Nflag = Variable('Nflag', initial =  0) # number of flagellates grazing on the attached bacteria
        if 'wa' in config['particle']:  
            wa = Variable('wa', initial =  0) # dz variable
        if 'Kh' in config['particle']:  
            Kh = Variable('Kh', initial =  config['particle']['Kh']) # Kh horizontal diff
        if 'MFcount' in config['param']:  
            MFcount = Variable('MFcount', initial =  config['param']['MFc']) # MF per parcel
        if 'cellvol' in config['particle']:  
            cellvol = Variable('cellvol', initial =  config['particle']['cellvol']) # MF per parcel
        if 'dtmax' in config['particle']:  
            dtmax = Variable('dtmax', initial =  86400*config['particle']['dtmax']) # max time run
        else:
            dtmax = Variable('dtmax', initial =  86400*5e3) #! max time run 14 years
    
    return MPParticle

def pandas_deploy(N,MFc,dtp):
    MFc = float(MFc)
    Rivers_deploy = pd.read_csv(N, index_col = [0])
    Pol = list(Rivers_deploy.Population_polluting)
    Lat = Rivers_deploy.Latitude
    Lon = Rivers_deploy.Longitude
    clat = []
    clon = []
    for i,loc in enumerate(Pol):
        for j in range(int(round((loc*250*dtp)/MFc,0))):
            clat.append(Lat.iat[i])
            clon.append(Lon.iat[i])
    return clat, clon, len(clat)   