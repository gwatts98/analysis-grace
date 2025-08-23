import calendar
import datetime
import importlib 
import numpy as np
import os
import sys

from parcels import Field, FieldSet, ParticleSet,Variable, JITParticle, AdvectionRK4

sys.path.append('/ocean/gwatts/home/analysis-grace')
#
from OP_functions_grace import *
#import OP_functions_shared as OP
# from Kernels import *


def timings(year, month, day, sim_length, number_outputs):
    start_time = datetime.datetime(year, month, day)
    month_days = sim_length # number of days to release particles
    data_length = max(sim_length, 1)
    duration = datetime.timedelta(days=sim_length)
    delta_t = 5 # s
    release_particles_every = 3600 # s; change this as needed: currently set to every 1 hours
    particles_per_group = 1

    number_particles = int(min(sim_length, month_days) * 86400 / release_particles_every)
    total_particles = number_particles * particles_per_group
    group_times = np.arange(0, release_particles_every * number_particles, release_particles_every)

    print ('number of particles', total_particles)

    output_interval = datetime.timedelta(seconds=sim_length * 86400 / number_outputs)
    print ('output_interval', output_interval)
    
    return (start_time, data_length, duration, delta_t, release_particles_every, number_particles, output_interval, group_times, particles_per_group, total_particles)


def name_outfile(year, month, sim_length, string):
    path = '/ocean/gwatts/home/analysis-grace/runs/hourlyhourly/'
    print (year, month, sim_length)
    fn = f'passive_particles_for_{day}-{month}-{year}_run_{sim_length}_days_'+string+'.zarr'
    return os.path.join(path, fn)


#### CREATING FIELDSETS and Setting Constants ####
def set_fieldsets_and_constants(start_time, data_length, delta_t):

    constants = {}

    # Iona Outfall Location
    constants['Iona_clat'] = [49.2022]
    constants['Iona_clon'] = [-123.3722]
    constants['Iona_z'] = 160 # m
    # constants['DepSize'] = 10 # m
    # Iona output sewage vs colloidal
    # constants['fraction_colloidal'] = 0.25 
    
    # Velocities
    varlist = ['U', 'V', 'W']
    filenames, variables = filename_set(start_time, data_length, varlist)
    dimensions = {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw','time': 'time_counter'}
    field_set = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=True, chunksize='auto')
    
    # Vertical Variables and Depth Related
    varlist=['Kz', 'totdepth', 'cell_size', 'ssh']
    filenames, variables = filename_set(start_time, data_length, varlist)
    # 2-D, no time
    dimensions = {'lon': 'glamt', 'lat': 'gphit'}
    TD = Field.from_netcdf(filenames['totdepth'], variables['totdepth'], dimensions, allow_time_extrapolation=True, chunksize='auto')
    field_set.add_field(TD)
    # 2-D, with time
    dimensions = {'lon': 'glamt', 'lat': 'gphit', 'time': 'time_counter'}
    SSH = Field.from_netcdf(filenames['ssh'], variables['ssh'], dimensions,allow_time_extrapolation=True, chunksize='auto')
    field_set.add_field(SSH)
    # 3-D on W
    dimensions = {'lon': 'glamt', 'lat': 'gphit', 'depth': 'depthw','time': 'time_counter'}
    Kz = Field.from_netcdf(filenames['Kz'], variables['Kz'], dimensions, allow_time_extrapolation=True, chunksize='auto')
    field_set.add_field(Kz)
    # 3-D on T
    dimensions = {'lon': 'glamt', 'lat': 'gphit', 'depth': 'deptht','time': 'time_counter'}
    e3t = Field.from_netcdf(filenames['cell_size'], variables['cell_size'], dimensions, allow_time_extrapolation=True, chunksize='auto')
    field_set.add_field(e3t)
    
    # dt_h = 1 / 3600. 
    # field_set.add_constant('sinkvel_sewage', 12.84 * dt_h) # m/hr * dt --> to m / s 
    # field_set.add_constant('sinkvel_marine', 5.52 * dt_h) # m/hr * dt --> to m / s 
    
    # abso = 0.038 / 86400 # Colloidal/Dissolved → Attached to Marine Particle /s 
    # deso_s = 3.2 / 86400 # Sewage Particle → Colloidal/Dissolved /s
    # deso_m = 1.6 / 86400 # Marine Particle → Colloidal/Dissolved /s
    # deso_sed = deso_m # same in the sediments
    # abso_sed = deso_sed * 30. / 70 # in the sediments, easier to find marine particles to bind to, 30/70 is ratio of suspended materials * tuning
    # # at 3 get 0.25, at 3 now getting 1.3 at 3.2 get 1.3 at 3.5 get 1.1, at 7 it was huge
    # sediment_burying = 1. / (365 * 86400) # Particles get buried by sediment
    # field_set.add_constant('abso_probability', 1 - np.exp(-abso * delta_t))
    # field_set.add_constant('deso_s_probability', 1 - np.exp(-deso_s * delta_t))
    # field_set.add_constant('deso_m_probability', 1 - np.exp(-deso_m * delta_t))
    # field_set.add_constant('deso_sed_probability', 1 - np.exp(-deso_sed * delta_t))
    # field_set.add_constant('abso_sed_probability', 1 - np.exp(-abso_sed * delta_t))
    # field_set.add_constant('sediment_burying_probability', 1 - np.exp(-sediment_burying * delta_t))
    # print (field_set.abso_probability, field_set.deso_s_probability, field_set.deso_m_probability)
    # print (field_set.abso_sed_probability, field_set.deso_sed_probability, field_set.sediment_burying_probability)
    
    # conversion factors
    deg2met = 111319.5
    latT = 0.6495
    field_set.add_constant('u_deg2mps', deg2met*latT)
    field_set.add_constant('v_deg2mps', deg2met)
    
    kappa = 0.42
    zo, rho = 0.07, 1024                              # from SalishSeaCast
    field_set.add_constant('log_z_star', np.log(zo))
    cdmin, cdmax = 0.0075, 2                          # from SalishSeaCast
    field_set.add_constant('lowere3t_o2', zo * np.exp(kappa / np.sqrt(cdmax)))
    field_set.add_constant('uppere3t_o2', zo * np.exp(kappa / np.sqrt(cdmin)))
    
    # tau_crit = 1e-2 # 0.01 0.05
    # tau_bury_crit = 0.2 # 0.8
    # field_set.add_constant('tau_constant', tau_crit / ((kappa ** 2) * rho))
    # field_set.add_constant('tau_constant_lower', tau_crit / (rho * cdmax))
    # field_set.add_constant('tau_constant_upper', tau_crit / (rho * cdmin))
    # field_set.add_constant('tau_bury_constant', tau_bury_crit / ((kappa ** 2) * rho))
    # field_set.add_constant('tau_bury_constant_lower', tau_bury_crit / (rho * cdmax))
    # field_set.add_constant('tau_bury_constant_upper', tau_bury_crit / (rho * cdmin))

    # print (field_set.tau_constant, field_set.tau_bury_constant)

    return field_set, constants


def OP_run(year, month, day, sim_length, number_outputs, string):

    # Set-up Run
    (start_time, data_length, duration, delta_t, 
         release_particles_every, number_particles, output_interval, group_times, particles_per_group, total_particles) = timings(year, month, day, sim_length, number_outputs)

    field_set, constants = set_fieldsets_and_constants(start_time, data_length, delta_t)

    outfile_states = name_outfile(year, month, sim_length, string)

    # Set-up Ocean Parcels
    class MPParticle(JITParticle):
        # status = Variable('status', initial=(np.random.rand(number_particles) >
        #                                      constants['fraction_colloidal']).astype(int) - 2)
        status = Variable('status', initial=-1)
        vvl_factor = Variable('fact', initial=1)
        # release_time = Variable('release_time', 
        #                 initial=np.arange(0, release_particles_every*number_particles, release_particles_every))
        release_time = Variable('release_time', 
                initial=np.repeat(group_times, particles_per_group))
        # size = Variable('size', initial=number_particles)

    # pset_states = ParticleSet(field_set, pclass=MPParticle, lon=constants['Iona_clon']*np.ones(number_particles), 
    #                       depth=constants['Iona_z']*np.ones(number_particles), 
    #                           lat = constants['Iona_clat']*np.ones(number_particles))
    pset_states = ParticleSet(field_set, pclass=MPParticle, lon=constants['Iona_clon']*np.ones(total_particles), 
                        depth=constants['Iona_z']*np.ones(total_particles), 
                            lat = constants['Iona_clat']*np.ones(total_particles))
   
    # pset_states = ParticleSet(field_set, pclass=MPParticle, lon=constants['Iona_clon']*np.ones(number_particles), 
    #                     depth = np.repeat(d, number_particles/len(d)), lat = constants['Iona_clat']*np.ones(number_particles))
    # pset_states = ParticleSet(field_set, pclass=MPParticle, lon=constants['Iona_clon']*np.ones(number_particles), 
    #                     depth = np.tile(d, int(np.ceil(number_particles / len(d))))[:number_particles], lat = constants['Iona_clat']*np.ones(number_particles))

    output_file = pset_states.ParticleFile(name=outfile_states, outputdt=output_interval)
    
    KE = (pset_states.Kernel(P_states) + pset_states.Kernel(Advection) + pset_states.Kernel(CheckOutOfBounds) 
        + pset_states.Kernel(KeepInOcean) + pset_states.Kernel(turb_mix))
    #     + pset_states.Kernel(export) 
    #  )
     
    # KE = (pset_states.Kernel(PBDEs_states) + pset_states.Kernel(Sinking) 
    #   + pset_states.Kernel(Advection)
    #   + pset_states.Kernel(turb_mix) + pset_states.Kernel(resuspension) 
    #   + pset_states.Kernel(CheckOutOfBounds) + pset_states.Kernel(export) 
    #   + pset_states.Kernel(KeepInOcean)
    #  )

    # Run!
    pset_states.execute(KE, runtime=duration, dt=delta_t, output_file=output_file)
    

if __name__ == "__main__":
    print("RUNNING OP_run!")
    # Input from the terminal
    year = int(sys.argv[1])  # Example: 2022
    month = int(sys.argv[2]) # Integer between 1 and 12
    day = int(sys.argv[3])
    sim_length = int(sys.argv[4]) 
    number_outputs = int(sys.argv[5])
    name_extension = str(sys.argv[6])
    
    if len(sys.argv) == 7:
        OP_run(year, month, day, sim_length, number_outputs, name_extension)
    else: 
        OP_run(year, month, day, sim_length, number_outputs)
    #
    ## How to run in the terminal:
    # python -m Susans_Model_Driver start_year start_month start_day length_sim_in_days number_outputs