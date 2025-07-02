#### ADVECTION ####
def Advection(particle, fieldset, time):
    #print('Advection kernel is running') 
    # Advection for all PBDEs in status 1, 2 and 3
    if particle.status > 0: 
        ssh = fieldset.sossheig[time, particle.depth, particle.lat, particle.lon] #SSH(t) sea surface height
        sshn = fieldset.sossheig[time+particle.dt, particle.depth, particle.lat, particle.lon] #SSH(t+dt) sea surface height in the next time step
        td = fieldset.totaldepth[time, particle.depth, particle.lat, particle.lon]#Total_depth 
        particle.fact = (1 + ssh / td)
        VVL = (sshn - ssh) * particle.depth / td
        #VVL = (sshn-ssh)*particle.depth/(td+ssh)
        #
        # calculate once and reuse
        dt_factor = 0.5 * particle.dt / particle.fact
        (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
        lon1 = particle.lon + u1*.5*particle.dt
        lat1 = particle.lat + v1*.5*particle.dt
        dep1 = particle.depth + w1 * dt_factor
        #
        (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]
        lon2 = particle.lon + u2*.5*particle.dt
        lat2 = particle.lat + v2*.5*particle.dt
        dep2 = particle.depth + w2 * dt_factor
        #
        (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]
        lon3 = particle.lon + u3*particle.dt
        lat3 = particle.lat + v3*particle.dt
        dep3 = particle.depth + w3 * 2 * dt_factor
        #
        (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]
        #
        wa = (w1 + 2*w2 + 2*w3 + w4) /6.
        particle_dlon = (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle_dlat = (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
        particle_ddepth = particle_ddepth + wa/particle.fact * particle.dt + VVL
        #
        if particle_ddepth + particle.depth < 0:
            particle_ddepth = - (2 * particle.depth + particle_ddepth)
#        tdn = fieldset.totaldepth[time, particle.depth + particle_ddepth, 
#                        particle.lat+particle_dlat, particle.lon+particle_dlon]
#        # advect into bottom: onto bottom
#        if particle_ddepth + particle.depth > tdn:
#            particle_ddepth = 2 * tdn - (2* particle.depth + particle_ddepth)
#
#
#### TURBULENT MIX ####
def turb_mix(particle,fieldset,time):
    if particle.status > 0:
        """Vertical mixing"""
        #Vertical mixing
        if particle.depth + 0.5 / particle.fact > td: #Only calculate gradient of diffusion for particles deeper than 0.5 otherwise OP will check for particles outside the domain and remove it.
            Kzdz = 2 * (fieldset.vert_eddy_diff[time, particle.depth, particle.lat, particle.lon] 
                        - fieldset.vert_eddy_diff[time, particle.depth-0.5/particle.fact, particle.lat, particle.lon]) #backwards difference 
        else: 
            Kzdz = 2 * (fieldset.vert_eddy_diff[time, particle.depth+0.5/particle.fact, particle.lat, particle.lon]
                        - fieldset.vert_eddy_diff[time, particle.depth, particle.lat, particle.lon]) #forward difference 
        dgrad = Kzdz * particle.dt / particle.fact
        if particle.depth + (0.5 * dgrad) > 0 and particle.depth + (0.5 * dgrad) < td:
            Kz = fieldset.vert_eddy_diff[time, particle.depth+0.5*dgrad, particle.lat, particle.lon] #Vertical diffusivity SSC  
        else:
            Kz = 0 
        #
        Rr = ParcelsRandom.uniform(-1, 1)
        d_random = sqrt(3 * 2 * Kz * particle.dt) * Rr / particle.fact
        dzs = (dgrad + d_random)
        
        #Apply turbulent mixing.       
        # reflect if mixed into bottom
        tdn = fieldset.totaldepth[time, particle.depth + particle_ddepth, 
                        particle.lat+particle_dlat, particle.lon+particle_dlon]
        if dzs + particle_ddepth + particle.depth > tdn:
            particle.depth = tdn #2 * tdn - (2* particle.depth + particle_ddepth + dzs)
            particle_ddepth = 0
            # add status for sedimented (11, 12 or 13) !!
        #
        elif dzs + particle.depth + particle_ddepth < 0:
            particle_ddepth = -(dzs + 2 * particle.depth + particle_ddepth) #reflection on surface
        #
        else:
            particle_ddepth += dzs #apply mixing    



def P_states(particle, fieldset, time):    
        
    if (time > particle.release_time):
        if particle.status < 0:
            particle.status = - particle.status


#
#### OTHERS ####
def export(particle,fieldset,time):
    if particle.lat<48.7 and particle.lon < -124.66:
        particle.status = 7

def CheckOutOfBounds(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds:    
        particle.delete()
#        
def KeepInOcean(particle, fieldset, time):
    if particle.state == StatusCode.ErrorThroughSurface:
        particle.depth = 0.0
        particle.state = StatusCode.Success  
        