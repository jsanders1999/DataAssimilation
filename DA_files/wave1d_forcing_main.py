import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import timeseries
import dateutil 
import datetime
import time
from scipy.fft import fft, ifft, fftfreq
#from numba import njit

from ErrorStatistics import RMSE, Bias, InfNorm, OneNorm
from PlottingFunctions_20230601 import plot_state, plot_series, plot_ensemble_series, plot_ensemble_series_uncertainty, plot_basic_bias

minutes_to_seconds=60.
hours_to_seconds=60.*60.
days_to_seconds=24.*60.*60.

def settings(forcing, ensemble_size,n_obs,western_boundary_type):
    s=dict() #hashmap to  use s['g'] as s.g in matlab
    # Constants
    s['g']=9.81 # acceleration of gravity
    s['D']=20.0 # Depth
    s['f']=1/(0.06*days_to_seconds) # damping time scale
    L=100.e3 # length of the estuary
    s['L']=L
    n=100 #number of cells
    s['n']=n    
    # Grid(staggered water levels at 0 (boundary) dx 2dx ... (n-1)dx
    #      velocities at dx/2, 3dx/2, (n-1/2)dx
    dx=L/(n+0.5)
    s['dx']=dx
    x_h = np.linspace(0,L-dx,n)
    s['x_h'] = x_h
    s['x_u'] = x_h+0.5    
    # initial condition
    s['h_0'] = np.zeros(n)
    s['u_0'] = np.zeros(n)    
    # time
    t_f=2.*days_to_seconds #end of simulation
    dt=10.*minutes_to_seconds
    s['dt']=dt
    reftime=dateutil.parser.parse("201312050000") #times in secs relative
    s['reftime']=reftime
    t=dt*np.arange(np.round(t_f/dt)+1) #MVL moved times to end of each timestep.
    s['t']=t
    s['n_obs']=n_obs #set as 5 to use 'tide' data as observations, set as 4 to use 'waterlevel' data as observations
    if (forcing == 0):
        if s['n_obs']==5:
            (bound_times,bound_values)=timeseries.read_series('tide_cadzand.txt')
            bound_t=np.zeros(len(bound_times))
            for i in np.arange(len(bound_times)):
                bound_t[i]=(bound_times[i]-reftime).total_seconds()
            s['h_left'] = np.interp(t,bound_t,bound_values) 
        else:
            (bound_times,bound_values) = timeseries.read_series('waterlevel_vlissingen.txt')
            bound_t = np.zeros(len(bound_times))
            for i in np.arange(len(bound_times)):
                bound_t[i] = (bound_times[i]-reftime).total_seconds()
            s['h_left'] = np.interp(t,bound_t,bound_values)
    else:
        s['h_left'] = generateBoundarywNoise(dt, reftime, t, ensemble_size,western_boundary_type,forcing)
        if ensemble_size == 1:
            s['h_left'] =s['h_left'][0]
    return s

def generateBoundarywNoise(dt, reftime, t, ensemble_size,western_boundary_type,forcing=1234):
    #print("dt = ",dt)
    np.random.seed(forcing)
    #Create an array to store the noise for all the ensembles
    noise = np.zeros((ensemble_size, len(t)))
    #std of the noise
    sigma = 0.2*(np.sqrt(1-np.exp(-dt/3)))
    alpha = np.exp(-dt/6)
    #Generate the different noise forcing terms that will added to the noiseless boundary
    for i in range(1,len(t)):
        #noise[:,i] = alpha*noise[:,i-1] + np.random.normal(0, sigma, size = (ensemble_size) )
        noise[:,i] = alpha*noise[:,i-1] + np.sqrt(1-alpha**2)*np.random.normal(0, sigma, size = (ensemble_size) )

    if (western_boundary_type==1):
        (bound_times,bound_values) = timeseries.read_series('tide_cadzand.txt')
        bound_t = np.zeros(len(bound_times))
        for i in np.arange(len(bound_times)):
            bound_t[i] = (bound_times[i]-reftime).total_seconds()
        BoundaryNoNoise = np.repeat([np.interp(t,bound_t,bound_values)], ensemble_size, axis = 0)
    elif (western_boundary_type==2):
        BoundaryNoNoise = np.repeat([2.5 * np.sin(2.0*np.pi/(12.*hours_to_seconds)*t)], ensemble_size, axis = 0)
    elif (western_boundary_type==3):
        twin_data = np.loadtxt('twin_1ensemble.csv',delimiter=',')
        BoundaryNoNoise = np.repeat([twin_data[0]],ensemble_size,axis=0)
    elif(western_boundary_type==4):
        (bound_times,bound_values) = timeseries.read_series('waterlevel_vlissingen.txt')
        bound_t = np.zeros(len(bound_times))
        for i in np.arange(len(bound_times)):
            bound_t[i] = (bound_times[i]-reftime).total_seconds()
        BoundaryNoNoise = np.repeat([np.interp(t,bound_t,bound_values)], ensemble_size, axis = 0)
    elif(western_boundary_type==5):
        BoundaryNoNoise = np.repeat([np.loadtxt('cadzand_waterlevel_model.csv',delimiter=',')],ensemble_size, axis = 0)

    #Add the noise to the noiseless boundary
    #Return the boundary with the noise added
    return BoundaryNoNoise + noise


def initialize(settings): #return (h,u,t) at initial time 
    #compute initial fields and cache some things for speed
    h_0=settings['h_0']
    u_0=settings['u_0']
    n=settings['n']
    x=np.zeros(2*n) #order h[0],u[0],...h[n],u[n]
    x[0::2]=h_0[:] #MVL 20220329 swapped order
    x[1::2]=u_0[:]
    #time
    t=settings['t']
    reftime=settings['reftime']
    dt=settings['dt']
    times=[]
    second=datetime.timedelta(seconds=1)
    for i in np.arange(len(t)):
        times.append(reftime+i*int(dt)*second)
    settings['times']=times
    #initialize coefficients
    # create matrices in form A*x_new=B*x+alpha 
    # A and B are tri-diagonal sparse matrices 
    Adata=np.zeros((3,2*n)) #order h[0],u[0],...h[n],u[n]  
    Bdata=np.zeros((3,2*n))
    #left boundary
    Adata[1,0]=1.
    #right boundary
    Adata[1,2*n-1]=1.
    # i=1,3,5,... du/dt  + g dh/sx + f u = 0
    #  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m] 
    #= u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
    g=settings['g'];dx=settings['dx'];f=settings['f']
    temp1=0.5*g*dt/dx
    temp2=0.5*f*dt
    for i in np.arange(1,2*n-1,2):
        Adata[0,i-1]= -temp1
        Adata[1,i  ]= 1.0 + temp2
        Adata[2,i+1]= +temp1
        Bdata[0,i-1]= +temp1
        Bdata[1,i  ]= 1.0 - temp2
        Bdata[2,i+1]= -temp1
    # i=2,4,6,... dh/dt + D du/dx = 0
    #  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])  
    #= h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])
    D=settings['D']
    temp1=0.5*D*dt/dx
    for i in np.arange(2,2*n,2):
        Adata[0,i-1]= -temp1
        Adata[1,i  ]= 1.0
        Adata[2,i+1]= +temp1
        Bdata[0,i-1]= +temp1
        Bdata[1,i  ]= 1.0
        Bdata[2,i+1]= -temp1    
    # build sparse matrix
    A=spdiags(Adata,np.array([-1,0,1]),2*n,2*n)
    B=spdiags(Bdata,np.array([-1,0,1]),2*n,2*n)
    #print("A shape ",np.shape(A))
    #print("B shape ",np.shape(B))
    A=A.tocsr()
    B=B.tocsr()
    settings['A']=A #cache for later use
    settings['B']=B
    return (x,t[0])

def timestep_noensemble(x,i,settings): #return (h,u) one timestep later
    # take one timestep
    temp=x.copy() 
    A=settings['A']
    B=settings['B']
    rhs=B.dot(temp) #B*x
    rhs[0]=settings['h_left'][i] #left boundary
    newx=spsolve(A,rhs)
    return newx

def timestep(x,i,settings, ensemble_ind): #return (h,u) one timestep later
    # take one timestep
    temp=x.copy() 
    A=settings['A']
    B=settings['B']
    rhs=B.dot(temp) #B*x
    rhs[0]=settings['h_left'][ensemble_ind, i] #left boundary
    newx=spsolve(A,rhs)
    return newx

def get_ilocs(s):
    L=s['L']
    dx=s['dx']
    xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
    ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int)
    return xlocs_waterlevel, xlocs_velocity, ilocs

def load_observations(s):
    t=s['t'][:]

    xlocs_waterlevel, xlocs_velocity, ilocs = get_ilocs(s)

    if s['n_obs'] == 5:
        #load observations
        (obs_times,obs_values)=timeseries.read_series('tide_cadzand.txt')
        observed_data=np.zeros((len(ilocs),len(obs_times)))
        observed_data[0,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series('tide_vlissingen.txt')
        observed_data[1,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series('tide_terneuzen.txt')
        observed_data[2,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series('tide_hansweert.txt')
        observed_data[3,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series('tide_bath.txt')
        observed_data[4,:]=obs_values[:]
    elif s['n_obs'] == 4:
        (obs_times,obs_values)=timeseries.read_series('waterlevel_vlissingen.txt')
        observed_data=np.zeros((len(ilocs),len(obs_times)))
        observed_data[1,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series('waterlevel_terneuzen.txt')
        observed_data[2,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series('waterlevel_hansweert.txt')
        observed_data[3,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series('waterlevel_bath.txt')
        observed_data[4,:]=obs_values[:]
    return observed_data

def simulate(forcing): 
    # for plots
    plt.close('all')
    # locations of observations
    s=settings(forcing, ensemble_size=1,n_obs=5,western_boundary_type=1)
    L=s['L']
    dx=s['dx']
    xlocs_waterlevel, xlocs_velocity, ilocs = get_ilocs(s)
    
    loc_names=[]
    names=['Cadzand','Vlissingen','Terneuzen','Hansweert','Bath']
    for i in range(len(xlocs_waterlevel)):
        loc_names.append('Averaged waterlevel at x=%.0f km %s for ensemble size %d'%(0.001*xlocs_waterlevel[i],names[i],1))
    for i in range(len(xlocs_velocity)):
        loc_names.append('Averaged velocity at x=%.0f km %s for ensemble size %d'%(0.001*xlocs_velocity[i],names[i],1))
    s['xlocs_waterlevel']=xlocs_waterlevel
    s['xlocs_velocity']=xlocs_velocity
    s['ilocs']=ilocs
    s['loc_names']=loc_names
    #
    (x,t0)=initialize(s)
    t=s['t'][:] #[:40]
    times=s['times'][:] #[:40]
    series_data=np.zeros((len(ilocs),len(t)))
    for i in np.arange(0,len(t)):
        #print('timestep %d'%i)
        x=timestep_noensemble(x,i,s)
        #plot_state(fig1,x,i,s) #show spatial plot; nice but slow
        series_data[:,i]=x[ilocs]
    return s, series_data

def TestOneSimNoForcing(forcing=0,PrintErrors=True,ShowPlots = True):
    #Run the model for the given ensemble size
    s, sim = simulate(forcing)
    n_obs = s['n_obs']
    
    #Load the observed data
    observed_data = load_observations(s)

    #    #Calculate the error metrics just for the height data at the harbors 
    rmse = RMSE(sim[5-n_obs:5], observed_data[5-n_obs:5])
    bias = Bias(sim[5-n_obs:5], observed_data[5-n_obs:5])
    infnorm =InfNorm(sim[5-n_obs:5], observed_data[5-n_obs:5])
    onenorm =OneNorm(sim[5-n_obs:5], observed_data[5-n_obs:5])
    if PrintErrors:
        if n_obs == 5:
            print(["Locations: ", 'Cadzand','Vlissingen','Terneuzen','Hansweert','Bath'])
        else:
            print(["Locations: ", 'Vlissingen','Terneuzen','Hansweert','Bath'])
        print(["Bias: ", *bias])
        print(["RMSE: ", *rmse])
        print(["InfNorm: ", *infnorm])
        print(["OneNorm: ", *onenorm])


    if ShowPlots:
        #Plot the simulation results agains the observed data
        plot_series(s['t'],sim,s,observed_data)

        #Plot the error for each harbor
        error_plot = sim[0:5]-observed_data[0:5]
        plot_basic_bias(s['t'], error_plot)

        plt.show()
    return

def Apply_EnKF_Filter(x, ilocs, y):
    """
    x : (ensemble size, 2*n)
    """
    sigma = 0.1
    n_obs = y.shape[0]
    #print(n_obs)
    #Observation operator (n_obs, 2*n)
    H = np.zeros((n_obs,x.shape[1]))
    #Only observe the waterlevel, not the vertical velocity
    for i, loc in enumerate(ilocs[5-n_obs:5]):
        H[i,loc] = 1
    
    #covariance matric of the noise added to the observations (n_obs*n_obs)
    #R = np.eye(n_obs)*(0.2*(np.sqrt(1-np.exp(-600/3))))**2 
    R = np.eye(n_obs)*sigma**2

    #Covariance of ensemble: (2*n, 2*n)
    C = np.cov(x.T)
    #plt.imshow(C)
    #plt.show()

    #Kalman Matrix
    K = C@H.T@np.linalg.inv(H@C@H.T+R)
    #if x[0,-2]!=0:
        #plt.imshow(K)
        #plt.axes('equal')
        #plt.show()
    
    #Apply filtering step to each x[j,:] in the ensemble x[:,:]
    for j in range(x.shape[0]):
        x[j,:] = x[j,:] + K@( y + np.random.normal(0, sigma, size = n_obs) - H@x[j,:] )

    return x

def simulateEnsemble(ensemble_size, stop_filtering, forcing, n_obs, western_boundary_type, twin=False, s=None, stateplots = False): 
    # for plots
    plt.close('all')
    if stateplots:
        fig1, ax1 = plt.subplots() #maps: all state vars at one time
    # locations of observations
    if s==None:
        s=settings(forcing, ensemble_size,n_obs,western_boundary_type)
    #L=s['L']
    #dx=s['dx']
    xlocs_waterlevel, xlocs_velocity, ilocs = get_ilocs(s)
    loc_names=[]
    names=['Cadzand','Vlissingen','Terneuzen','Hansweert','Bath']
    for i in range(len(xlocs_waterlevel)):
        loc_names.append('Averaged waterlevel at x=%.0f km %s for ensemble size %d'%(0.001*xlocs_waterlevel[i],names[i],ensemble_size))
    for i in range(len(xlocs_velocity)):
        loc_names.append('Averaged velocity at x=%.0f km %s for ensemble size %d'%(0.001*xlocs_velocity[i],names[i],ensemble_size))
    s['loc_names']=loc_names
    #
    (x,t0)=initialize(s)
    #Make an array of ensemble_size realizations of x
    x_ensemble = np.tile(x, (ensemble_size, 1)) #shape: (ensemble size, 2*n)
    t=s['t'][:] #[:40]
    times=s['times'][:] #[:40]
    series_data_mean=np.zeros((len(ilocs),len(t)))
    series_data_full=np.zeros((ensemble_size, len(ilocs),len(t)))
    
    if twin:
        observed_data = np.loadtxt('twin_1ensemble.csv',delimiter=',')
    else:
        observed_data = load_observations(s)

    #fig1,ax1 = plt.subplots()
    for i in np.arange(0,len(t)):
        #print('timestep %d'%i)
        for j in range(ensemble_size):
            x_ensemble[j,:] = timestep(x_ensemble[j,:],i,s, j)

        #Apply the Stochastic Ensemble Kalman Filetering step to x_ensemble[:,:]
        y = observed_data[5-s['n_obs']:5,i]
        #print("y", y)

        #if i<60: (stop filtering after a certain time for forecasting in question 10)
        if i < np.floor(stop_filtering*len(t)/48).astype(int)+1:
            x_ensemble = Apply_EnKF_Filter(x_ensemble, ilocs, y)
        if stateplots:
            plot_state(fig1,x_ensemble[:,:],i,s, ilocs, y) #show spatial plot; nice but slow
        series_data_mean[:,i]=np.mean(x_ensemble[:, ilocs], axis = 0)#x[ilocs]
        series_data_full[:,:, i] = x_ensemble[:, ilocs]

        #Interpolate to find x at exaclty 37.25 hours (Three times the tidal period, will function as the initial condition in another simulation)
        if t[i] == 133800+3*600:
            x_3T = 1/2*np.mean(x_ensemble[:,:], axis = 0)
        elif t[i] == 134400+3*600:
            x_3T += 1/2*np.mean(x_ensemble[:,:], axis = 0)
    
    return s, series_data_mean, series_data_full, x_ensemble, x_3T

def TestEnsemble(ensemble_size,stop_filtering, forcing, n_obs, western_boundary_type,twin=False,PrintErrors=True,ShowPlots = True):
    #Run the model for the given ensemble size
    #ensemble_size = 50
    start_time = time.time()
    s, sim_mean, sim_full, x_ensemble, x_3T = simulateEnsemble(ensemble_size, stop_filtering, forcing, n_obs, western_boundary_type, twin, s=None, stateplots = False)
    #np.savetxt('twin_1ensemble.csv',sim_mean[:],delimiter=',')
    
    #Load the observed data
    #observed_data = np.loadtxt('twin_1ensemble.csv',delimiter=',')
    n_obs = s['n_obs']
    if twin:
        observed_data = np.loadtxt('twin_1ensemble.csv',delimiter=',')
    else:
        observed_data = load_observations(s)
    
    #Calculate the error metrics just for the height data at the harbors 
    rmse = RMSE(sim_mean[5-n_obs:5], observed_data[5-n_obs:5])
    bias = Bias(sim_mean[5-n_obs:5], observed_data[5-n_obs:5])
    infnorm =InfNorm(sim_mean[5-n_obs:5], observed_data[5-n_obs:5])
    onenorm =OneNorm(sim_mean[5-n_obs:5], observed_data[5-n_obs:5])
    if PrintErrors:
        if n_obs == 5:
            print(["Locations: ", 'Cadzand','Vlissingen','Terneuzen','Hansweert','Bath'])
        else:
            print(["Locations: ", 'Vlissingen','Terneuzen','Hansweert','Bath'])
        print(["Bias: ", *bias])
        print(["RMSE: ", *rmse])
        print(["InfNorm: ", *infnorm])
        print(["OneNorm: ", *onenorm])
    

    if ShowPlots:
        #Plot the simulation results agains the observed data
        plot_ensemble_series_uncertainty(s['t'],sim_full,s,observed_data,stop_filtering, n_obs, western_boundary_type)

        #Plot the error for each harbor
        error_plot = sim_mean[0:n_obs]-observed_data[0:n_obs]
        #plot_basic_bias(s['t'], error_plot)

        plt.show()
    print("Ensemble size: ",ensemble_size)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("")
    return np.mean(rmse)

def TestOneSimNoForcing(forcing,PrintErrors,ShowPlots = True):
    #Run the model for the given ensemble size
    s, sim = simulate(forcing)
    n_obs = s['n_obs']
    
    #Load the observed data
    observed_data = load_observations(s)

    #    #Calculate the error metrics just for the height data at the harbors 
    rmse = RMSE(sim[5-n_obs:5], observed_data[5-n_obs:5])
    bias = Bias(sim[5-n_obs:5], observed_data[5-n_obs:5])
    infnorm =InfNorm(sim[5-n_obs:5], observed_data[5-n_obs:5])
    onenorm =OneNorm(sim[5-n_obs:5], observed_data[5-n_obs:5])
    if PrintErrors:
        if n_obs == 5:
            print(["Locations: ", 'Cadzand','Vlissingen','Terneuzen','Hansweert','Bath'])
        else:
            print(["Locations: ", 'Vlissingen','Terneuzen','Hansweert','Bath'])
        print(["Bias: ", *bias])
        print(["RMSE: ", *rmse])
        print(["InfNorm: ", *infnorm])
        print(["OneNorm: ", *onenorm])


    if ShowPlots:
        #Plot the simulation results agains the observed data
        plot_series(s['t'],sim,s,observed_data)

        #Plot the error for each harbor
        error_plot = sim[0:5]-observed_data[0:5]
        plot_basic_bias(s['t'], error_plot)

        plt.show()
    return

def RMSEFit(x,a,b):
    return a + (x)**(-b)

def LoopEnsemble(max_ensemble_size,ensemble_increment,seed_no=1234):
    rmse_list = []
    n_loops = np.floor(max_ensemble_size/ensemble_increment).astype(int)
    for j in range(1,n_loops+1):
        rmse_list.append(TestEnsemble(j*ensemble_increment,48, seed_no, 5, 1,twin=False,PrintErrors=False,ShowPlots = False))
    return rmse_list

def ChangingSeedLoops(max_ensemble_size,ensemble_increment,n_simulations,seed_start=1234,ShowPlots=True):
    n_loops = np.floor(max_ensemble_size/ensemble_increment).astype(int)
    rmse_means = np.zeros((n_simulations,n_loops))
    x_vector = np.linspace(ensemble_increment,max_ensemble_size,n_loops)
    for i in range(n_simulations):
        print('SEED = '+str(seed_start+i))
        rmse_means[i,:] = np.array(LoopEnsemble(max_ensemble_size,ensemble_increment,seed_start+i))
    rmse_overall_mean = np.mean(rmse_means, axis=0)
    fitparams, pcov = curve_fit(RMSEFit, x_vector, rmse_overall_mean)
    if ShowPlots:
        for i in range(rmse_means.shape[0]):
            plt.plot(x_vector,rmse_means[i,:],'o',label='seed = '+str(1234+i),markerfacecolor='None')
        plt.plot(x_vector,rmse_overall_mean,'o',label='overall mean',color='black')
        plt.xlabel('Ensemble size N')
        plt.ylabel('Mean RMSE')
        plt.plot(np.linspace(ensemble_increment,max_ensemble_size,200), RMSEFit(np.linspace(ensemble_increment,max_ensemble_size,200), *fitparams),'r',label='fit: %5.3f + N^(-%5.3f)' % tuple(fitparams))
        plt.legend()
        #plt.savefig('mean_RMSE_vs_ensemble.eps', format='eps')
    plt.show()

def TestEnsembleInitialCondition(ensemble_size, start_forecasting, n_observations, western_boundary, PrintErrors, ShowPlots = True):
    #Run the model for the given ensemble size
    #ensemble_size = 50
    start_time = time.time()
    s, sim_mean_init, sim_full_init, x_ensemble_init, x_3T_init = simulateEnsemble(ensemble_size, stop_filtering=48, forcing=1234, n_obs=5, western_boundary_type=1, twin=False, s=None, stateplots = False)
    x_init = x_3T_init #at t =37.25 hours

    fig = plt.figure()
    xh=s['x_h']
    ax1=fig.add_subplot(211)
    ax1.plot(xh, x_init[0::2])
    ax1.set_ylabel('h')
    xu=s['x_u']
    ax2=fig.add_subplot(212)
    ax2.plot(xu, x_init[1::2])
    ax2.set_ylabel('u')

    if ShowPlots:
        plt.show()

    s["h_0"] = x_init[::2] 
    s["y_0"] = x_init[1::2]
    s["n_obs"] = n_observations
    s['h_left'] = generateBoundarywNoise(s['dt'], s['reftime'], s['t'], ensemble_size,western_boundary,1234)    
    s, sim_mean, sim_full, x_ensemble, x_3T = simulateEnsemble(ensemble_size, stop_filtering=start_forecasting, forcing=1234, n_obs=4, western_boundary_type=4, s = s, twin=False, stateplots = False)
    
    #Load the observed data
    n_obs = s['n_obs']
    observed_data = load_observations(s)
    peak_times = (np.floor((26.5+0.4*np.arange(5))*len(s['t'])/48)+1).astype(int)
    #print(peak_times)
    peak_diff = 0
    for i in range(5-n_obs,5):
        peak_diff = peak_diff + np.abs(sim_mean[i,peak_times[i]]-observed_data[i,peak_times[i]])/observed_data[i,peak_times[i]]
    peak_diff = peak_diff/n_obs
    
    #Calculate the error metrics just for the height data at the harbors 
    rmse = RMSE(sim_mean[5-n_obs:5], observed_data[5-n_obs:5])
    bias = Bias(sim_mean[5-n_obs:5], observed_data[5-n_obs:5])
    infnorm =InfNorm(sim_mean[5-n_obs:5], observed_data[5-n_obs:5])
    onenorm =OneNorm(sim_mean[5-n_obs:5], observed_data[5-n_obs:5])
    if PrintErrors:
        if n_obs == 5:
            print(["Locations: ", 'Cadzand','Vlissingen','Terneuzen','Hansweert','Bath'])
        else:
            print(["Locations: ", 'Vlissingen','Terneuzen','Hansweert','Bath'])
        print(["Bias: ", *bias])
        print(["RMSE: ", *rmse])
        print(["InfNorm: ", *infnorm])
        print(["OneNorm: ", *onenorm])
    

    if ShowPlots:
        #Plot the simulation results agains the observed data
        plot_ensemble_series_uncertainty(s['t'],sim_full,s,observed_data,start_forecasting, n_obs, western_boundary)

        #Plot the error for each harbor
        error_plot = sim_mean[0:n_obs]-observed_data[0:n_obs]
        #plot_basic_bias(s['t'], error_plot)

        plt.show()
    print("Ensemble size: ",ensemble_size)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("")
    return np.mean(rmse),peak_diff

def wave_subtraction(wave1,wave2):
    return fft(wave1)-fft(wave2)

def storm_surge(ShowPlots):
    s = settings(1234,1,5,1)
    loc_names=[]
    names=['Cadzand','Vlissingen','Terneuzen','Hansweert','Bath']
    filename = []
    xlocs_waterlevel, xlocs_velocity, ilocs = get_ilocs(s)
    for i in range(len(xlocs_waterlevel)):
        loc_names.append('Averaged waterlevel at x=%.0f km %s for ensemble size %d'%(0.001*xlocs_waterlevel[i],names[i],1))
    for i in range(len(xlocs_velocity)):
        loc_names.append('Averaged velocity at x=%.0f km %s for ensemble size %d'%(0.001*xlocs_velocity[i],names[i],1))
    s['n_obs'] = 5
    obs_tide = load_observations(s)
    s['n_obs'] =4
    obs_waterlevel = load_observations(s)
    plt.figure()
    plt.plot()
    t = s['t']
    #loc_names=s['loc_names']
    ntimes=min(len(t),obs_waterlevel.shape[1])
    surge = np.zeros((5,ntimes))
    surge_hour = np.zeros(5)
    start_surge = np.zeros(5)
    surge_period = 29.5
    surge_hour = 18-0.4*(4-np.arange(5,dtype=int))
    start_surge = np.floor(surge_hour*ntimes/48).astype(int)+1
    end_surge = start_surge + np.floor(surge_period*ntimes/48).astype(int)+1
    f_surge = np.zeros((5,end_surge[0]-start_surge[0]),dtype=complex)
    #print("f_surge ",np.size(f_surge[0,:]))
    f_temp = np.zeros((end_surge[0]-start_surge[0]),dtype=complex)
    if_surge = np.zeros((5,end_surge[0]-start_surge[0]))
    for i in range(5):
            surge[i,start_surge[i]:end_surge[i]] = obs_waterlevel[i,start_surge[i]:end_surge[i]]-obs_tide[i,start_surge[i]:end_surge[i]]
            f_surge[i,:] = wave_subtraction(obs_waterlevel[i,start_surge[i]:end_surge[i]],obs_tide[i,start_surge[i]:end_surge[i]])
            if i >0:
                f_temp = f_temp + f_surge[i,:]
    modeled_surge = ifft(f_temp/4)
    obs_waterlevel[0,:] = obs_tide[0,:]
    obs_waterlevel[0,start_surge[0]:end_surge[0]] = obs_waterlevel[0,start_surge[0]:end_surge[0]] + modeled_surge

    if ShowPlots:
        for i in range(5):
            fig,ax=plt.subplots()
            ax.axvspan(surge_hour[i], surge_hour[i]+surge_period, facecolor='#fffced')
            ax.fill_between(np.array(t)/3600, obs_waterlevel[i,0:ntimes], obs_tide[i,0:ntimes], facecolor='#ffe8ea') #e3ccdd 
            ax.plot(np.array(t[0:ntimes])/3600 ,obs_tide[i,0:ntimes],'o', markeredgecolor='red',markerfacecolor='white',label = "Tide data",markersize=2)
            if i==0:
                datalabel = 'Modeled waterlevel data'
            else:
                datalabel = 'Waterlevel data'
            ax.plot(np.array(t[0:ntimes])/3600 ,obs_waterlevel[i,0:ntimes],'o', markeredgecolor='black',markerfacecolor='white',label = datalabel,markersize=2)
            ax.vlines(x=surge_hour[i],ymin=-3,ymax=5,color='orange',linewidth=0.75)
            ax.vlines(x=surge_hour[i]+surge_period,ymin=-3,ymax=5,color='orange',linewidth=0.75)
            ax.set_xlabel('time [h]')
            ax.set_title('Tide vs storm surge water level at '+names[i])
            ax.set_ylabel('height [m]')
            ax.plot(np.array(t[start_surge[i]:end_surge[i]])/3600,modeled_surge,color='magenta',label='modeled surge')
            if i>0:
                ax.plot(np.array(t[start_surge[i]:end_surge[i]])/3600,surge[i,start_surge[i]:end_surge[i]],'.',color='blue',label='observed surge',markersize=1)
            ax.legend()
            filename=('storm_surge_'+names[i]+'.eps')
            #fig.savefig(filename)
        fig,ax1 = plt.subplots()
        DFT_end_index = (np.floor((end_surge[0]-start_surge[0])/2)).astype(int)
        ax1.plot(np.arange(DFT_end_index),abs(f_temp[0:DFT_end_index]/4),'o',markersize=3)
        #freq = fftfreq(end_surge[0]-start_surge[0])
        #ax1.plot(freq[0:DFT_end_index]*(2*DFT_end_index),abs(f_temp[0:DFT_end_index]/4),'o',markersize=3) #for checking
        ax1.set_title('Averaged storm surge DFT components vs time frequency index j')
        ax1.set_xlabel('time frequency index j')
        ax1.set_ylabel('DFT components')
        plt.show()
    return obs_waterlevel[0,:]

def ForecastingComparison(western_boundary,forecast_start,forecast_increment,no_forecasts,ShowPlots):
    rmse_list = []
    peak_diff_list = []
    for i in range(no_forecasts):
        rmse_mean, peak_diff = TestEnsembleInitialCondition(100, forecast_start+i, 4, western_boundary, PrintErrors =True, ShowPlots = False)
        rmse_list.append(rmse_mean)
        peak_diff_list.append(peak_diff)
    #x_vector =np.linspace(forecast_start,forecast_start+forecast_increment*no_forecasts,no_forecasts)
    x_vector = np.linspace(0,forecast_increment*no_forecasts,no_forecasts)
    rmse_list.reverse()
    peak_diff_list.reverse()
    if ShowPlots:
        fig, ax1 = plt.subplots()
        fig, ax2 = plt.subplots()
        ax1.plot(x_vector, rmse_list, 'o',label='Mean RMSE')
        ax2.plot(x_vector, peak_diff_list, 'o',label='Mean relative error at peak time')
        if western_boundary == 4:
            ax1.set_title('Mean RMSE of forecasting results starting from varying times \n with Vlissingen data as western BC')
            ax2.set_title('Peak time errors of forecasting results starting from varying times \n with Vlissingen data as western BC')
        elif western_boundary == 5:
            ax1.set_title('Mean RMSE of forecasting results starting from varying times \n with Cadzand tide data + average surge as western BC')
            ax2.set_title('Peak time errors of forecasting results starting from varying times \n with Cadzand tide data + average surge as western BC')
        ax1.set_xlabel('Lead-up time (time to peak when forecasting begins) [h]')
        ax2.set_xlabel('Lead-up time (time to peak when forecasting begins) [h]')
        ax1.set_ylabel('Mean RMSE')
        ax2.set_ylabel('Mean relative error')
        plt.show()


if __name__ == "__main__":
    answer = True
    while answer:
        print("Data Assimilation Project by Julian Sanders (4675045) and A. Mauditra A. Matin (5689252)")
        choice = input("Go to Question (choose 3, 4, 6, 7, 8, 9, or 10; enter any other number to exit): ")
        if choice == '3':
            TestOneSimNoForcing(forcing=0,PrintErrors =True,ShowPlots = True)
        elif choice =='4':
            TestOneSimNoForcing(forcing=1234,PrintErrors =True,ShowPlots = True)
        elif choice == '6':
            TestEnsemble(50,stop_filtering=48, forcing=1234, n_obs=5, western_boundary_type=1,twin=False,PrintErrors=True,ShowPlots = True)
            twin_choice = input("Show twin experiment? y/n: ")
            if twin_choice == 'y':
                TestEnsemble(50,stop_filtering=48, forcing=1234, n_obs=5, western_boundary_type=1,twin=True,PrintErrors=True,ShowPlots = True)
            elif twin_choice == 'n':
                answer = False
                break
            else: twin_choice = input("Invalid input. Enter any character: ")
        elif choice == '7':
            #VERY SLOW
            ChangingSeedLoops(1000,100,5,seed_start=1234,ShowPlots=True)
        elif choice == '8':
            print("Set 'stateplots' to 'True' in simulateEnsemble in the function TestEnsembleInitialCondition to see state plots")
            TestEnsembleInitialCondition(ensemble_size=100, start_forecasting=48, n_observations=5, western_boundary=1, PrintErrors =True, ShowPlots = True)
        elif choice == '9':
            generate_cadzand_waterlevel = input('Generate Cadzand waterlevel data? y/n: ')
            if generate_cadzand_waterlevel == 'y':
                cadzand_waterlevel_model = storm_surge(True)
                np.savetxt('cadzand_waterlevel_model.csv',cadzand_waterlevel_model,delimiter=',')
            boundary_q = True
            while boundary_q:
                boundary_choice = input("Enter 1 for Vlissingen data as western boundary condition, 2 for Cadzand tide + modeled surge data as boundary condition: ")
                if boundary_choice == '1':
                    TestEnsembleInitialCondition(ensemble_size=100, start_forecasting=48, n_observations=4, western_boundary=4, PrintErrors =True, ShowPlots = True)
                    boundary_q = False
                    break
                elif boundary_choice == '2':
                    TestEnsembleInitialCondition(ensemble_size=100, start_forecasting=48, n_observations=4, western_boundary=5, PrintErrors =True, ShowPlots = True)
                    boundary_q = False
                    break
                else:
                    boundary_choice = input("Invalid. Enter any character: ")

        elif choice == '10':
            forecast_choice = input("Enter 1 to see forecast results from one time, 2 to see comparison of errors for different forecast results: ")
            if forecast_choice == '1':
                forecast_time = input("Enter when to start forecasting (<= 48): ")
                TestEnsembleInitialCondition(ensemble_size=100, start_forecasting=int(forecast_time), n_observations=4, western_boundary=5, PrintErrors =True, ShowPlots = True)
            elif forecast_choice == '2':
                    ForecastingComparison(5,16,1,10,ShowPlots=True)
            else:
                answer = False
                break
            
        else:
            print("Invalid input. Please enter 3, 4, 6, 7, 8, 9, or 10. \n")
            answer2 = input("Would you like to continue? y/n: ")
            if answer2 == 'y':
                answer = True
            elif answer2 == 'n':
                answer = False
                break
            else:
                answer2 = input("Invalid input. Would you like to continue? y/n: ")

