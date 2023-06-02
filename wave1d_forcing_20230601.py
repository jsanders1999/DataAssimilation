import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import timeseries
import dateutil 
import datetime
import time
#from numba import njit

from ErrorStatistics import RMSE, Bias, InfNorm, OneNorm
from PlottingFunctions_20230601 import plot_state, plot_series, plot_ensemble_series, plot_ensemble_series_uncertainty, plot_basic_bias

minutes_to_seconds=60.
hours_to_seconds=60.*60.
days_to_seconds=24.*60.*60.

def settings(forcing, ensemble_size):
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
    s['n_obs']=4 #set as 5 to use 'tide' data as observations, set as 4 to use 'waterlevel' data as observations
    if (forcing == 1):
        s['h_left'] = generateBoundarywNoise(dt, reftime, t, ensemble_size,4)
    else:
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
    return s

def generateBoundarywNoise(dt, reftime, t, ensemble_size,type):
    #print("dt = ",dt)
    np.random.seed(1238)
    #Create an array to store the noise for all the ensembles
    noise = np.zeros((ensemble_size, len(t)))
    #std of the noise
    sigma = 0.2*(np.sqrt(1-np.exp(-dt/3)))
    alpha = np.exp(-dt/6)
    #Generate the different noise forcing terms that will added to the noiseless boundary
    for i in range(1,len(t)):
        #noise[:,i] = alpha*noise[:,i-1] + np.random.normal(0, sigma, size = (ensemble_size) )
        noise[:,i] = alpha*noise[:,i-1] + np.sqrt(1-alpha**2)*np.random.normal(0, sigma, size = (ensemble_size) )

    if (type==1):
        (bound_times,bound_values) = timeseries.read_series('tide_cadzand.txt')
        bound_t = np.zeros(len(bound_times))
        for i in np.arange(len(bound_times)):
            bound_t[i] = (bound_times[i]-reftime).total_seconds()
        BoundaryNoNoise = np.repeat([np.interp(t,bound_t,bound_values)], ensemble_size, axis = 0)
    elif (type==2):
        BoundaryNoNoise = np.repeat([2.5 * np.sin(2.0*np.pi/(12.*hours_to_seconds)*t)], ensemble_size, axis = 0)
    elif (type==3):
        twin_data = np.loadtxt('twin_1ensemble.csv',delimiter=',')
        BoundaryNoNoise = np.repeat([twin_data[0]],ensemble_size,axis=0)
    elif(type==4):
        (bound_times,bound_values) = timeseries.read_series('waterlevel_vlissingen.txt')
        bound_t = np.zeros(len(bound_times))
        for i in np.arange(len(bound_times)):
            bound_t[i] = (bound_times[i]-reftime).total_seconds()
        BoundaryNoNoise = np.repeat([np.interp(t,bound_t,bound_values)], ensemble_size, axis = 0)

    #Add the noise to the noiseless boundary
    #Return the boundary with the noise added
    return BoundaryNoNoise + noise

def timestep(x,i,settings, ensemble_ind): #return (h,u) one timestep later
    # take one timestep
    temp=x.copy() 
    A=settings['A']
    B=settings['B']
    rhs=B.dot(temp) #B*x
    rhs[0]=settings['h_left'][ensemble_ind, i] #left boundary
    newx=spsolve(A,rhs)
    return newx

def timestep_noensemble(x,i,settings): #return (h,u) one timestep later
    # take one timestep
    temp=x.copy() 
    A=settings['A']
    B=settings['B']
    rhs=B.dot(temp) #B*x
    rhs[0]=settings['h_left'][i] #left boundary
    newx=spsolve(A,rhs)
    return newx

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

def get_ilocs(s):
    L=s['L']
    dx=s['dx']
    xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
    ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int)
    '''
    if s['n_obs']==5:
        xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
        xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
        ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int) 
    elif s['n_obs']==4:
        xlocs_waterlevel=np.array([0.25*L,0.5*L,0.75*L,0.99*L])
        xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
        ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int)
    '''
    return xlocs_waterlevel, xlocs_velocity, ilocs

def load_observations(s):
    times=s['times'][:] #[:40]
    #L=s['L']
    #dx=s['dx']
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



def simulate(forcing): #setting forcing=1 adds noise to the western boundary condition
    # for plots
    plt.close('all')
    #fig1,ax1 = plt.subplots() #maps: all state vars at one time
    # locations of observations
    s=settings(forcing, 1)
    L=s['L']
    dx=s['dx']
    xlocs_waterlevel, xlocs_velocity, ilocs = get_ilocs(s)
    
    #xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    #xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
    #ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int) #indices of waterlevel locations in x
    #print(ilocs)
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

def simulateEnsemble(ensemble_size): 
    # for plots
    plt.close('all')
    #fig1,ax1 = plt.subplots() #maps: all state vars at one time
    # locations of observations
    s=settings(1, ensemble_size) #forcing is always 1 for the ensembles
    L=s['L']
    dx=s['dx']
    xlocs_waterlevel, xlocs_velocity, ilocs = get_ilocs(s)
    #xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    #xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
    #ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int) #indices of waterlevel locations in x
    #print(ilocs)
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
    
    observed_data = load_observations(s)
    #observed_data = np.loadtxt('twin_1ensemble.csv',delimiter=',')

    #fig1,ax1 = plt.subplots()
    for i in np.arange(0,len(t)):
        #print('timestep %d'%i)
        for j in range(ensemble_size):
            x_ensemble[j,:] = timestep(x_ensemble[j,:],i,s, j)
        #Apply the Stochastic Ensemble Kalman Filetering step to x_ensemble[:,:]
        
        y = observed_data[5-s['n_obs']:5,i]
        #print("y", y)

        #if i<60: (stop filtering after a certain time for forecasting in question 10)
        x_ensemble = Apply_EnKF_Filter(x_ensemble, ilocs, y)
        #plot_state(fig1,x_ensemble[:,:],i,s) #show spatial plot; nice but slow
        series_data_mean[:,i]=np.mean(x_ensemble[:, ilocs], axis = 0)#x[ilocs]
        series_data_full[:,:, i] = x_ensemble[:, ilocs]

    
    return s, series_data_mean, series_data_full

def Apply_EnKF_Filter(x, ilocs, y):
    """
    x : (ensemble size, 2*n)
    """
    n_obs = y.shape[0]
    #print(n_obs)
    #Observation operator (n_obs, 2*n)
    H = np.zeros((n_obs,x.shape[1]))
    #Only observe the waterlevel, not the vertical velocity
    for i, loc in enumerate(ilocs[5-n_obs:5]):
        H[i,loc] = 1
    
    #covariance matric of the noise added to the observations (n_obs*n_obs)
    #R = np.eye(n_obs)*(0.2*(np.sqrt(1-np.exp(-600/3))))**2 
    R = np.eye(n_obs)*0.01

    #Covariance of ensemble: (2*n, 2*n)
    C = np.cov(x.T)
    #plt.imshow(C)
    #plt.show()

    #Kalman Matrix
    K = C@H.T@np.linalg.inv(H@C@H.T+R)
    #plt.imshow(K)
    #plt.show()
    
    #Apply filtering step to each x[j,:] in the ensemble x[:,:]
    for j in range(x.shape[0]):
        x[j,:] = x[j,:] + K@( y - H@x[j,:] )

    return x

def TestEnsemble(ensemble_size,PrintErrors,ShowPlots = True):
    #Run the model for the given ensemble size
    #ensemble_size = 50
    start_time = time.time()
    s, sim_mean, sim_full = simulateEnsemble(ensemble_size)
    #np.savetxt('twin_1ensemble.csv',sim_mean[:],delimiter=',')
    
    #Load the observed data
    #observed_data = np.loadtxt('twin_1ensemble.csv',delimiter=',')
    n_obs = s['n_obs']
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
        plot_ensemble_series_uncertainty(s['t'],sim_full,s,observed_data)

        #Plot the error for each harbor
        error_plot = sim_mean[0:n_obs]-observed_data[0:n_obs]
        #plot_basic_bias(s['t'], error_plot)

        plt.show()
    print("Ensemble size: ",ensemble_size)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("")
    return np.mean(rmse)

def TestOneSimNoForcing(PrintErrors,ShowPlots = True):
    #Run the model for the given ensemble size
    s, sim = simulate(forcing = 0)
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

def LoopEnsemble(max_ensemble_size,ensemble_increment,ShowPlots=True):
    rmse_list = []
    n_loops = np.floor(max_ensemble_size/ensemble_increment).astype(int)
    x_vector = np.linspace(ensemble_increment,max_ensemble_size,n_loops)
    for j in range(1,n_loops+1):
        rmse_list.append(TestEnsemble(j*ensemble_increment,False,False))
    print(rmse_list)
    popt, pcov = curve_fit(RMSEFit, x_vector, rmse_list)
    if ShowPlots:
        plt.figure()
        plt.plot(np.linspace(ensemble_increment,max_ensemble_size,num=200), RMSEFit(np.linspace(ensemble_increment,max_ensemble_size,num=200), *popt), 'r-',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
        plt.plot(x_vector, rmse_list,'o',label='Mean RMSE')
        plt.title('Mean RMSE and fitted mean RMSE function')
        plt.xlabel('Ensemble size N')
        plt.ylabel('Mean RMSE')
        plt.legend()
        plt.show()
    return rmse_list

    
if __name__ == "__main__":

    #LoopEnsemble(1000,100,True)

    TestEnsemble(100,True,True)
    #TestOneSimNoForcing(True,True)

