import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import timeseries
import dateutil 
import datetime

from ErrorStatistics import RMSE, Bias, InfNorm, OneNorm
from PlottingFunctions import plot_state, plot_series, plot_basic_bias

minutes_to_seconds=60.
hours_to_seconds=60.*60.
days_to_seconds=24.*60.*60.

def settings(forcing):
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
    if (forcing == 1):
        np.random.seed(forcing)
        #1.1) simple function with forcing
        '''
        noise = np.zeros(len(t))
        for i in range(1,len(t)):
            noise[i] = np.exp(-dt/6)*noise[i-1] + np.random.default_rng().normal(0, 0.2*(np.sqrt(1-np.exp(-dt/3))))
        s['h_left'] = 2.5 *np.sin(2.0*np.pi/(12.*hours_to_seconds)*t) + noise
        '''
        #2.1) read from file + forcing
        (bound_times,bound_values)=timeseries.read_series('tide_cadzand.txt')
        bound_t=np.zeros(len(bound_times))
        for i in np.arange(len(bound_times)):
            bound_t[i]=(bound_times[i]-reftime).total_seconds()
        noise = np.zeros(len(t))
        for i in range(1,len(t)):
            noise[i] = np.exp(-dt/6)*noise[i-1] + np.random.default_rng().normal(0, 0.2*(np.sqrt(1-np.exp(-dt/3))))
        s['h_left'] = np.interp(t,bound_t,bound_values) + noise
        
    else:
        #boundary (western water level)
        #1) simple function
        #s['h_left'] = 2.5 * np.sin(2.0*np.pi/(12.*hours_to_seconds)*t)
        #2) read from file
        (bound_times,bound_values)=timeseries.read_series('tide_cadzand.txt')
        bound_t=np.zeros(len(bound_times))
        for i in np.arange(len(bound_times)):
            bound_t[i]=(bound_times[i]-reftime).total_seconds()
        s['h_left'] = np.interp(t,bound_t,bound_values)   
    return s

def timestep(x,i,settings): #return (h,u) one timestep later
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
    A=A.tocsr()
    B=B.tocsr()
    settings['A']=A #cache for later use
    settings['B']=B
    return (x,t[0])

def simulate(forcing, ensemble_size): #setting forcing=1 adds noise to the western boundary condition, ensemble_size is there for plotting purposes
    # for plots
    plt.close('all')
    #fig1,ax1 = plt.subplots() #maps: all state vars at one time
    # locations of observations
    s=settings(forcing)
    L=s['L']
    dx=s['dx']
    xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
    ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int) #indices of waterlevel locations in x
    #print(ilocs)
    loc_names=[]
    names=['Cadzand','Vlissingen','Terneuzen','Hansweert','Bath']
    for i in range(len(xlocs_waterlevel)):
        loc_names.append('Averaged waterlevel at x=%.0f km %s for ensemble size %d'%(0.001*xlocs_waterlevel[i],names[i],ensemble_size))
    for i in range(len(xlocs_velocity)):
        loc_names.append('Averaged velocity at x=%.0f km %s for ensemble size %d'%(0.001*xlocs_velocity[i],names[i],ensemble_size))
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
        x=timestep(x,i,s)
        #plot_state(fig1,x,i,s) #show spatial plot; nice but slow
        series_data[:,i]=x[ilocs]
    
    return s, series_data

#main program
if __name__ == "__main__":
    ensemble_size = 4
    s, sim = simulate(1,ensemble_size)
    
    #Ensemble Kalman Filter
    #for i in range(1,ensemble_size):
    #    s, sim_new = simulate(1,ensemble_size)
    #    sim = sim +sim_new
    #sim = sim/ensemble_size

    times=s['times'][:] #[:40]
    L=s['L']
    dx=s['dx']
    t=s['t'][:]

    xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
    ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int) 
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
    print(observed_data.shape)

    #Calculate the error metrics just for the height data at the harbors 
    rmse = RMSE(sim[0:5], observed_data[0:5])
    bias = Bias(sim[0:5], observed_data[0:5])
    infnorm =InfNorm(sim[0:5], observed_data[0:5])
    onenorm =OneNorm(sim[0:5], observed_data[0:5])
    print(bias)
    print(rmse)
    print(infnorm)
    print(onenorm)

    #Plot the simulation results agains the observed data
    plot_series(s['t'],sim,s,observed_data)

    #Plot the error for each harbor
    error_plot = sim[0:5]-observed_data[0:5]
    plot_basic_bias(t, error_plot)

    plt.show()

