import numpy as np
import matplotlib.pyplot as plt

def plot_state(fig,x,i,s, ilocs, y):
    #plot all waterlevels and velocities at one time
    fig.clear()
    xh=s['x_h']
    L=s['L']
    dx=s['dx']
    xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    ax1=fig.add_subplot(211)
    for j in range(x.shape[0]):
        ax1.plot(xh,x[j, 0::2])
    ax1.plot(xlocs_waterlevel, y, marker = "o", linestyle = "None" )
        #print(ilocs, y)
    ax1.set_ylabel('h')
    xu=s['x_u']
    ax2=fig.add_subplot(212)
    for j in range(x.shape[0]):
        ax2.plot(xu,x[j, 1::2])
    ax2.set_ylabel('u')
    #plt.savefig("fig_map_%3.3d.png"%i)
    plt.draw()
    plt.pause(0.1)

    #plt.show()
    #return

def plot_series(t,series_data,s,obs_data):
    # plot timeseries from model and observations
    loc_names=s['loc_names']
    nseries=len(loc_names)
    for i in range(nseries):
        fig,ax=plt.subplots()
        ax.plot(np.array(t)/3600,series_data[i,:],'b-', label = "Model data")
        ax.set_title(loc_names[i])
        ax.set_xlabel('time [hours]')
        if i<5:
            ax.set_ylabel('height [m]')
        else:
            ax.set_ylabel('velocity [m/s]')
        ntimes=min(len(t),obs_data.shape[1])
        ax.plot(np.array(t[0:ntimes])/3600 ,obs_data[i,0:ntimes],'k-', label = "Observed data")
        ax.legend()
        #plt.savefig(("%s.png"%loc_names[i]).replace(' ','_'))

def plot_ensemble_series(t,series_data,s,obs_data):
    # plot timeseries from model and observations
    loc_names=s['loc_names']
    nseries=len(loc_names)
    for i in range(nseries):
        fig,ax=plt.subplots()
        for n in range(series_data.shape[0]):
            ax.plot(np.array(t)/3600,series_data[n,i,:],'b-')#, label = "Model data {}".format{n})
        ax.set_title(loc_names[i])
        ax.set_xlabel('time [hours]')
        if i<5:
            ax.set_ylabel('height [m]')
        else:
            ax.set_ylabel('velocity [m/s]')
        ntimes=min(len(t),obs_data.shape[1])
        ax.plot(np.array(t[0:ntimes])/3600 ,obs_data[i,0:ntimes],'k-', label = "Observed data")
        ax.legend()
        #plt.savefig(("%s.png"%loc_names[i]).replace(' ','_'))

def plot_ensemble_series_uncertainty(t,series_data,s,obs_data):
    # plot timeseries from model and observations
    if s['n_obs'] == 5:
        loc_names=s['loc_names']
        nseries=len(loc_names)
        mean_arr = np.mean(series_data[:,:,:], axis = 0)
        std_arr = np.std(series_data[:,:,:], axis = 0)  
        print("st. deviation: ", np.std(series_data[:,:,:], axis = (0,2)))
        print(np.shape(obs_data))
        for i in range(nseries):
            fig,ax=plt.subplots()
            ntimes=min(len(t),obs_data.shape[1])
            ax.plot(np.array(t[0:ntimes])/3600 ,obs_data[i,0:ntimes],'o', markeredgecolor='black',markerfacecolor='None',label = "Observed data",markersize=2)
            ax.fill_between(np.array(t)/3600, mean_arr[i,:] - std_arr[i,:], mean_arr[i,:] + std_arr[i,:], alpha=0.2,facecolor='blue')
            ax.plot(np.array(t)/3600,mean_arr[i,:] + std_arr[i,:],'b--',linewidth=0.75)
            ax.plot(np.array(t)/3600,mean_arr[i,:] - std_arr[i,:],'b--',linewidth=0.75)
            ax.plot(np.array(t)/3600,mean_arr[i,:],'b-', label = "Ensemble data",linewidth=1.25)
            ax.set_title(loc_names[i])
            ax.set_xlabel('time [hours]')
            if i<5:
                ax.set_ylabel('height [m]')
            else:
                ax.set_ylabel('velocity [m/s]')
            
            
            ax.legend()
            #plt.savefig(("%s.png"%loc_names[i]).replace(' ','_'))
    elif s['n_obs'] ==4:
        loc_names=s['loc_names']
        nseries=len(loc_names)
        mean_arr = np.mean(series_data[:,:,:], axis = 0)
        std_arr = np.std(series_data[:,:,:], axis = 0)  
        #print("st. deviation: ", np.std(series_data[:,:,:], axis = (0,2)))
        print(np.shape(obs_data))
        fig,ax=plt.subplots()
        for i in range(nseries):
            fig,ax=plt.subplots()
            ntimes=min(len(t),obs_data.shape[1])
            ax.plot(np.array(t[0:ntimes])/3600 ,obs_data[i,0:ntimes],'o', markeredgecolor='black',markerfacecolor='None',label = "Observed data",markersize=2)
            ax.fill_between(np.array(t)/3600, mean_arr[i,:] - std_arr[i,:], mean_arr[i,:] + std_arr[i,:], alpha=0.2,facecolor='blue')
            ax.plot(np.array(t)/3600,mean_arr[i,:] + std_arr[i,:],'b--',linewidth=0.75)
            ax.plot(np.array(t)/3600,mean_arr[i,:] - std_arr[i,:],'b--',linewidth=0.75)
            ax.plot(np.array(t)/3600,mean_arr[i,:],'b-', label = "Ensemble data",linewidth=1.25)
            ax.set_title(loc_names[i])
            ax.set_xlabel('time [hours]')
            if i<5:
                ax.set_ylabel('height [m]')
            else:
                ax.set_ylabel('velocity [m/s]')
    return


def plot_basic_bias(t,x):
    name = ['Cadzand','Vlissingen','Terneuzen','Hansweert','Bath']
    for i in range(5):
        fig,ax=plt.subplots()
        ax.set_title("Model error for %s"%name[i])
        ax.plot(t/3600,x[i,:],'b-')
        ax.set_xlabel('time [hours]')
        ax.set_ylabel('h_model - h_data [m]')
        ax.set_ylim([-1.5,1.5])
        #plt.savefig(("errorplot_%s.png"%name[i]))
    return