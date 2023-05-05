import numpy as np
import math
import random
import matplotlib.pyplot as plt

class LIFNeuron:
    """
    A class used to encapsulate the dynamics of an LIF neuron,
    unit is volts
    """
    def __init__(self,
        t_ahp=8e-3 + 1e-3 * np.random.rand(),  #8e-3
        tau_RC=10e-3 + 20e-3 * np.random.rand(),
        V_rest=-70e-3,
        V_th=-55e-3,
        E_EPSP=0,
        E_IPSP=-80e-3,
        gEPSP=10,  
        gIPSP=20): 
        """
        t_ahp: the afterhyperpolarization period (in sec)
        tau_RC: the RC time constant of the membrane
        V_rest: the resting/reset voltage of the neuron
        V_th: the threshold for spiking
        E_EPSP: excitatory synapse reversal potential 
        E_IPSP: inhibitory synapse reversal potential 
        gEPSP: excitatory synapse  conductance
        gIPSP: inhibitory synapse conductance
        """
        self.t_ahp = t_ahp
        self.tau_RC = tau_RC
        self.V_rest = V_rest
        self.V_th = V_th
        self.hyper = V_rest-15e-3 #new
        self.E_EPSP = E_EPSP 
        self.E_IPSP = E_IPSP 
        self.gEPSP = gEPSP
        self.gIPSP = gIPSP
        self.R = 1 # by convention
        # choose a random initial voltage between V_rest and V_th
        self.V = V_rest + np.random.rand() * (V_th - V_rest)
        self.ahp_time = 0 # time remaining in the aferhyperpolarization

    def update(self, J_in, dt, J_E, J_I):
        """
        Update the membrane voltage given the supplied input current and neuron input

        J_in: the input current at the current timestep
        J_E: whether there's excitatory input from neurons or not at the current timestep 
        J_I: whether there's inhibitory input from neurons or not at the current timestep 
        dt: the timestep in sec

        Returns a boolean variable indicating if the neuron fired on the given timestep
        """
        # check if we are in the afterhyperpolarization period
        # if self.ahp_time == self.t_ahp:
        #     self.V = self.hyper
        #     self.ahp_time -= dt
        # elif self.ahp_time > 0:
        #     self.V = self.V+dt*15e-3/self.t_ahp
        #     self.ahp_time -= dt 
        if self.ahp_time > 0: # without hyperpolarization
            self.V = self.V_rest
            self.ahp_time -= dt 
        else:
            # J_in(t) = J_R + J_C + J_PC 
            # J_in(t) = [V(t) - V_rest] / R + C * (dV(t)/dt) + [V(t)-E_EPSP] * gEPSP + [V(t)-E_IPSP] * gIPSP  
            delta_V = (1 / self.tau_RC) * (J_in * self.R + (self.V_rest - self.V) + J_E/3*self.gEPSP*self.R*(self.E_EPSP - self.V) + J_I*self.gIPSP*self.R*(self.E_IPSP-self.V)) * dt 
            self.V = self.V + delta_V
        # check if we fired
        if self.V > self.V_th:
            self.ahp_time = self.t_ahp
            return True
        return False

def run_simulation(neuron, J_in, dt, J_E, sep_J_E, J_I, sep_J_I):
    """
    Using a single LIF neuron and apply the passed current

    neuron: LIFNeuron
    J_in: An input current array of length "duration"
    dt: the timestep in seconds

    Plots the membrane voltage as a function of time with
    vertical lines signifying the time of each spike.

    Returns figure
    """
    v_of_t = np.zeros(len(J_in))
    spiketrain = np.zeros(len(J_in), dtype=bool)
    # assuming dt_E>dt
    for i in range(0, len(J_in)):
        spiketrain[i] = neuron.update(J_in[i], dt, J_E[i], J_I[i])
        v_of_t[i] = neuron.V

    fig, axes = plt.subplots(3, 1, sharex=True) 
    time_axis = np.arange(0, len(J_in) * dt, dt)

    # rasters of excitatory input for all excitatory neurons
    ax1 = axes[0]
    ax1.eventplot(sep_J_E, linelengths=2, linewidths=2) 
    ax1.set_xlabel('Excitatory Spike')
    ax1.set_ylabel('Neuron')

    ax2 = axes[1] # rasters of inhibitory input for all inhibitory neurons
    ax2.eventplot(sep_J_I, linelengths=2, linewidths=2) 
    ax2.set_xlabel('Inhibitory Spike')
    ax2.set_ylabel('Neuron')

    ax3 = axes[2] # membrane voltage
    spike_indices = np.where(spiketrain)[0]
    target_spike = [x for x in spike_indices if x>=10000 and x<=20001] 
    for spike_index in target_spike: #spike_indices of target neuron
        ax3.axvline(spike_index * dt * 10000, color="r")
    print("Number of spikes: ", len(spike_indices))
    spike_rate = len(spike_indices) / (len(J_in) * dt)
    print("Mean firing rate: {:f} spk/s".format(spike_rate))
    # ax.plot(time_axis, v_of_t * 1000, label="Voltage")
    ax3.plot(time_axis[10000:20001]*10000, v_of_t[10000:20001] * 1000, label="Voltage") # voltage of target neuron 
    ax3.set_xticks([10000,12500, 15000, 17500, 20000])
    ax3.set_xlabel("Time (0.1 ms)")
    ax3.set_ylabel("Membrane voltage (mV)")
    fig.suptitle("Best Simulation’s Membrane Voltage\nBefore Adding Regular Inhibitory Spikes")
    # plt.show()
    # plt.savefig("voltage_before.svg", format="svg")
    
    fig = plt.figure() # autocorrelogram
    mega_lst = spiketrain_to_ACG(spike_indices)
    plt.hist(mega_lst, bins=50)
    plt.xlabel('Time from each spike (0.1ms)')
    plt.xticks([-400,-200,0,200,400])
    plt.ylabel('Number of spikes')
    fig.suptitle("Autocorrelagram of Simulated Data\n(100% Synchrony)")
    plt.savefig("100sync_ACG.svg", format="svg")
    #plt.title("Autocorrelagram")

    fig = plt.figure() # interspike distribution
    interspike_dist = interspike(spike_indices)
    plt.hist(interspike_dist, bins=50) 
    plt.xlabel('Interspike time (0.1ms)') 
    plt.xlim(0,900)
    plt.xticks([0,100,200,300,400,500,600,700,800,900])
    plt.ylabel('Number of spikes')
    fig.suptitle("Interspike Interval Distribution of Simulated Data\n(100% Synchrony)")
    plt.savefig("100sync_ISI.svg", format="svg")
    #plt.title("Interspike Interval Distribution")

    fig = plt.figure() # peristimulus time historgram
    PSTH = psth(spike_indices)
    plt.hist(PSTH, bins=500) #100
    plt.xlabel('Time from stimulation (0.1ms)')
    plt.xlim(-250, 250) #maybe delete
    plt.ylabel('Number of spikes')
    # plt.title("Peristimulus Time Histogram")
    fig.suptitle("Best Simulation’s Peristimulus Time Histogram\nBefore Adding Regular Inhibitory Spikes")
    # plt.savefig("PSTH_before.svg", format="svg")
    return spike_rate

def simulate_one_EPSP_one_IPSP(neuron, duration=100, dt=0.1e-3, n_E=135, n_I=40, synchrony_pct=0): #added dt_E and dt_I, unit is 0.1ms #duration unit is s 
    J_in = np.zeros(math.ceil(duration / dt)) 
    # Excitatory Input one spike every dt_E, unit is dt
    J_E = np.zeros(math.ceil(duration / dt)) 
    sep_J_E = []
    sep_J_I = []
    
    for num in range(n_E):
        neuron_J_E = np.zeros(math.ceil(duration / dt)) 
        dt_E = np.linspace(0.03,0.075, len(range(n_E))) # 30ms-75ms as mean 
        for i in range(len(J_E)): 
            if i%(dt_E[num]//dt)==0: 
                rand = random.randint(100, 400) #10-40ms random
                try:
                    J_E[i+rand-1] += 1
                    neuron_J_E[i+rand-1] += 1
                except IndexError:
                    continue
        E_indices = np.where(neuron_J_E)[0]
        E_spike = [x for x in E_indices if x>=10000 and x<=20000] # data of 1-2 second for visualization 
        sep_J_E.append(E_spike)
        
    J_I = np.zeros(math.ceil(duration / dt)) 
    neuron_J_I = np.zeros(math.ceil(duration / dt))
    dt_I = 0.05
    for i in range(len(J_I)):
        if i%(dt_I//dt)==0:
            rand = random.randint(100, 400)
            try:
                J_I[i+rand-1] += 1
                neuron_J_I[i+rand-1] += 1
            except IndexError:
                continue
    I_indices = np.where(neuron_J_I)[0]
    I_spike = [x for x in I_indices if x>=10000 and x<=20000] # data of 1-2 second for visualization #20000
    sep_J_I.append(I_spike)

    sync = round(n_I*synchrony_pct/100)

    # synchronous part
    for num in range(1, sync): 
        sep_J_I.append(I_spike)

    # spontaneous part
    for num in range(sync, n_I):
        neuron_J_I = np.zeros(math.ceil(duration / dt)) 
        dt_I = np.linspace(0.03,0.07,len(range(sync, n_I))) # 30ms-70ms as mean 
        for i in range(len(J_I)): 
            if i%(dt_I[num-sync]//dt)==0: 
                rand = random.randint(100, 400) #10-40ms random
                try:
                    J_I[i+rand-1] += 1
                    neuron_J_I[i+rand-1] += 1
                except IndexError:
                    continue
            # if i%2000 == 0:
            #     if neuron_J_I[i-1]==0:
            #         J_I[i-1] += 1
            #         neuron_J_I[i-1] += 1  
        I_indices = np.where(neuron_J_I)[0]
        I_spike = [x for x in I_indices if x>=10000 and x<=20000] # data of 1-2 second for visualization 
        sep_J_I.append(I_spike)
    
    return run_simulation(neuron, J_in, dt, J_E, sep_J_E, J_I, sep_J_I) #J_in should be incoming current

def spiketrain_to_ACG(spike_lst):
    mega_lst = []
    n = len(spike_lst)
    for spike in spike_lst:
        lower = spike-500
        upper = spike+500
        for i in range(len(spike_lst)):
            if spike_lst[i]>=lower and spike_lst[i]<upper:
                mega_lst.append(spike_lst[i]-spike)
            if spike_lst[i]>upper:
                break
    #eliminate by subtracting from itself 
    mega_lst = [x for x in mega_lst if x!=0]
    return mega_lst

def interspike(spike_lst):
    intspk = []
    for i in range(len(spike_lst)-1):
        intspk.append(spike_lst[i+1]-spike_lst[i])
    return intspk

def psth(spike_lst):
    peristimulus = []
    for spike in spike_lst:
        cmp_to_stimulus = spike%2000
        if cmp_to_stimulus==0:
            continue
        if cmp_to_stimulus<1000:
            peristimulus.append(cmp_to_stimulus)
        else:
            peristimulus.append(cmp_to_stimulus-2000)
    return peristimulus

if __name__ == '__main__':
    # Main entry point
    plt.ioff()
    neuron = LIFNeuron()
    simulate_one_EPSP_one_IPSP(neuron, synchrony_pct=100) #synchrony_pct
    # figure = plt.figure()
    # spike_rates = []
    # for i in range(10):
    #     spike_rates.append(simulate_one_EPSP_one_IPSP(LIFNeuron()))
    # plt.plot(spike_rates)
    # plt.ylabel('target neuron spike rate')
    # plt.xticks([0,20,40,60,80,100])
    # plt.show()
