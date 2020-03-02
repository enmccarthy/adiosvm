#!/usr/bin/env python3
#from mpi4py import MPI
import numpy as np
import adios2
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import operator
from operator import add
from matplotlib.font_manager import FontProperties
DEFAULT={
    "graph_layout" : "individual", # or grouped
    "memory": { "graph_type" : "line" },
    "cpu": { "graph_type" : "line"},
    "io": {"graph_type" : "line"} }
# Global variables
cpu_component_count = 9
cpu_components = ['Guest','I/O Wait', 'IRQ', 'Idle', 'Nice', 'Steal', 'System', 'User', 'soft IRQ']
previous_mean = {}
previous_count = {}
current_period = {}
period_values = {}
#cpu_df = pd.DataFrame([np.zeros(len(cpu_components)), columns=cpu_components)
#mem_components = ['Memory Footprint (VmRSS) (KB)','Peak Memory Usage Resident Set Size (VmHWM) (KB)','meminfo:MemAvailable (MB)','meminfo:MemFree (MB)','meminfo:MemTotal (MB)']
mem_components = ['Memory Footprint (VmRSS) (KB)','Peak Memory Usage Resident Set Size (VmHWM) (KB)','program size (kB)','resident set size (kB)']
mem_components_short = ['VmRSS','VmHWM','program size','RSS']
io_components = ['io:cancelled_write_bytes', 'io:rchar', 'io:read_bytes', 'io:syscr', 'io:syscw', 'io:wchar', 'io:write_bytes']
io_components_short = ['cancelled_write_bytes', 'rchar', 'read_bytes', 'syscr', 'syscw', 'wchar', 'write_bytes']

RANGE = 10    
def SetupArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instream", "-i", help="Name of the input stream", required=True)
    parser.add_argument("--outfile", "-o", help="Name of the output file", default="screen")
    parser.add_argument("--nompi", "-nompi", help="ADIOS was installed without MPI", action="store_true")
    parser.add_argument("--displaysec", "-dsec", help="Float representing gap between plot window refresh", default=0.2)
    args = parser.parse_args()

    args.displaysec = float(args.displaysec)
    args.nx = 1
    args.ny = 1
    args.nz = 1
    
    return args

def dumperiod_valuesars(vars_info):
    # print variables information
    for name, info in vars_info.items():
        print("variable_name: " + name)
        for key, value in info.items():
            print("\t" + key + ": " + value)
        print("\n")

def initialize_globals():
    global cpu_components
    global previous_mean
    global previous_count
    global current_period
    global period_values
    for c in cpu_components:
        previous_mean[c] = 0
        previous_count[c] = 0
        current_period[c] = 0
        period_values[c] = []

    for m in mem_components:
        previous_mean[m] = 0
        previous_count[m] = 0
        current_period[m] = 0
        period_values[m] = []

    for i in io_components:
        previous_mean[i] = 0
        previous_count[i] = 0
        current_period[i] = 0
        period_values[i] = []

def get_utilization(is_cpu, fr_step, vars_info, components, previous_mean, previous_count, current_period, period_values):
    for c in components:
        substr = c
        if is_cpu:
            substr = "cpu: "+c+" %"
        # Get the current mean value
        mean_var = substr + " / Mean"
        shape_str = vars_info[mean_var]["Shape"].split(',')
        shape = list(map(int,shape_str))
        mean_values = fr_step.read(mean_var)
        # Get the number of events
        count_var = substr + " / Num Events"
        shape2_str = vars_info[count_var]["Shape"].split(',')
        shape2 = list(map(int,shape2_str))
        count_values = fr_step.read(count_var)
        # Convert to MB if necessary
        if not is_cpu and "KB" in c.upper():
            mean_values[0] = mean_values[0] / 1000.0
        # Compute the total values seen 
        total_value = mean_values[0]*count_values[0]
        # What's the value from the current frame?
        if previous_count[c] < count_values[0]:
            current_period[c] = (total_value - (previous_mean[c] * previous_count[c])) / (count_values[0] - previous_count[c])
            previous_mean[c] = mean_values[0]
            previous_count[c] = count_values[0]
        #print(c,mean_values[0],count_values[0],total_value, current_period[c])
        period_values[c] = np.append(period_values[c], current_period[c])

def get_top5_cpu(fr_step, vars_info):
    num_ranks = int(vars_info["num_threads"]["Shape"].split(',')[0])
    num_threads = int(vars_info["num_threads"][0])
    cpu_means = {}
    cpu_values = {}
    for name, info in vars_info.items():
        name_split = name.split("/")
        if ".TAU application" in name:
            continue
        if "addr=" in name:
            continue
        if name_split[0] in cpu_components:
            shape_str = vars_info[name]["Shape"].split(',')
            shape = list(map(int, shape_str)) 
            mean_values = fr_step.read(name)
            cup_values[name_split[0]] = []
            cpu_values[name_split[0]].append(mean_values[0])
            index = num_threads
            while index < shape[0]:
                cpu_values[name_split[0]].append(mean_values[index])
                index = index + num_threads
            cpu_means[shortname] = np.sum(cpu_values[name_split[0]]) / num_ranks
    limit = 0
    others = len(timer_means)-5
def get_top5(fr_step, vars_info):
    num_ranks = 16
    num_threads = int(vars_info["num_threads"]["Max"])
    timer_means = {}
    timer_values = {}
    for name, info in vars_info.items():
        if ".TAU application" in name:
            continue
        if "addr=" in name:
            continue
        if "Exclusive TIME" in name:
            shape_str = vars_info[name]["Shape"].split(',')
            shape = list(map(int,shape_str))
            mean_values = fr_step.read(name)
            #print("mean_values")
            #print(mean_values)
            shortname = name.replace(" / Exclusive TIME", "")
            timer_values[shortname] = []
            timer_values[shortname].append(mean_values[0])
            index = num_threads
            while index < shape[0]:
                timer_values[shortname].append(mean_values[index])
                index = index + num_threads
            timer_means[shortname] = np.sum(timer_values[shortname]) / num_ranks   
    limit = 0
    others = len(timer_means) - 5
    timer_values["other"] = [0] * num_ranks
    for key, value in sorted(timer_means.items(), key=lambda kv: kv[1]):
        limit = limit + 1
        if limit <= others:
            timer_values["other"] = list( map(add, timer_values["other"], timer_values[key]) )
            del timer_values[key]
    #print(timer_values)
    return timer_values
#plot stacked bar chart
#potentially replace x with a df
def plot_stacked_bar(ax, x, fontsize, title, components):
    values = np.zeros((len(components),len(period_values[components[0]])))
    for i,c in enumerate(components):
        values[i] = period_values[c]
    ax.stackplot(x[-RANGE:], values[:][-RANGE:], labels=components)
    
    ax.legend(loc='lower left')
    
    fontdict={}
    fontdict['fontsize']=fontsize/2
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("step", fontdict=fontdict)
    ax.set_ylabel("percent", fontdict=fontdict)
    ax.yaxis.tick_right()
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(bbox_to_anchor=(1.10,0.5), loc="center left", borderaxespad=0, prop=fontP)
def plot_bar_chart(ax, x, fontsize, title, components):
    #this is going to need to be the average
    for key in components:
        ax.bar(x,top5[key],label=((key[:30] + '..') if len(key) > 30 else key))
    ax.legend(loc='lower left')

    fontdict={}
    fontdict['fontsize']=fontsize/2
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("rank", fontdict=fontdict)
    ax.set_ylabel("Time (us)", fontdict=fontdict)
    ax.yaxis.tick_right()
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(bbox_to_anchor=(1.10,0.5), loc="center left", borderaxespad=0, prop=fontP)

def plot_line_graph(ax,x,fontsize,title,components):
    per_len = len(period_values[components[0]])
    for m in components:
        if RANGE < per_len:
            ax.plot(x[-RANGE:], period_values[m][-RANGE:], label=m)
        else:
            ax.plot(x, period_values[m][-RANGE:], label=m)
    ax.legend(loc="lower left")
    fontdict={}
    fontdict['fontsize']=fontsize/2
    ax.set_title(title,fontsize=fontsize)
    ax.set_xlabel("step", fontdict=fontdict)
    ax.set_ylabel("MB", fontdict=fontdict)
    ax.yaxis.tick_right()
    #ax.set_yscale("log")
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(bbox_to_anchor=(1.10, 0.5), loc="center_left", borderaxespad=0, prop=fontP)


def plot_timers(ax, x, fontsize, top5):
    #this is a bar chart
    for key in top5:
        ax.bar(x[-RANGE:],top5[key][-RANGE:],label=((key[:30] + '..') if len(key) > 30 else key))
    ax.legend(loc='lower left')

    fontdict={}
    fontdict['fontsize']=fontsize/2
    ax.set_title("Top 5 timers per rank", fontsize=fontsize)
    ax.set_xlabel("rank", fontdict=fontdict)
    ax.set_ylabel("Time (us)", fontdict=fontdict)
    ax.yaxis.tick_right()
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(bbox_to_anchor=(1.10,0.5), loc="center left", borderaxespad=0, prop=fontP)

def plot_utilization_two(args, x, fontsize, step, top5, func_dict, data):
    #this is all of them together
    print("plotting", end='...', flush=True)
    all_together = data[graph_layout]
    if all_together == 'grouped':
        fig = plt.figure(4, figsize=(8,8), constrained_layout=True)
        gs = gridspec.GridSpec(4, 1, figure=fig)
        cpu = fig.add_subplot(gs[0, 0])
        mem = fig.add_subplot(gs[1, 0])
        io = fig.add_subplot(gs[2, 0])
        timers = fig.add_subplot(gs[3, 0])
        plot_stacked_bar(cpu,x,fontsize,"CPU_Utilization", cpu_components)
        #plot_cpu_utilization(cpu, x, fontsize)
        plot_line_graph(mem,x, fontsize, "mem_Utilization", mem_components)
        #plot_mem_utilization(mem, x, fontsize)
        plot_io_utilization(io, x, fontsize)
        plot_timers(timers, np.arange(16), fontsize, top5)
        plt.tick_params(axis='both', which='both', labelsize = fontsize/2)
        print("here")
    else:
        fig = plt.figure(1, figsize=(8,8), constrained_layout=True)
        cpu = fig.add_subplot()
        plot_cpu_utilization(cpu, x, fontsize)
        fig.savefig(args.outfile+".png")
        
    print("writing", end='...', flush=True)
    #come back to this ion call
    #plt.ion()
    if all_together:
        if (args.outfile == "screen"):
            plt.show()
            plt.pause(args.displaysec)
        else:
            print("here")
            imgfile = args.outfile+"_"+"{0:0>5}".format(step)+"dif2.png"
            fig.savefig(imgfile)

    plt.clf()
    print("done.") 
def plot_utilization(aregs, x, fontsize, step, top5):
    func_dict = {"line" : plot_line_graph,
                 "bar"  : plot_bar_chart,
                 "stacked" : plot_stacked_bar}
    components_dict = { "cpu" : cpu_components,
                        "io"  : io_components,
                        "memory" : mem_components}
    
    try:
        with open("monitor_config.JSON") as config_file:
            data = json.load(config_file)
            print("custom settings loaded")
    except IOError:
        data = DEFAULT
        print("default settings loaded")
    print("plotting", end='...', flush=True)
    count = 0
    total = len(data.keys())
    SEP = True
    if SEP:
        fig = plt.figure()
        gs = gridspec.GridSpec(1,1, figure=fig)
    else:
        fig = plt.figure(total, figsize=(8,8), constrained_layout=True)
        gs = gridspec.GridSpec(total, 1, figure=fig)
    for key in data.keys():
        if SEP:
            holder = fig.add_subplot(gs[count,0])
        else:
            holder = fig.add_subplot(gs[count,0])
            count += 1
        graph_type = data[key]["graph_type"]
        func_dict[graph_type](holder, x, fontsize, key + " Utilization",components_dict[key])
        if SEP: 
            imgfile = args.outfile+"_"+"{0:0>5}".format(step)+str(key).capitalize()+".png"
            fig.savefig(imgfile)
            plt.clf()
    if not SEP:
        plt.tick_params(axis="both", which="both", labelsize=fontsize/2)
        imgfile = args.outfile+"_"+"{0:0>5}".format(step)+"dif2.png"
        fig.savefig(imgfile)
        plt.clf()
    print("done.")
        
def process_file(args):
    fontsize=12
    filename = args.instream
    print ("Opening:", filename)
    if not args.nompi:
        fr = adios2.open(filename, "r", MPI.COMM_SELF, "adios2.xml", "TAUProfileOutput")
    else:
        fr = adios2.open(filename, "r", "adios2.xml", "TAUProfileOutput")
    #num_threads = fr[0].available_variables()["num_threads"]["max"]
    initialize_globals()
    cur_step = 0 
    for fr_step in fr:
        # track current step
        cur_step = fr_step.current_step()
        print(filename, "Step = ", cur_step)
        # inspect variables in current step
        vars_info = fr_step.available_variables()
        #dumperiod_valuesars(vars_info)
        get_utilization(True, fr_step, vars_info, cpu_components, previous_mean, previous_count, current_period, period_values)
        get_utilization(False, fr_step, vars_info, mem_components, previous_mean, previous_count, current_period, period_values)
        get_utilization(False, fr_step, vars_info, io_components, previous_mean, previous_count, current_period, period_values)
        top5 = get_top5(fr_step, vars_info)

        x=range(0,cur_step+1)
        plot_utilization(args, x, fontsize, cur_step, top5)

if __name__ == '__main__':
    args = SetupArgs()
    #print(args)
    process_file(args)

