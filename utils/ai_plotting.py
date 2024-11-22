import matplotlib
import matplotlib.pyplot as plt
import numpy as np

MATLAB_COLORS = [];
MATLAB_COLORS.append((0, 0.4470, 0.7410));
MATLAB_COLORS.append((0.8500, 0.3250, 0.0980));
MATLAB_COLORS.append((0.9290, 0.6940, 0.1250));
MATLAB_COLORS.append((0.4940, 0.1840, 0.5560));
MATLAB_COLORS.append((0.4660, 0.6740, 0.1880));
MATLAB_COLORS.append((0.3010, 0.7450, 0.9330));
MATLAB_COLORS.append((0.6350, 0.0780, 0.1840));
MATLAB_COLORS.append("seagreen");
MATLAB_COLORS.append("olive");
MATLAB_COLORS.append("hotpink");
MATLAB_COLORS.append("tomato");
MATLAB_COLORS.append("plum");
MATLAB_COLORS.append("black");
MATLAB_COLORS.append("slategrey");
MATLAB_COLORS.append("gold");


def get_matlab_colors():
    MATLAB_COLORS = [];
    MATLAB_COLORS.append((0, 0.4470, 0.7410));
    MATLAB_COLORS.append((0.8500, 0.3250, 0.0980));
    MATLAB_COLORS.append((0.9290, 0.6940, 0.1250));
    MATLAB_COLORS.append((0.4940, 0.1840, 0.5560));
    MATLAB_COLORS.append((0.4660, 0.6740, 0.1880));
    MATLAB_COLORS.append((0.3010, 0.7450, 0.9330));
    MATLAB_COLORS.append((0.6350, 0.0780, 0.1840));
    MATLAB_COLORS.append("seagreen");
    MATLAB_COLORS.append("olive");
    MATLAB_COLORS.append("hotpink");
    MATLAB_COLORS.append("tomato");
    MATLAB_COLORS.append("plum");
    MATLAB_COLORS.append("black");
    MATLAB_COLORS.append("slategrey");
    MATLAB_COLORS.append("gold");
    return MATLAB_COLORS;

def aingura_plot_style(plt_style='seaborn-v0_8-white',full_format=True,
                        figsize=(10,8),line_width=2,font_size=20,
                        grid_color='silver'):
    """Function to standardize the plots for Aingura's team

    Args:
        plt_style (str, optional): matplotlib styles to apply (see available styles in plt.style.available). Defaults to 'default'.
        figsize (tuple, optional): tuple with heigh and width size fo figures in inches. Defaults to (10,8).
        line_width (int, optional): line width of the plots. Defaults to 2.
        font_size (int, optional): base figure's font size. Defaults to 20.
        grid_color (str, optional): grid color. Defaults to 'silver'.
    """
    import matplotlib 
    import matplotlib.pyplot as plt 
    matplotlib.style.use(plt_style)

    if (full_format):
        matplotlib.rcParams.update({'font.size': font_size}) 
        plt.rcParams["figure.figsize"] = figsize
        plt.rc('legend', fontsize=font_size-2)
        plt.rc('axes', titlesize=font_size+4)
        plt.rc('axes', grid=True)
        plt.rc('xtick.minor',visible=True)
        plt.rc('ytick.minor',visible=True)
    
        matplotlib.rc('xtick', labelsize=font_size) 
        matplotlib.rc('ytick', labelsize=font_size)
    
        plt.rc('figure', titlesize=font_size+4)
        plt.rc('lines', lw=line_width)
        plt.rc('grid', c=grid_color, ls='-', lw=0.4)  # solid gray grid lines
        plt.rc("axes.grid",which='both')

    plt.rcParams['image.cmap'] = 'viridis'
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman']+plt.rcParams['font.serif']

def conf_one_axis_fs(axis_i,label_fs=16,tick_fs=14,lg=0.5):
    """
    Function to configure an axis properly.
    inputs:: 
    axis:axis of figure
    lend_fs: fontsize of the legend.
    label_fs: fontsize of the labels.
    tick_fs: fontsize fo the ticks.
    """
    axis_i.minorticks_on();
    axis_i.grid(which='major',color='silver',ls='-',lw= lg);
    axis_i.grid(which='minor',color='silver',ls='--',lw=lg);
    axis_i.xaxis.get_label().set_fontsize(label_fs)
    axis_i.yaxis.get_label().set_fontsize(label_fs)
    axis_i.tick_params(axis='x', labelsize=tick_fs)
    axis_i.tick_params(axis='y', labelsize=tick_fs)

def conf_plot_fs(axis,legd_fs=14,label_fs=16,tick_fs=14):
    """
    Function to configure plot properly.
    inputs:: 
    axis:axis of figure
    lend_fs: fontsize of the legend.
    label_fs: fontsize of the labels.
    tick_fs: fontsize fo the ticks.
    """
    plt.rc('legend', fontsize=legd_fs)
    lg=0.5
    # if (not isinstance(axis_, (np.ndarray, np.generic))):
    #     axis_ = [axis_]
    if len(np.shape(axis))==0:
        conf_one_axis_fs(axis,label_fs=label_fs,tick_fs=tick_fs,lg=lg)    
    else:
        axis_c=axis
        if (len(axis.shape)==2):
            axis_c = np.reshape(axis,(axis.shape[0]*axis.shape[1],))
        for axis_i in axis_c:
            conf_one_axis_fs(axis_i,label_fs=label_fs,tick_fs=tick_fs,lg=lg)
      
def conf_plot_fs_2(axis,legd_fs=14,label_fs=16,tick_fs=14):
    """
    Function to configure plot properly.
    inputs:: 
    axis:axis of figure
    lend_fs: fontsize of the legend.
    label_fs: fontsize of the labels.
    tick_fs: fontsize fo the ticks.
    """
    plt.rc('legend', fontsize=legd_fs)
    lg=0.5
    if len(np.shape(axis))==0:
        axis.minorticks_on();
        axis.grid(which='major',color='silver',ls='-',lw= lg);
        axis.grid(which='minor',color='silver',ls='--',lw=lg);
        axis.xaxis.get_label().set_fontsize(label_fs)
        axis.yaxis.get_label().set_fontsize(label_fs)
        axis.tick_params(axis='x', labelsize=tick_fs)
        axis.tick_params(axis='y', labelsize=tick_fs)
    elif len(np.shape(axis))==2:
        for k in range(np.shape(axis)[0]):
            for q in range(np.shape(axis)[1]):
                axis[k][q].minorticks_on();
                axis[k][q].grid(which='major',color='silver',ls='-',lw= lg);
                axis[k][q].grid(which='minor',color='silver',ls='--',lw=lg);
                axis[k][q].xaxis.get_label().set_fontsize(label_fs)
                axis[k][q].yaxis.get_label().set_fontsize(label_fs)
                axis[k][q].tick_params(axis='x', labelsize=tick_fs)
                axis[k][q].tick_params(axis='y', labelsize=tick_fs)
    else:
        for k in range(len(axis[:])):
            axis[k].minorticks_on();
            axis[k].grid(which='major',color='silver',ls='-',lw= lg);
            axis[k].grid(which='minor',color='silver',ls='--',lw=lg);
            axis[k].xaxis.get_label().set_fontsize(label_fs)
            axis[k].yaxis.get_label().set_fontsize(label_fs)
            axis[k].tick_params(axis='x', labelsize=tick_fs)
            axis[k].tick_params(axis='y', labelsize=tick_fs)

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def enable_grid_1D(axis_i):
    axis_i.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    axis_i.minorticks_on()
    axis_i.grid(True)

def enable_grid(axis):
    if (not isinstance(axis, (np.ndarray, np.generic, list))):
        enable_grid_1D(axis)
    else:
        axis_c=axis
        if (len(axis.shape)==2):
            axis_c = np.reshape(axis,(axis.shape[0]*axis.shape[1],))
        for axis_i in axis_c:
            enable_grid_1D(axis_i)

def undock_plots(use_tk =  True):
    import matplotlib
    from IPython import get_ipython
    if (use_tk):
        matplotlib.use('TkAgg')
        get_ipython().run_line_magic("matplotlib", "tk")
    else:
        # qt
        matplotlib.use('Qt5Agg')
        get_ipython().run_line_magic("matplotlib", "qt5")

def enable_autoreload():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

def axis_as_array(axis):
    if (not isinstance(axis, (np.ndarray, np.generic))):
        axis = np.array([axis])
    return axis


def get_tight_axis(id_figure,n_rows,n_cols=1,hspace=0, wspace=0,sharex='col', sharey='row',height_ratios=[]):
    fig = plt.figure(id_figure)
    plt.clf();
    if (len(height_ratios)==0):
        height_ratios = np.ones(n_rows);
    # height_ratios[0:-1] = 4;
    gs = fig.add_gridspec(n_rows, n_cols, hspace=hspace, wspace=wspace, height_ratios=height_ratios.tolist())
    return fig,gs.subplots(sharex=sharex, sharey=sharey)

def color(n,plot=False):
        """
        """
        def z_rotation(vector,theta):
            """Rotates 3-D vector around z-axis"""
            R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
            return np.dot(R,vector)
        colors = []
        from itertools import product
        bounds = np.array([0.9,0])
        count = 3
        perm = list(product([0,1],[0,1],[0,1]))
        seguir = True
        rot = 0
        while seguir:
            for i in perm:
                colori = bounds[list(i)]
                if rot ==1:
                    colori = z_rotation(colori-np.ones(3)*0.5,np.pi/4)+np.ones(3)*0.5
                colors.append(colori)
                if len(colors) ==n+1:
                    seguir = False
                    break
            if rot ==0:
                rot = 1
            else:
                rot=  0
            bounds = np.array([0.5-1/count,0.5+1/count])
            count = count*2
        colors = np.array(colors)    
        if plot==True:
            fig = plt.figure("colors")
            fig.clf()
            ax = fig.add_subplot(projection='3d')
            for i in range(n):
                ax.scatter(colors[i,0],colors[i,1],colors[i,2],color= colors[i])
            ax.grid("on")
        return colors[1:]

def plot_multiple_signals(signals2plot,id_figure=1
                             ,hspace=0, wspace=0,sharex='col', sharey='row'
                             ,fontsize=11,marker='-',linewidth=1,labelsize = 10,colors=False):
    
    n_rows = len(signals2plot)
    # fig, axis = plt.subplots(n_rows, 1,num=id_figure,sharex='all',sharey='all');
    fig,axis = get_tight_axis(id_figure,n_rows=len(signals2plot),n_cols=1
                            ,hspace=hspace,wspace=wspace
                            ,sharex=sharex,sharey=sharey)
    axis =  axis_as_array(axis)
    for i in range(n_rows):
        if(colors):
            n_signals = signals2plot[0][1].shape[1]
            colores = color(n_signals)
            for j in range(n_signals):
                axis[i].plot(signals2plot[i][0],signals2plot[i][1][:,j],marker,linewidth=linewidth,color=colores[j])
        else:
            axis[i].plot(signals2plot[i][0],signals2plot[i][1],marker,linewidth=linewidth)
        axis[i].set_ylabel(signals2plot[i][2],fontweight='bold',size=fontsize)
        axis[i].tick_params(axis='x', labelsize=labelsize)
        axis[i].tick_params(axis='y', labelsize=labelsize)
    enable_grid(axis)
    # plt.tight_layout()
    return fig, axis

def set_fontsize_xy_ticks(fontsize):
    matplotlib.rc('xtick', labelsize=fontsize) 
    matplotlib.rc('ytick', labelsize=fontsize)


def set_axis_bold():
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

def get_plot_colors(n):
    from itertools import cycle, islice
    return np.array(list(islice(cycle(
                # ['#377eb8', '#ff7f00', '#4daf4a',
                #     '#f781bf', '#a65628', '#984ea3',
                #     '#999999', '#e41a1c', '#dede00']
                ['#1f77b4', '#ff7f0e', '#2ca02c',
                    '#d62728', '#9467bd', '#8c564b',
                    '#e377c2', '#7f7f7f','#bcbd22', '#17becf']
                ),
            int(n))))

def set_colors_boxplot(boxplot_d,edge_color,fill_colors,median_color,lw=2):
    for element in ['boxes','whiskers', 'fliers', 'caps']:
        for bp_element_i in boxplot_d[element]:
            plt.setp(bp_element_i, color=edge_color)
    
    for element in ['means','medians']:
        for bp_element_i in boxplot_d[element]:
            plt.setp(bp_element_i, color=median_color,lw=lw)

    for i_c,patch in enumerate(boxplot_d['boxes']):
        if (callable(fill_colors)):
            patch.set_facecolor(fill_colors(i_c)) 
        else:
            patch.set_facecolor(fill_colors[i_c]) 

def plot_boxplot(data_dict, id_figure=1, edge_color ='black',median_color='C1',colors=[],
                    axis=None,
                    add_median_plot=True):
    data_to_list = []
    xticks_lables = np.sort(list(data_dict.keys()))
    medians = []
    for key in xticks_lables:
        data_to_list.append(data_dict[key])
        if (add_median_plot):
            medians.append(np.median(data_dict[key]))
            if (np.isnan(medians[-1])):
                medians[-1]=0
                if (len(medians)>1):
                    medians[-1]=medians[-2]
    
    if (axis is None):
        plt.figure(id_figure)
        plt.clf()
        fig, axis = plt.subplots(1, 1,num=id_figure,sharex='col');# , sharey='col' , sharex='col'
    boxplot_info = axis.boxplot(data_to_list,labels=xticks_lables,patch_artist=True)
    data_to_list_x = np.arange(len(medians))+1
    # axis.set_xticklabels(xticks_lables)
    if (colors==[]):
        colors = get_plot_colors(len(xticks_lables))
    set_colors_boxplot(boxplot_info,edge_color,colors,median_color)
    if (add_median_plot):
        axis.plot(data_to_list_x,medians,'k')
    
    enable_grid(axis)
    # plt.tight_layout()
    return axis

def plot_imshow(x, y, z, axis,db_scale,sw_bar=True,orientation="vertical",cmap=plt.cm.Spectral):
    import matplotlib.dates as mdates
    fzz = 12
    z1 = ((z-z.min())/(z.max()-z.min()))
    min_v = np.min(z)
    max_v = np.max(z)
    # print(min_v,max_v)
    # Plot
    im = axis.imshow(z, 
                        extent=[x[0],x[-1],y[0],y[-1]], 
                        cmap=cmap,
                        vmin=min_v, vmax=max_v,
                         origin='lower', aspect='auto')
    # im = axis.imshow(x_stft, cmap=plt.cm.GnBu,vmin=min_v, vmax=max_v, origin='lower', aspect='auto')
    # Labels 
    axis.set_xlim(x[0],x[-1])
    axis.set_ylim(y[0],y[-1])
    # Color bar
    if (sw_bar):
        cbar = plt.colorbar(im,ax=axis, fraction=0.046, pad=0.04,orientation=orientation)
        cbar.ax.get_yaxis().labelpad = 15
        if (db_scale):
            cbar.ax.set_ylabel('Amplitude [dB]', rotation=270,fontsize=fzz)
        else:
            cbar.ax.set_ylabel('Amplitude [mag]', rotation=270,fontsize=fzz)
        cbar.ax.tick_params(labelsize=10)
        return im,cbar
    return im

def create_figure(id_figure=1,nrows=1,ncols=1,sharey='none',sharex='none'):
    fig=plt.figure(id_figure)
    plt.clf();
    fig, axis = plt.subplots(nrows, ncols, sharex=sharex,sharey=sharey,num=id_figure);
    axis =  axis_as_array(axis)
    return fig, axis
