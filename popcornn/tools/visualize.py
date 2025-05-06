import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

plot_params = {
    'font_size' : 20,
    'fig_size' : (10, 10)
}
gridspec = {
        'height_ratios' : [3, 0.2, 1],
        'hspace' : 0.,
        'left' : 0.15,
        'right' : 0.96,
        'top' : 0.95,
        'bottom' : 0.07,
    }

def from_numpy(np_arrays):
    for idx in range(len(np_arrays)):
        np_arrays[idx] = torch.tensor(np_arrays[idx])
    return np_arrays

def to_numpy(arrays):
    for idx in range(len(arrays)):
        arrays[idx] = arrays[idx].detach().to('cpu').numpy()
    return arrays

def eval_contour_vals(
    potential,
    x_min,
    x_max,
    y_min,
    y_max,
    num_points=1024,
    add_dof=False
):
    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = np.linspace(y_min, y_max, num_points)
    l,r = np.meshgrid(x_vals, y_vals)
    size = len(x_vals)
    args = np.reshape(np.stack([l,r],axis=2), (-1, 2))
    if add_dof:
        args = np.concatenate([args, np.zeros((args.shape[0], 1))], axis=1)
        #    jnp.array([x_vals, y_vals, jnp.zeros(len(x_vals))]).transpose()
        #)
        #trans_xy_vals = np.array(trans_xy_vals)
        #x_vals = jnp.expand_dims(trans_xy_vals[:,0], -1)
        #y_vals = jnp.expand_dims(trans_xy_vals[:,1], -1)
    args = from_numpy([args])[0]
    z_vals = potential(args)
    trans_xy_vals = potential.point_transform(args)
    z_vals, trans_xy_vals = to_numpy([z_vals, trans_xy_vals])
    return (np.reshape(trans_xy_vals[:,0], (size, size)),
        np.reshape(trans_xy_vals[:,1], (size, size)),
        np.reshape(z_vals, (size, size))
    )#jnp.apply_along_axis(function, 2, args)


def contour_2d(
        ax,
        potential,
        plot_min_max,
        levels,
        contour_vals=None,
        return_contour_vals=False,
        add_dof=False,
    ):
    if contour_vals is not None:
        ax.contour(*contour_vals[:3], levels=contour_vals[3])
        return ax, contour_vals

    if plot_min_max is None:
        raise ValueError("Must specify max and min for x and y when plotting contours.")
    if levels is None:
        raise ValueError("Must specify contour levels when plotting pes_fxn countours.")
    x_min, x_max, y_min, y_max = plot_min_max
 
    x_vals, y_vals, z_vals = eval_contour_vals(
        potential, x_min, x_max, y_min, y_max, add_dof=add_dof
    )
    ax.contour(x_vals, y_vals, z_vals, levels=levels, cmap='coolwarm')
    return ax, (x_vals, y_vals, z_vals, levels) 


def _plot_added_rot_dof(
        path,
        name,
        radius,
        pes_fxn=None,
        plot_min_max=None,
        levels=None,
        contour_vals=None,
    ):

    plot_min_max = (
        plot_min_max[0] + radius,        
        plot_min_max[1] + radius,        
        plot_min_max[2],        
        plot_min_max[3]
    )
    fig_size = (plot_params['fig_size'][0], plot_params['fig_size'][0]*1.2)
    _gridspec = dict(gridspec)
    _gridspec['height_ratios'] = _gridspec['height_ratios'][:] + [1]
    path_th = np.array(np.arctan2(path[:,0], path[:,-1]))
    time = np.linspace(0, 1, len(path_th))

    fig_xz, ax_xz = plt.subplots(4, 1, figsize=fig_size, gridspec_kw=_gridspec)
    _, contour_vals = _plot_path(
        ax_xz, path, pes_fxn, plot_min_max, levels,
        contour_vals=contour_vals, return_contour_vals=True, add_dof=True
    )
    ax_xz[3].plot(time, path_th)
    ax_xz[3].set_xlim(0,1)
    ax_xz[3].set_ylabel('Azimuthal [rad]', fontsize=plot_params['font_size'])
    
    
    fig_x, ax_x = plt.subplots(4, 1, figsize=fig_size, gridspec_kw=_gridspec)
    _, contour_vals = _plot_path(
        ax_x,
        np.concatenate([path[:,:-1], np.zeros((path.shape[0], 1))], axis=-1),
        pes_fxn, plot_min_max, levels,
        contour_vals=contour_vals, return_contour_vals=True, add_dof=True
    )
    ax_x[3].plot(time, path_th)
    ax_x[3].set_xlim(0,1)
    ax_x[3].set_ylabel('Azimuthal [rad]', fontsize=plot_params['font_size'])

    _gridspec = dict(gridspec)
    _gridspec['height_ratios'] = _gridspec['height_ratios'][:]
    _gridspec['height_ratios'][1] = _gridspec['height_ratios'][1]*1.5
    fig_x_z, ax_x_z = plt.subplots(3, 1, figsize=fig_size, gridspec_kw=_gridspec)
    ax_x_z[0].plot(path[:,0]-radius, path[:,-1]-radius, '-k')
    ax_x_z[0].set_xlim(np.amin(path[:,0])-radius, np.amax(path[:,0])-radius)
    ax_x_z[0].set_xlabel(f"x - {radius}", fontsize=plot_params['font_size'])
    #ax_x_z[0].set_ylim(-1*radius, 0)
    ax_x_z[0].set_ylabel(f"z - {radius}", fontsize=plot_params['font_size'])
    ax_x_z[1].set_visible(False)
    ax_x_z[2].plot(time, path_th)
    ax_x_z[2].set_xlim(0,1)
    ax_x_z[2].set_ylabel('Azimuthal [rad]', fontsize=plot_params['font_size'])
    ax_x_z[2].set_xlabel('Time [arb]', fontsize=plot_params['font_size'])


    return [fig_xz, fig_x, fig_x_z], [ax_xz, ax_x, ax_x_z], ['', 'x', 'xz_plane'], [True, True, False], contour_vals

 
def _plot_path(
        ax,
        path,
        pes_fxn=None,
        plot_min_max=None,
        levels=None,
        contour_vals=None,
        return_contour_vals=False,
        add_dof=False
    ):

    if pes_fxn is not None:
        _, contour_vals = contour_2d(
            ax[0],
            pes_fxn,
            plot_min_max,
            levels,
            contour_vals=contour_vals,
            add_dof=add_dof
        )
    else:
        contour_vals = None
    
    path = from_numpy([path])[0]
    path = pes_fxn.point_transform(path)
    path = to_numpy([path])[0]
    ax[0].plot(path[:,0], path[:,1], color='k', linestyle='-', marker='.')
    velocity = np.sqrt(np.sum((path[:-1] - path[1:])**2, axis=-1))
    ax[2].plot(np.linspace(0, 1, len(path)-1), velocity, color='k', linestyle='-')

    return ax, contour_vals


def plot_path(
        file_name,
        time,
        positions,
        energy,
        velocity,
        force,
        loss,
    ):
    fig, axs = plt.subplots(6, 1, figsize=(4, 6), sharex=True)
    axs[0].plot(time)
    axs[0].set_ylabel('Time')
    axs[1].plot(np.linalg.norm(positions, axis=1))
    axs[1].set_ylabel('positions')
    axs[2].plot(energy)
    axs[2].set_ylabel('Energy')
    axs[3].plot(np.linalg.norm(velocity, axis=1))
    axs[3].set_ylabel('Velocity')
    axs[4].plot(np.linalg.norm(force, axis=1))
    axs[4].set_ylabel('Force')
    axs[5].plot(loss)
    axs[5].set_ylabel('Loss')
    axs[5].set_xlabel('Index')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def animate_optimization_2d(
        paths,
        title,
        contour_file,
        pes_fxn=None,
        plot_min_max=None,
        levels=None,
        contour_vals=None,
        add_translation_dof=False,
        add_azimuthal_dof=False,
        plot_dir='./plots/'
    ):

    fig, ax = plt.subplots()
    if pes_fxn is not None:
        ax = contour_2d(ax, pes_fxn, plot_min_max, levels)
    #ax = contour_2d(function, x_min, x_max, y_min, y_max, levels)
    ax[0].set_title(title)
    fig.savefig(contour_file + '.png')
    plot_artist = ax[0].plot([], [], color='red', linestyle='-')[0]


    def animation_function(path):
        if add_translation_dof and False:
            plot_artist.set_data(path[:,0] - path[:,-1], path[:,1])
        else:
            plot_artist.set_data(path[:,0], path[:,1])
        return plot_artist


    print("Plotting animation", len(paths), paths[0].shape)
    ani = animation.FuncAnimation(fig, animation_function, frames=paths)
    os.makedirs(plot_dir, exist_ok=True)
    ani.save(os.path.join(plot_dir, contour_file + ".gif"))

