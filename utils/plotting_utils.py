import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_all_graphs(fig, coordinates_passed, spl_sampled, intensity_loss_2d, potential_sampling_locations, 
                    survey_grid, observed_pred_global, lower_local, upper_local, 
                    var_iter_local, var_iter_global, rmse_local_true, rmse_global_true, 
                    lengthscale, noise, covar_trace, covar_totelements, covar_nonzeroelements, 
                    AIC, BIC, f2_H_local, f2_H_global, x_max):
    ax1 = fig.add_subplot(4, 3, 1, projection='3d')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('sound intensity')
    ax1.view_init(20, 20)
    ax1.set_title('asv on surface '+str(coordinates_passed[-1]))
    x_distances = survey_grid[:, 0]
    y_distances = survey_grid[:, 1]
    X, Y = np.meshgrid(x_distances, y_distances)
    ax1.plot_surface(X, Y, intensity_loss_2d, cmap='viridis', alpha=0.7)

    
    asv_path = ax1.plot3D([coord[0] for coord in coordinates_passed], [coord[1] for coord in coordinates_passed], spl_sampled, color='black')
    asv = ax1.scatter3D(coordinates_passed[-1][0], coordinates_passed[-1][1], 
                        spl_sampled[-1], s=100, color='black', marker='*', zorder=1)
   
    for i_test in range(len(potential_sampling_locations)):
        ax1.plot(potential_sampling_locations[i_test][0].numpy()*np.array([1, 1]), potential_sampling_locations[i_test][1].numpy()*np.array([1, 1]), 
                 np.array([lower_local[i_test].numpy(), upper_local[i_test].numpy()]), 'gray')

 
    #AX2 is diff
    ax2 = fig.add_subplot(4, 3, 2)
    ax2.set_xlim(-60,60)
    ax2.set_ylim(-60,60)
    ax2.set_xlabel('x coordinate')
    ax2.set_ylabel('y coordinate')
    ax2.set_title('asv on surface ' + str(coordinates_passed[-1]))
    contour = ax2.contourf(X, Y, intensity_loss_2d, cmap='viridis')
    cbar = fig.colorbar(contour)
    cbar.set_label('Sound Pressure Level')
    asv_path = ax2.plot([coord[0] for coord in coordinates_passed], [coord[1] for coord in coordinates_passed], color='pink')
    asv = ax2.scatter(coordinates_passed[-1][0], coordinates_passed[-1][1], color='pink', marker='*', label='ASV', s=100)
    ax2.scatter(x_max[0], x_max[1], color='red', label='global maximum uncertainity', s=50)



    ax3 = fig.add_subplot(4, 3, 3)
    ax3.plot(range(0, len(var_iter_local)),
             var_iter_local, color='blue', marker='.')
    ax3.plot(range(0, len(var_iter_global)),
             var_iter_global, color='black', marker='*')
    ax3.set_xlabel('number of samples')
    ax3.set_title('variance of samples')
    ax3.legend(['local', 'global'], loc='upper right')


    
    ax4 = fig.add_subplot(4, 3, 4)
    local_rms = ax4.plot(range(0, len(rmse_local_true)),
                         rmse_local_true, color='blue', marker='.', label='local')
    global_rms = ax4.plot(range(0, len(rmse_global_true)),
                          rmse_global_true, color='black', marker='*', label='global')
    ax4.set_xlabel('number of samples')
    ax4.legend(['local', 'global'], loc='upper right')
    ax4.set_title('rmse of learned model')
    
    ax5 = fig.add_subplot(4, 3, 5)
    ax6 = fig.add_subplot(4, 3, 6)

    lengthscale_arr = np.array(lengthscale)
    length0_plot = ax5.plot(range(0, len(lengthscale)),
                         lengthscale_arr[:,0], color='mediumpurple', marker='.', label='lengthscale')
    length1_plot = ax5.plot(range(0, len(lengthscale)),
                         lengthscale_arr[:,1], color='rebeccapurple', marker='.', label='lengthscale')
    ax5.set_ylim([0, np.max(lengthscale_arr)*1.1])
    noise_plot = ax6.plot(range(0, len(noise)),
                         noise, color='teal', marker='.', label='noise')
    ax5.set_xlabel('number of samples')
    ax5.set_ylabel('lengthscale',color='mediumpurple')
    ax6.set_ylabel('noise',color='teal')
    ax6.set_xlabel('number of samples')

    ax7 = fig.add_subplot(4, 3, 7)
    tot_elements = ax7.plot(range(0, len(covar_totelements)),
                         covar_totelements, color='mediumvioletred', marker='.', label='total_elements')
    nonzero_elements = ax7.plot(range(0, len(covar_nonzeroelements)),
                         covar_nonzeroelements, color='blue', marker='.', label='nonzero_elements')
    ax7.set_ylabel('elements')
    ax7.set_xlabel('number of samples')
    ax7.legend(['total', 'nonzero'], loc='lower right')
    
    ax8 = fig.add_subplot(4, 3, 8)
    AIC_plot = ax8.plot(range(0, len(AIC)), AIC, color='blue', marker='.', label='AIC')
    BIC_plot = ax8.plot(range(0, len(BIC)), BIC, color='black', marker='*', label='BIC')
    ax8.set_xlabel('number of samples')
    ax8.legend(['AIC', 'BIC'], loc='lower right')
    ax8.set_ylabel('information criteria of learned model')
    
    ax9 = fig.add_subplot(4, 3, 9)
    RKHS_local_plot = ax9.plot(range(0, len(f2_H_local)), f2_H_local, color='blue', marker='.', label='f^2_H')
    RKHS_global_plot = ax9.plot(range(0, len(f2_H_global)), f2_H_global, color='black', marker='.', label='f^2_H')
    ax9.set_ylabel('RKHS_norm')
    ax9.set_xlabel('number of samples')
    ax8.legend(['local', 'global'], loc='lower right')

    ax10 = fig.add_subplot(4, 3, 10)
    trace = ax10.plot(range(0, len(covar_trace)),
                         covar_trace, color='pink', marker='.', label='trace')
    ax10.set_xlabel('number of samples')
    ax10.set_ylabel('trace',color='pink')

    return ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10