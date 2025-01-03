import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_all_graphs(coordinates_passed, spl_sampled, intensity_loss_2d, 
                    x_test_global, observed_pred_global, lower_local, upper_local, 
                    var_iter_local, var_iter_global, rmse_local_true, rmse_global_true, 
                    lengthscale, noise, covar_trace, covar_totelements, covar_nonzeroelements, 
                    AIC, BIC, f2_H_local, f2_H_global, x_max, image_path, iteration):
    """
    Plot 13 graphs to visualize the GP model performance and exploration.
    """
    fig = plt.figure(figsize=(24, 16))

    # 1. 3D Surface Plot
    ax1 = fig.add_subplot(4, 3, 1, projection='3d')
    ax1.set_title(f"Surface at Iteration {iteration}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Sound Intensity")
    x_distances = np.arange(-50, 51, 1).astype(float)
    y_distances = np.arange(-50, 51, 1).astype(float)
    X, Y = np.meshgrid(x_distances, y_distances)
    ax1.plot_surface(X, Y, intensity_loss_2d, cmap='viridis', alpha=0.7)
    ax1.scatter3D(
        [coord[0] for coord in coordinates_passed],
        [coord[1] for coord in coordinates_passed],
        spl_sampled, color='black'
    )

    # 2. Contour Plot
    ax2 = fig.add_subplot(4, 3, 2)
    ax2.contourf(X, Y, intensity_loss_2d, cmap='viridis')
    ax2.set_title("Contour Plot")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # 3. Variance History
    ax3 = fig.add_subplot(4, 3, 3)
    ax3.plot(var_iter_local, label='Local Variance', color='blue', marker='.')
    ax3.plot(var_iter_global, label='Global Variance', color='black', marker='*')
    ax3.legend()
    ax3.set_title("Variance History")
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Variance")

    # 4. RMSE History
    ax4 = fig.add_subplot(4, 3, 4)
    ax4.plot(rmse_local_true, label='Local RMSE', color='blue', marker='.')
    ax4.plot(rmse_global_true, label='Global RMSE', color='black', marker='*')
    ax4.legend()
    ax4.set_title("RMSE History")
    ax4.set_xlabel("Iterations")
    ax4.set_ylabel("RMSE")

    # 5. Lengthscale
    ax5 = fig.add_subplot(4, 3, 5)
    lengthscale_arr = np.array(lengthscale)
    ax5.plot(lengthscale_arr[:, 0], label='Lengthscale Dim 1', color='purple', marker='.')
    ax5.plot(lengthscale_arr[:, 1], label='Lengthscale Dim 2', color='green', marker='.')
    ax5.legend()
    ax5.set_title("Lengthscale Over Iterations")
    ax5.set_xlabel("Iterations")
    ax5.set_ylabel("Lengthscale")

    # 6. Noise
    ax6 = fig.add_subplot(4, 3, 6)
    ax6.plot(noise, label='Noise', color='orange', marker='.')
    ax6.set_title("Noise Over Iterations")
    ax6.set_xlabel("Iterations")
    ax6.set_ylabel("Noise")

    # 7. Covariance Elements
    ax7 = fig.add_subplot(4, 3, 7)
    ax7.plot(covar_totelements, label='Total Elements', color='red', marker='.')
    ax7.plot(covar_nonzeroelements, label='Non-Zero Elements', color='blue', marker='.')
    ax7.legend()
    ax7.set_title("Covariance Matrix Elements")
    ax7.set_xlabel("Iterations")
    ax7.set_ylabel("Count")

    # 8. AIC and BIC
    ax8 = fig.add_subplot(4, 3, 8)
    ax8.plot(AIC, label='AIC', color='blue', marker='.')
    ax8.plot(BIC, label='BIC', color='black', marker='*')
    ax8.legend()
    ax8.set_title("Information Criteria")
    ax8.set_xlabel("Iterations")
    ax8.set_ylabel("Value")

    # 9. RKHS Norm
    ax9 = fig.add_subplot(4, 3, 9)
    ax9.plot(f2_H_local, label='Local RKHS Norm', color='blue', marker='.')
    ax9.plot(f2_H_global, label='Global RKHS Norm', color='black', marker='*')
    ax9.legend()
    ax9.set_title("RKHS Norm")
    ax9.set_xlabel("Iterations")
    ax9.set_ylabel("Value")

    # Save and show
    plt.tight_layout()
    save_path = f"{image_path}/iteration_{iteration}.png"
    plt.savefig(save_path)
    plt.show()
