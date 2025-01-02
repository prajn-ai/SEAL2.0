import numpy as np
import matplotlib.pyplot as plt

def underwater_sound_spreading_spherical_2d(x_distance, y_distance, source_level):
    distance = np.sqrt(x_distance**2 + y_distance**2)
    intensity_loss = 20 * np.log10(distance)
    intensity_loss = source_level - intensity_loss
    return intensity_loss

def underwater_sound_spreading_spherical_2d_multisource(x_distance, y_distance, source_levels, x_sources, y_sources):
    intensity_loss = np.zeros_like(x_distance, dtype=float)

    for i in range(len(source_levels)):
        # Calculate distance
        distance = np.sqrt((x_distance - x_sources[i])**2 + (y_distance - y_sources[i])**2)
        distance = np.where(distance == 0, 1, distance)  # Replace zero distances with a small value

        # Compute intensity for the source
        intensity_loss_source = source_levels[i] - 20 * np.log10(distance)
        

        # Convert to linear scale and accumulate
        intensity_loss += 10**(intensity_loss_source / 10)

    # Convert back to decibel scale
    total_intensity_db = 10 * np.log10(intensity_loss)
    return total_intensity_db



x_distances = np.arange(-50, 51, 1).astype(float)
y_distances = np.arange(-50, 51, 1).astype(float)
x_grid, y_grid = np.meshgrid(x_distances, y_distances)

# Specify the locations and source levels of the two sources
x_sources = [-10.0, -20.0, 40.0, 38.0, 10.0, -5.0]  # Change these to the desired x-coordinates of the sources
y_sources = [-20.0, 30.0, 2.0, -40.0, -45.0, 26.0]  # Change these to the desired y-coordinates of the sources
source_levels = [100, 125, 120, 130, 115, 135]  # Change these to the desired source levels (one for each source)

# Calculate intensity loss for each combination of x and y distances for both sources
intensity_loss_2d = underwater_sound_spreading_spherical_2d_multisource(x_grid, y_grid, source_levels, x_sources, y_sources)
sound_field_data = np.column_stack((x_grid.flatten(), y_grid.flatten(), intensity_loss_2d.flatten()))
np.savetxt('6_source_sim.csv', sound_field_data, delimiter=',', header='x_distance,y_distance,spl', comments='')

print(np.max(intensity_loss_2d))
vmin = int(np.min(intensity_loss_2d))
vmax = int(np.max(intensity_loss_2d))

fig = plt.figure()

ax_spl = fig.add_subplot(111, projection='3d')
surf=ax_spl.plot_surface(x_grid, y_grid, intensity_loss_2d, cmap='viridis', vmin=vmin, vmax=vmax)
ax_spl.set_xlabel('X Distance (m)', fontsize=12)
ax_spl.set_ylabel('Y Distance (m)', fontsize=12)
# ax_spl.set_zlabel('Sound Intensity (dB)')
ax_spl.set_xticks(np.arange(-50, 51, 20))  # Custom ticks for x-axis
ax_spl.set_yticks(np.arange(-50, 51, 20))
# ax_spl.set_zticks(np.arange(vmin, vmax))
ax_spl.tick_params(axis='x', labelsize=10)
ax_spl.tick_params(axis='y', labelsize=10)
ax_spl.tick_params(axis='z', labelsize=10)
# ax_spl.set_zticks([])

ax_spl.set_xlim([-50, 50])
ax_spl.set_ylim([-50, 50])  
cbar = fig.colorbar(surf, ax=ax_spl, pad=0.05)
cbar.set_label('Sound Pressure Level (dB)', fontsize=12)
cbar.set_ticks(np.arange(vmin, vmax + 2, 5))
# fig.colorbar(surf, label="Sound Pressure Level dB") 
plt.show()