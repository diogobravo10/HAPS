import json
import os
import time
import numpy as np
import pyvista as pv
import openmdao.api as om
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cartopy.crs as ccrs
import cartopy.feature as cfeature

##################################################
#
#
# Cartopy fails with debugger
#
#
###################################################

paths_file = 'solving_longitudinal_paths.json'
lateral_paths_file = 'solving_lateral_paths.json'

# Real-world location of the cylinder trajectory's center (local x=0, y=0)
center_lat = 38.7
center_lon = -28.0
_EARTH_RADIUS = 6371000.0  # m, mean Earth radius
g = 9.81  # m/s^2, consistent with the kinematics models

# Min. Lat 36N, Max. Lat 40N, Min. Long 30W, Max. Long 26W - shared by the lateral
# 2D plot and the 3D plot's basemap, so both show the same real-world footprint.
azores_extent = (-29., -27., 37.5, 39.5)


def local_xy_to_lonlat(x, y, lat0=center_lat, lon0=center_lon):
    """Flat-Earth (equirectangular) approximation converting local x/y (m, centered
    on lat0/lon0) to longitude/latitude in degrees. Fine at the km-scale extents of
    this trajectory; would need a proper geodesic for much larger ones."""
    lat = lat0 + np.degrees(y / _EARTH_RADIUS)
    lon = lon0 + np.degrees(x / (_EARTH_RADIUS * np.cos(np.radians(lat0))))
    return lon, lat


def lonlat_to_local_xy(lon, lat, lat0=center_lat, lon0=center_lon):
    """Inverse of local_xy_to_lonlat: longitude/latitude in degrees to local x/y in
    meters, true-to-scale under the same flat-Earth approximation."""
    y = np.radians(lat - lat0) * _EARTH_RADIUS
    x = np.radians(lon - lon0) * _EARTH_RADIUS * np.cos(np.radians(lat0))
    return x, y


def render_basemap_image(lon_min, lon_max, lat_min, lat_max, pixels=80):
    """Rasterize a cartopy basemap over the given lon/lat extent and return it as
    an (H, W, 4) RGBA array, north-up."""
    dpi = 100
    fig = plt.figure(figsize=(pixels / dpi, pixels / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1), projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor='#bcdff1')
    ax.add_feature(cfeature.LAND, facecolor='#e8e4d8')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_axis_off()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    plt.close(fig)
    return img


def add_basemap(ax3, x, y, lat0=center_lat, lon0=center_lon, margin=0.2, pixels=80, extent=None):
    """Texture-map a cartopy basemap onto the z=0 plane of a 3D axes.

    By default, covers the lon/lat footprint of the given local x/y trajectory points
    (in meters) plus a margin. Pass extent=(lon_min, lon_max, lat_min, lat_max) to
    instead render a fixed real-world box, converted to local x/y true-to-scale under
    the same flat-Earth approximation - x/y are then only used for the docstring's
    caller convenience and play no role in sizing the plane.

    pixels controls the resolution of the textured surface: plot_surface draws one
    quad per pixel, so this is an interactive-rotation-speed/visual-detail trade-off
    (80 -> ~6400 quads; the default 512 from an early version made rotating painfully
    slow for essentially no visible benefit on a coastline-scale map)."""
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        x_corners, y_corners = lonlat_to_local_xy(
            np.array([lon_min, lon_max, lon_min, lon_max]),
            np.array([lat_min, lat_min, lat_max, lat_max]), lat0, lon0)
        x_min, x_max = x_corners.min(), x_corners.max()
        y_min, y_max = y_corners.min(), y_corners.max()
        corner_lon = np.array([lon_min, lon_max])
        corner_lat = np.array([lat_min, lat_max])
    else:
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        pad_x = (x_max - x_min) * margin or 1.0
        pad_y = (y_max - y_min) * margin or 1.0
        x_min, x_max = x_min - pad_x, x_max + pad_x
        y_min, y_max = y_min - pad_y, y_max + pad_y

        corner_lon, corner_lat = local_xy_to_lonlat(
            np.array([x_min, x_max, x_min, x_max]), np.array([y_min, y_min, y_max, y_max]), lat0, lon0)

    img = render_basemap_image(corner_lon.min(), corner_lon.max(), corner_lat.min(), corner_lat.max(), pixels=pixels)
    img = np.flipud(img)  # image row 0 is north (top); flip so row 0 aligns with y_min below

    xs = np.linspace(x_min, x_max, img.shape[1] + 1)
    ys = np.linspace(y_min, y_max, img.shape[0] + 1)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)

    ax3.plot_surface(X, Y, Z, facecolors=img / 255.0, rstride=1, cstride=1,
                      shade=False, zorder=0, antialiased=False)
    return (x_min, x_max), (y_min, y_max)

# Anchored to this file's own location (not the current working directory),
# since the working directory a script runs with depends on how it's launched
# (e.g. many IDE "Run" buttons use the workspace root, not the file's folder)
# and a plain relative path can silently resolve outside the repo.
_this_dir = os.path.dirname(os.path.abspath(__file__))
plots_out_dir = os.path.join(_this_dir, '..', 'nice_plots_3d_trajectory', 'Trajectory Optimization', 'Cylinder')

# Shared phase color code - same mapping used across all figures
colors = {'climb': 'tab:blue', 'cruise': 'tab:green', 'descent': 'tab:red'}


def stitch(varname, case, phases=('climb', 'cruise', 'descent')):
    """Concatenate a timeseries variable across the given phases, in order."""
    return np.concatenate([
        case.get_val(f'traj.{phase}.timeseries.{varname}')[:, 0]
        for phase in phases
    ])


def load_case(paths, key):
    return om.CaseReader(paths[key]).get_case('final')


def main():
    with open(paths_file) as f:
        long_paths = json.load(f)
    with open(lateral_paths_file) as f:
        lat_paths = json.load(f)

    long_sol = load_case(long_paths, 'solution')
    long_sim = load_case(long_paths, 'simulation')
    lat_sol = load_case(lat_paths, 'solution')
    lat_sim = load_case(lat_paths, 'simulation')

    # --- Longitudinal: h(t), V(t), gamma(t) ---
    t_sol, h_sol, V_sol, gg_sol = (stitch(v, long_sol) for v in ('time', 'h', 'V', 'gg'))
    t_sim, h_sim, V_sim, gg_sim = (stitch(v, long_sim) for v in ('time', 'h', 'V', 'gg'))
    t_split1 = long_sol.get_val('traj.climb.timeseries.time')[-1, 0]
    t_split2 = long_sol.get_val('traj.cruise.timeseries.time')[-1, 0]

    gg_sol = np.rad2deg(gg_sol)
    gg_sim = np.rad2deg(gg_sim)

    # Which longitudinal phase each longitudinal solution point belongs to
    phase_of_long_t = np.where(t_sol < t_split1, 'climb',
                       np.where(t_sol < t_split2, 'cruise', 'descent'))

    fig1, axs1 = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    fig1.suptitle('Longitudinal profile')

    for name, color in colors.items():
        mask = phase_of_long_t == name
        if np.any(mask):
            axs1[0].plot(t_sol[mask], h_sol[mask], '-o', ms=4, color=color, label=name)
            axs1[1].plot(t_sol[mask], V_sol[mask], '-o', ms=4, color=color)
            axs1[2].plot(t_sol[mask], gg_sol[mask], '-o', ms=4, color=color)

    axs1[0].plot(t_sim, h_sim, '-', color='gray', lw=2, alpha=0.6, label='simulation')
    axs1[1].plot(t_sim, V_sim, '-', color='gray', lw=2, alpha=0.6)
    axs1[2].plot(t_sim, gg_sim, '-', color='gray', lw=2, alpha=0.6)

    axs1[0].axvline(t_split1, color='gray', ls='--', lw=2)
    axs1[0].axvline(t_split2, color='gray', ls='--', lw=2)
    axs1[0].set_ylabel('h (m)')
    axs1[0].legend()

    axs1[1].axvline(t_split1, color='gray', ls='--', lw=2)
    axs1[1].axvline(t_split2, color='gray', ls='--', lw=2)
    axs1[1].set_ylabel('V (m/s)')

    axs1[2].axvline(t_split1, color='gray', ls='--', lw=2)
    axs1[2].axvline(t_split2, color='gray', ls='--', lw=2)
    axs1[2].set_xlabel('time (s)')
    axs1[2].set_ylabel('gamma (deg)')

    # --- Lateral: y(x), tt(t) ---
    # The lateral maneuver is a single phase ('phase0' for the cylinder loiter model,
    # or seg0, seg1, ... for the older multi-segment layout), so seg_names still
    # drives the stitching either way.
    seg_names = lat_paths['seg_names']
    x_sol, y_sol, t_lat_sol, tt_sol = (
        stitch(v, lat_sol, seg_names) for v in ('x', 'y', 'time', 'tt'))
    x_sim, y_sim, t_lat_sim, tt_sim = (
        stitch(v, lat_sim, seg_names) for v in ('x', 'y', 'time', 'tt'))

    tt_sol = np.rad2deg(tt_sol)
    tt_sim = np.rad2deg(tt_sim)

    # Which longitudinal phase each lateral solution point falls into, based on
    # absolute time - same color code as the longitudinal plots and the 3D plot.
    phase_of_lat_t = np.where(t_lat_sol < t_split1, 'climb',
                     np.where(t_lat_sol < t_split2, 'cruise', 'descent'))

    fig2 = plt.figure(figsize=(6, 6))
    ax_map = fig2.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax_tt = fig2.add_subplot(2, 1, 2)
    fig2.suptitle('Lateral trajectory')

    # Same real-world footprint and land/ocean/coastline styling as the 3D plot's
    # basemap, but drawn natively by cartopy instead of texture-mapped onto a surface -
    # a plain 2D GeoAxes doesn't need the raster workaround mplot3d does.
    ax_map.set_extent(azores_extent, crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.OCEAN, facecolor='#bcdff1')
    ax_map.add_feature(cfeature.LAND, facecolor='#e8e4d8')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)

    for name, color in colors.items():
        mask = phase_of_lat_t == name
        if np.any(mask):
            lon_sol_m, lat_sol_m = local_xy_to_lonlat(x_sol[mask], y_sol[mask])
            ax_map.plot(lon_sol_m, lat_sol_m, '-o', ms=4, color=color, label=name, transform=ccrs.PlateCarree())
            ax_tt.plot(t_lat_sol[mask], tt_sol[mask], '-o', ms=4, color=color)

    lon_sim, lat_sim = local_xy_to_lonlat(x_sim, y_sim)
    ax_map.plot(lon_sim, lat_sim, '-', color='gray', lw=2, alpha=0.6, label='simulation', transform=ccrs.PlateCarree())
    ax_tt.plot(t_lat_sim, tt_sim, '-', color='gray', lw=2, alpha=0.6)

    ax_map.gridlines(draw_labels=True)
    ax_map.legend()

    ax_tt.set_xlabel('time (s)')
    ax_tt.set_ylabel('tt (deg)')

    # --- 3D (x, y, h): lateral ground track combined with the longitudinal altitude
    # profile at those same times, colored by which longitudinal phase each point
    # falls into. Solution points are connected by a line (in time order); the
    # simulation is overlaid as a thin continuous line for comparison. ---
    h_at_lat_sol_t = np.interp(t_lat_sol, t_sol, h_sol)
    h_at_lat_sim_t = np.interp(t_lat_sim, t_sim, h_sim)

    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111, projection='3d')
    # By default mplot3d recomputes each artist's draw order from its actual 3D depth
    # every frame, which can let the (huge, flat) basemap surface paint over the
    # trajectory depending on view angle even though it's physically below it. Disabling
    # that makes it respect the zorder we set explicitly instead (basemap=0, trajectory=10).
    ax3.computed_zorder = False

    basemap_xlim, basemap_ylim = add_basemap(ax3, np.concatenate([x_sol, x_sim]), np.concatenate([y_sol, y_sim]),
                                              extent=azores_extent, pixels=200)

    for name, color in colors.items():
        mask = phase_of_lat_t == name
        if np.any(mask):
            ax3.plot(x_sol[mask], y_sol[mask], h_at_lat_sol_t[mask], '-o', ms=4, color=color, label=name, zorder=10)

    ax3.plot(x_sim, y_sim, h_at_lat_sim_t, '-', color='gray', lw=2, alpha=0.6, label='simulation', zorder=10)

    # Pin the floor to exactly azores_extent's own bounds (via basemap_xlim/basemap_ylim,
    # which add_basemap derived from that same extent) - otherwise mplot3d's default
    # autoscale margin leaves a visible gap between the textured plane and the walls.
    ax3.set_xlim(*basemap_xlim)
    ax3.set_ylim(*basemap_ylim)

    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_zlabel('h (m)')
    ax3.set_title('3D trajectory')
    ax3.legend()

    os.makedirs(plots_out_dir, exist_ok=True)
    fig1.savefig(os.path.join(plots_out_dir, 'longitudinal_profile.png'), dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(plots_out_dir, 'lateral_trajectory.png'), dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(plots_out_dir, 'trajectory_3d.png'), dpi=150, bbox_inches='tight')

    plt.show()



_this_dir = os.path.dirname(os.path.abspath(__file__))

# Expected STL body-axes convention: standard aerospace body frame -
# X forward (along the chord, nose direction), Y spanwise (right wingtip positive),
# Z down. Update this path once the corrected export is in place.
stl_path = os.path.join(_this_dir, '..', 'zephyr.step.STL')


def load_pose_timeseries(n_frames=300):
    """Build a combined 6-DOF pose history (position + heading/pitch/bank), resampled
    onto a uniform time grid so the animation plays at constant speed.

    Position (x, y) and heading come from the lateral (constant-radius loiter)
    solution; altitude h and flight-path angle gamma (pitch) come from the
    longitudinal solution, interpolated onto the lateral solution's own time grid
    (they're two separate collocated solves sharing the same absolute mission
    timeline - see solving_cascade_V2.py). Bank angle isn't a state in the cylinder
    kinematics model (lateral_kinematics_cylinder.py has no phi), so it's derived
    from the standard coordinated-turn relation phi = atan(V^2 / (g*R)) instead.
    """
    with open(paths_file) as f:
        long_paths = json.load(f)
    with open(lateral_paths_file) as f:
        lat_paths = json.load(f)

    long_sol = load_case(long_paths, 'solution')
    lat_sol = load_case(lat_paths, 'solution')

    t_sol, h_sol, V_sol, gg_sol = (stitch(v, long_sol) for v in ('time', 'h', 'V', 'gg'))

    seg_names = lat_paths['seg_names']
    t_lat, x_lat, y_lat, tt_lat = (stitch(v, lat_sol, seg_names) for v in ('time', 'x', 'y', 'tt'))
    R = lat_sol.get_val('traj.phase0.parameters:R')[0]

    V_lat = np.interp(t_lat, t_sol, V_sol)
    h_lat = np.interp(t_lat, t_sol, h_sol)
    gg_lat = np.interp(t_lat, t_sol, gg_sol)

    psi = tt_lat + np.pi / 2  # heading = direction of travel, tangent to the circle
    # tt increases counterclockwise (viewed from above) -> a left turn, so the
    # coordinated-turn bank is to the left (negative, in the +phi=right-wing-down
    # convention used by body_to_enu_rotation below).
    phi = -np.arctan2(V_lat ** 2, g * R)

    t_uniform = np.linspace(t_lat.min(), t_lat.max(), n_frames)
    x_u = np.interp(t_uniform, t_lat, x_lat)
    y_u = np.interp(t_uniform, t_lat, y_lat)
    h_u = np.interp(t_uniform, t_lat, h_lat)
    psi_u = np.interp(t_uniform, t_lat, psi)
    gamma_u = np.interp(t_uniform, t_lat, gg_lat)
    phi_u = np.interp(t_uniform, t_lat, phi)

    return t_uniform, x_u, y_u, h_u, psi_u, gamma_u, phi_u


def body_to_enu_rotation(psi, gamma, phi):
    """3x3 rotation taking a standard aerospace body vector (X=forward/nose,
    Y=right/spanwise, Z=down) to world ENU coordinates (X=East (local x), Y=North
    (local y), Z=Up (h)).

    psi is heading in radians, measured counterclockwise from East (matching
    tt + pi/2 from the cylinder kinematics - NOT compass bearing); gamma is the
    flight-path angle in radians (positive = climbing); phi is bank angle in
    radians (positive = right wing down). Internally this converts to the
    standard 3-2-1 (yaw-pitch-roll) Euler sequence in North-East-Down, then maps
    NED -> ENU, since that's the convention nearly every aerospace reference uses.
    """
    psi_compass = np.pi / 2 - psi
    cy, sy = np.cos(psi_compass), np.sin(psi_compass)
    ct, st = np.cos(gamma), np.sin(gamma)
    cr, sr = np.cos(phi), np.sin(phi)

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    R_ned = Rz @ Ry @ Rx

    ned_to_enu = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    return ned_to_enu @ R_ned


# Maps the STL's own native axes to the (forward, right, down) body frame
# body_to_enu_rotation expects. Determined by inspecting the mesh's own point
# extremes rather than assumed: the nose sits at X=0 and the tail boom at
# X=13520 (max), so +X points aft, not forward - forward is -X. Right is then
# fixed by requiring forward x right = down (a proper, non-mirroring rotation)
# with down confirmed as +Z: that forces right = -Y.
_NATIVE_TO_BODY = np.diag([-1.0, -1.0, 1.0])


def find_fuselage_reference_point(mesh, tol=50.0):
    """The raw STL origin (0,0,0) isn't on the fuselage - it's off near the right
    wingtip (Y=0 sits just short of the Y-min wingtip edge), so pinning that point
    to the trajectory made the aircraft fly nose/wingtip-first instead of body-first.
    This instead finds the actual nose tip on the fuselage centerline: the cluster
    of mesh points at the nose station (native X close to its mesh-wide minimum)
    and at midspan (native Y close to the span's midpoint), averaged together."""
    pts = mesh.points
    x_nose = pts[:, 0].min()
    y_center = (pts[:, 1].min() + pts[:, 1].max()) / 2
    mask = (np.abs(pts[:, 0] - x_nose) < tol) & (np.abs(pts[:, 1] - y_center) < tol)
    if not np.any(mask):
        raise ValueError('No mesh points found near the nose/midspan - increase tol.')
    return pts[mask].mean(axis=0)


def pose_matrix(x, y, h, psi, gamma, phi, scale, origin_native=(0.0, 0.0, 0.0)):
    """4x4 transform mapping raw STL points (native units, native mesh axes) directly
    to world (x, y, h) meters - combines the native->body axis fix-up, the
    body->ENU rotation, a uniform scale (mm -> m times any visual exaggeration),
    and the translation to the aircraft's current position, in one matrix
    suitable for Actor.user_matrix.

    origin_native is the point (in raw mesh coordinates) that gets pinned to
    (x, y, h) - i.e. the point that actually follows the trajectory. Defaults to
    the mesh's own (0,0,0), but that's rarely where you want it (see
    find_fuselage_reference_point)."""
    R = (body_to_enu_rotation(psi, gamma, phi) @ _NATIVE_TO_BODY) * scale
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, h] - R @ np.asarray(origin_native)
    return T


def add_basemap_pyvista(pl, extent=azores_extent, pixels=400):
    """Texture-map a cartopy basemap onto a flat z=0 plane, covering the same
    real-world extent as plot_cascade.py's 3D plot. Unlike matplotlib's mplot3d
    (which needs one quad per pixel to fake a texture and still gets the z-order
    wrong against other artists), VTK does real texture mapping and real depth
    buffering, so this is a single quad and composites correctly with the
    aircraft/path automatically."""
    lon_min, lon_max, lat_min, lat_max = extent
    img = render_basemap_image(lon_min, lon_max, lat_min, lat_max, pixels=pixels)
    # Unlike matplotlib's plot_surface (see plot_cascade.add_basemap), PyVista's
    # texture_map_to_plane() already puts image row 0 at the plane's +Y (north) edge,
    # so no flip here - flipping (as an earlier version did) mirrors north/south and
    # shifts every feature (e.g. islands) to the wrong side of the trajectory relative
    # to the correctly-oriented 2D cartopy plot. Verified with a marked test texture.

    # The loiter loop is centered on local (0,0) by construction (x=R*cos(tt),
    # y=R*sin(tt) in lateral_kinematics_cylinder.py has no offset), so pinning that
    # local origin to (center_lat, center_lon) here is what puts the trajectory's
    # radius center at the real-world coordinates given.
    x_corners, y_corners = lonlat_to_local_xy(
        np.array([lon_min, lon_max, lon_min, lon_max]), np.array([lat_min, lat_min, lat_max, lat_max]),
        lat0=center_lat, lon0=center_lon)
    x_min, x_max = x_corners.min(), x_corners.max()
    y_min, y_max = y_corners.min(), y_corners.max()

    plane = pv.Plane(center=((x_min + x_max) / 2, (y_min + y_max) / 2, 0), direction=(0, 0, 1),
                      i_size=x_max - x_min, j_size=y_max - y_min, i_resolution=1, j_resolution=1)
    plane.texture_map_to_plane(inplace=True)
    pl.add_mesh(plane, texture=pv.Texture(img[:, :, :3]))
    return (x_min, x_max), (y_min, y_max)


default_output_gif = os.path.join(plots_out_dir, 'aircraft_animation.gif')


def _build_scene(mesh, x, y, h, off_screen, show_path, show_basemap):
    """Construct a fresh Plotter with the basemap/path/aircraft added, ready for
    its actor's user_matrix to be updated frame by frame."""
    pl = pv.Plotter(off_screen=off_screen)

    if show_basemap:
        add_basemap_pyvista(pl)

    if show_path:
        path = np.column_stack([x, y, h])
        pl.add_mesh(pv.lines_from_points(path), color='gray', line_width=2)

    actor = pl.add_mesh(mesh, color='#3b6ea5')
    pl.add_axes()
    return pl, actor


# User-picked (position, focal_point, up) - captured from pl.camera_position after
# manually framing the preferred view (small island to the north) in an interactive
# session.
PREFERRED_CAMERA_POSITION = [(125852.81118727505, -200077.13490616553, 284093.78255628323),
 (15821.553419882259, -8596.921817865681, 12796.992046896816),
 (-0.14713233157700878, 0.7790144688209152, 0.6094985925926169)]


def set_north_up_camera(pl):
    """Apply the preferred camera framing (fixed, not scene-dependent) that puts
    the small island near (28W, 39N) toward the top/north of the image."""
    pl.camera_position = PREFERRED_CAMERA_POSITION


def animate_aircraft(stl_path=stl_path, n_frames=300, fps=20, model_scale=None,
                      output_gif=default_output_gif, show_path=True, show_basemap=True,
                      interactive_preview=True):
    """Animate the STL aircraft model flying the combined longitudinal/lateral
    trajectory in its own 3D plot.

    model_scale: uniform multiplier applied on top of the mm->m unit conversion.
    If None, it's auto-picked so the model's wingspan is ~10% of the trajectory's
    horizontal extent - the loiter radius is tens of km while the aircraft is
    ~25m across, so true 1:1 scale would render it as an invisible speck.

    interactive_preview: if True (default), first opens a live PyVista window that
    loops the animation continuously until you close it by hand, before anything
    is written to disk.

    output_gif: if given (and not None), saved after the interactive preview window
    is closed (or immediately, if interactive_preview=False). Defaults to
    nice_plots_3d_trajectory/Trajectory Optimization/Cylinder/aircraft_animation.gif,
    alongside plot_cascade.py's own figures.
    """
    t, x, y, h, psi, gamma, phi = load_pose_timeseries(n_frames=n_frames)

    mesh = pv.read(stl_path)
    origin_native = find_fuselage_reference_point(mesh)
    mm_to_m = 0.001
    wingspan_native = mesh.bounds[3] - mesh.bounds[2]  # Y bounds = spanwise extent
    wingspan_m = wingspan_native * mm_to_m

    if model_scale is None:
        traj_extent = max(x.max() - x.min(), y.max() - y.min())
        exaggeration = 0.1 * traj_extent / wingspan_m if wingspan_m > 0 else 1.0
        scale = mm_to_m * exaggeration
    else:
        scale = mm_to_m * model_scale

    def pose(i):
        return pose_matrix(x[i], y[i], h[i], psi[i], gamma[i], phi[i], scale, origin_native)

    if interactive_preview:
        pl, actor = _build_scene(mesh, x, y, h, off_screen=False,
                                  show_path=show_path, show_basemap=show_basemap)
        actor.user_matrix = pose(0)
        set_north_up_camera(pl)

        # Detect a real window close via VTK's own ExitEvent (fired when the title
        # bar's X - or 'q' - terminates the interactor), rather than polling
        # pl._closed: that flag isn't reliably set by an OS-level close on Windows,
        # so a "while not pl._closed" loop can spin forever afterward, hammering a
        # broken render window and flooding the console with VTK warnings.
        closed = {'flag': False}
        pl.iren.add_observer('ExitEvent', lambda *_: closed.__setitem__('flag', True))

        pl.show(auto_close=False, interactive_update=True)
        i = 0
        while not closed['flag']:
            actor.user_matrix = pose(i % n_frames)
            pl.render()  # explicit, rather than relying on update()'s internal throttling
            pl.iren.process_events()
            time.sleep(1.0 / fps)
            i += 1

        # Print the exact (position, focal_point, up) you left the camera at, so it
        # can be pasted into PREFERRED_CAMERA_POSITION or handed to another plot.
        print('Final camera_position:', pl.camera_position)
        pl.close()

    if output_gif is not None:
        pl, actor = _build_scene(mesh, x, y, h, off_screen=True,
                                  show_path=show_path, show_basemap=show_basemap)
        actor.user_matrix = pose(0)
        set_north_up_camera(pl)
        os.makedirs(os.path.dirname(output_gif), exist_ok=True)
        pl.open_gif(output_gif, fps=fps)
        for i in range(n_frames):
            actor.user_matrix = pose(i)
            pl.write_frame()
        pl.close()



if __name__ == '__main__':
    print('start')
    main()
    animate_aircraft()
    print('end')
