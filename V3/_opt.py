import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime, timedelta
import solving_longitudinal
from concurrent.futures import ProcessPoolExecutor, as_completed


def stitch(varname, prob):
    """Concatenate a timeseries variable across the climb, cruise and descent phases."""
    return np.concatenate([
        prob.get_val(f'traj.{phase}.timeseries.{varname}')[:, 0]
        for phase in ('climb', 'cruise', 'descent')
    ])



def soc_month(days_array, latitudes_array, M_sw=3.7, mbat_sw=2.0):
    """SOC grid (latitudes x days) for one month's days - one parallel work unit
    of `filtering_yearly_mean_power_contour`.

    Cells are the end-of-mission SOC, 0 if SOC <= 0.2, or -1 if the optimization
    did not complete successfully.
    """
    # Unique problem name per month so parallel workers don't share the same
    # OpenMDAO output directory / recorder file
    prob_name = f"solving_longitudinal_soc_{days_array[0]:%Y_%m}"

    SOC_month = np.zeros((len(latitudes_array), len(days_array)))

    for lat_idx, lat in enumerate(latitudes_array):
        for day_idx, current_day in enumerate(days_array):
            prob, exp_out, success = solving_longitudinal.main(M_sw=M_sw, mbat_sw=mbat_sw, start_date=current_day, lat=lat,
                                                               soc_initial=0.2, prob_name=prob_name)

            if not success:
                SOC_month[lat_idx, day_idx] = -1
                print(f'Optimization failed at latitude {lat:.2f} deg, day {current_day:%Y-%m-%d}: SOC = -1')
                continue

            SOC_array = stitch('SOC', prob)
            SOC = 1 - (max(SOC_array) - SOC_array[-1]) if max(SOC_array) > 1 else SOC_array[-1]
            SOC_month[lat_idx, day_idx] = SOC if SOC > 0.2 else 0

            print(f'Processed latitude {lat:.2f} deg, day {current_day:%Y-%m-%d}: SOC = {SOC_month[lat_idx, day_idx]:.2f}')

    return SOC_month


def filtering_yearly_mean_power_contour(design_vars, year, latitudes_array, max_workers=12):
    """Like `yearly_mean_power_contour` but filter cells where available
    power (from batteries or propulsion limit) is insufficient.

    Each month is solved in parallel (`soc_month`) and its columns are stitched
    back into SOC_distribution. Produces and shows a contour plot (PNG file).

    `year` is the list of per-month day arrays returned by `build_year`.
    """

    M_Sw, Mbat_Sw = design_vars[0], design_vars[1]

    # Unpack the months back into a single, chronological day axis
    days_array = np.concatenate(year)

    start_day = days_array[0]
    total_days = (days_array[-1] - start_day).days + 1
    day_numbers = np.array([
        (current_day - start_day).days + 1
        for current_day in days_array
    ], dtype=int)

    SOC_distribution = np.zeros((len(latitudes_array), len(days_array)))

    # Column offset of each month inside SOC_distribution
    offsets = np.cumsum([0] + [len(month) for month in year[:-1]])

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(soc_month, month, latitudes_array, M_sw=M_Sw, mbat_sw=Mbat_Sw): (offset, len(month), f"{month[0]:%Y-%m}")
                   for month, offset in zip(year, offsets)}
        for fut in as_completed(futures):
            offset, n_days, month_label = futures[fut]
            SOC_distribution[:, offset:offset + n_days] = fut.result()
            print(f'{month_label} done')


    fig, ax = plt.subplots(figsize=(14, 7))
    max_SOC = float(np.max(SOC_distribution))
    levels = np.arange(0, max_SOC + 0.01, 0.1)
    if levels.size < 2:
        levels = np.array([0.0, 1.0])
    contour = ax.contourf(day_numbers, latitudes_array, SOC_distribution, levels=levels, cmap='cividis')

    ax.set_xlabel('Day of year')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title(f'SOC distribution')
    ax.set_xlim(1, total_days)

    square_x = [1, 1, 365, 365, 1]
    square_y = [33, 43, 43, 33, 33]
    line_azores, = ax.plot(square_x, square_y, color='red', linewidth=2, label='Azores EEZ')    
    
    patch_azores = Patch(
        facecolor='none',
        edgecolor=line_azores.get_color(),
        linewidth=line_azores.get_linewidth(),
        label=line_azores.get_label()
    )

    square_x = [1, 1, 365, 365, 1]
    square_y = [-28, -10, -10, -28, -28]
    line_moz, = ax.plot(square_x, square_y, color='orange', linewidth=2, label='Mozambique EEZ')
    
    patch_moz = Patch(
        facecolor='none',
        edgecolor=line_moz.get_color(),
        linewidth=line_moz.get_linewidth(),
        label=line_moz.get_label()
    )

    ax.legend(handles=[patch_azores, patch_moz], loc='upper right')

    # ax.scatter(355, 43, marker='x', color='red', s=150, linewidths=2)

    fig.colorbar(contour, ax=ax, ticks=levels, label=r'SOC')
    plt.tight_layout()
    filename = f'fig_mean_power_distribution.png'
    plt.savefig(filename, dpi=200)
    plt.pause(0.1)
    plt.show(block=False)


    print(f'Contour image saved to {filename}')

def build_year(start_date, end_date, dday):
    """Split [start_date, end_date] into calendar months, sampled every `dday` days.

    Returns a list with one entry per month; each entry is an object array of the
    sampled datetimes falling in that month (months with no samples are dropped).
    """
    days = [start_date + timedelta(days=i) for i in range(0, (end_date - start_date).days + 1, dday)]
    months = {}
    for d in days:
        months.setdefault((d.year, d.month), []).append(d)
    return [np.array(months[key], dtype=object) for key in sorted(months)]


def obj_fun(design_vars, days_array):
    """Count how many days of `days_array` have at least one latitude in
    [S_lat, N_lat] where the mission ends with SOC > 0.2.

    Returns the score, or 1e9 if the design violates the mass constraint.
    """

    N_lat = 43
    S_lat = 33
    dlat = 10

    score = 0
    # m_sw, mbat_sw, mprop_sw = design_vars[0], design_vars[1], design_vars[2]

    # if 0.9 * M_Sw - Mbat_Sw - Mprop_Sw < 1.5: # ensure carrying ability of 0.15 + structural mass of 0.2
    #     return 1e9
    
    M_Sw, Mbat_Sw = design_vars[0], design_vars[1]
    if 0.8 * M_Sw - Mbat_Sw < 0.5: # ensure carrying ability of 0.15 + structural mass of 0.2
        return 1e9

    # Unique problem name per month so parallel workers don't share the same
    # OpenMDAO output directory / recorder file
    prob_name = f"solving_longitudinal_{days_array[0]:%Y_%m}"

    latitudes_array = np.arange(S_lat, N_lat + dlat, dlat, dtype=float)

    for day in days_array:
        for lat in latitudes_array:
            prob, exp_out, success = solving_longitudinal.main(M_sw=M_Sw, mbat_sw=Mbat_Sw, start_date=day, lat=lat,
                                                               soc_initial=0.2, prob_name=prob_name)

            if not success:
                print(f'Optimization failed at latitude {lat:.2f} deg, day {day:%Y-%m-%d}')
                continue

            SOC_array = stitch('SOC', prob)
            SOC = 1 - (max(SOC_array) - SOC_array[-1]) if max(SOC_array) > 1 else SOC_array[-1]

            print(f'Processed latitude {lat:.2f} deg, day {day:%Y-%m-%d}: SOC = {SOC:.2f}')

            if SOC > 0.2:
                score = score + 1
                break

    return score


if __name__ == '__main__':

    start_date = datetime(2027, 1, 1, 6, 0)
    end_date = datetime(2028, 1, 31, 6, 0)
    dday = 10

    year = build_year(start_date, end_date, dday)
    x0 = [3.7, 2]

    # scores = {}
    # with ProcessPoolExecutor(max_workers=12) as exe:
    #     futures = {exe.submit(obj_fun, x0, month): f"{month[0]:%Y-%m}" for month in year}
    #     for fut in as_completed(futures):
    #         month_label = futures[fut]
    #         scores[month_label] = fut.result()
    #         print(f'{month_label}: score = {scores[month_label]}')

    # total_score = sum(scores[m] for m in sorted(scores))
    # print(f'Total score: {total_score}')



    N_lat = 60
    S_lat = -60
    dlat = 15

    latitudes_array = np.arange(S_lat, N_lat + dlat, dlat, dtype=float)

    filtering_yearly_mean_power_contour(x0, year, latitudes_array)
    a=1
