import numpy as np
import matplotlib.pyplot as plt

from dymos.examples.racecar.spline import get_spline, get_track_points
from dymos.examples.racecar.tracks import (
    AbuDhabi, Bahrain, Bahrain_short, Barcelona, Monaco, Monza, Spa,
    ovaltrack, doubleturn, multiturn, singleturn, straight, uturn,
)  # track curvature imports

tracks = {
    'ovaltrack': ovaltrack,
    'singleturn': singleturn,
    'doubleturn': doubleturn,
    'multiturn': multiturn,
    'uturn': uturn,
    'straight': straight,
    'Monaco': Monaco,
    'Barcelona': Barcelona,
    'Monza': Monza,
    'Spa': Spa,
    'Bahrain': Bahrain,
    'Bahrain_short': Bahrain_short,
    'AbuDhabi': AbuDhabi,
}

ncols = 4
nrows = int(np.ceil(len(tracks) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = axes.flatten()

for ax, (name, track) in zip(axes, tracks.items()):
    points = get_track_points(track)
    finespline, gates, gatesd, curv, slope = get_spline(points, s=0.0)

    ax.plot(finespline[0], finespline[1], '-k', lw=1.5)
    ax.plot(finespline[0][0], finespline[1][0], 'go', ms=6, label='start')
    ax.set_aspect('equal')
    ax.set_title(f'{name}\nL={track.get_total_length():.0f} m')
    ax.set_xticks([])
    ax.set_yticks([])

for ax in axes[len(tracks):]:
    ax.axis('off')

axes[0].legend(loc='upper right', fontsize=8)
fig.tight_layout()
plt.show()
