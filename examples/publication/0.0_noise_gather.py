"""
Noise gathering:

Noise samples for PSD estimations
"""

from pathlib import Path

from dtempest.gw.utils import query_noise

if __name__ == '__main__':
    times = [ 	1268431194.1]  # or the preferred times
    ifos = ('L1', 'H1', 'V1')

    files_dir = Path(__file__).parent / 'files'  # Main directory
    noise_dir = files_dir / 'Noise'  # Noise directory

    [query_noise(t, ifos, noise_dir, 500, verbose=True) for t in times]