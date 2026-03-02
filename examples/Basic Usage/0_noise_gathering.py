from pathlib import Path
from dtempest.gw.generation.utils import query_noise

if __name__ == '__main__':
    """
    Noise gathering:
    500 second noise segments for each detector and times.
    """

    times = [ 	1268431194.1]  # or the preferred times
    ifolist = ('L1', 'H1', 'V1')

    files_dir = Path('')  # Main directory
    noise_dir = files_dir / 'Noise'  # Noise directory

    [query_noise(t, ifolist, noise_dir, verbose=True) for t in times]