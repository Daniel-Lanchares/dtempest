import torch
import numpy as np
from pathlib import Path

import gwosc.api.v2 as apiv2

from dtempest.gw import CBCEstimator
from dtempest.gw.catalog import Event, BarredEventDict, check_barred_events

def sample_and_save(event: str, model: CBCEstimator, outdir: Path = "", nsamples: int = 10000, **gwosc_kwargs) -> Path:

    merger = Event(event, **model.get_dataset_metadata(stage=0)["injection_kwargs"], **gwosc_kwargs)
    image = merger.data

    with torch.no_grad():
        sdict = model.sample_dict(nsamples, context=image)
    
    sample_dir = outdir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    filepath = sample_dir / f'{event}_{nsamples}'
    np.savez(filepath, **sdict)
    return filepath

def sample_catalog(
        catalog: str,
        model: CBCEstimator,
        outdir: Path = "",
        nsamples: int = 10000,
        exceptions: BarredEventDict = None,
        **gwosc_kwargs):
    if exceptions is None:
        exceptions = {}
    print(f"Samplig catalog {catalog}.")
    events = (event for event in apiv2.fetch_event_versions(catalogs=catalog) 
              if set(event["detectors"]) == set(model.ifos))

    for event in events:

        if check_barred_events(name, exceptions):
            continue
        
        if len(event["name"].split("_")) == 1:
            event["name"] = event["shortName"].split("-")[0]
        
        if (outdir / "samples" / f"{event['name']}_{nsamples}.npz").exists():
            print(f"\nSkipping event {event['name']} because it was already sampled.\n")
            continue

        print(f"\nSampling event {event['name']}.\n")
        sample_and_save(event["name"], model, outdir, nsamples, **gwosc_kwargs)


if __name__ == '__main__':
    """
    Sampling from trained model over an entire catalog (minus exceptions).
    """
    name = 'GP15_example' # Model name
    catalog = "GWTC-2.1-confident"
    # catalog = "GWTC-3-confident"

    files_dir = Path('') # Main directory
    train_dir = files_dir / 'Model'
    outdir = train_dir / 'training_test_0'
    

    flow = CBCEstimator.load_from_file(outdir / f'{name}.pt', device ='cpu') # Ensure that it leaves on the cpu to avoid cuda-availability problems
    flow.eval()

    print(f'{flow.name} metadata')
    flow.pprint_metadata()

    exceptions = {
        ("GW200306_093714", "GW200225_060421", "GW200220_124850", "GW200128_022011",
         "GW191222_033537", "GW191204_110529", "GW191129_134029", "GW191109_010717",
         "GW191103_012549", "GW190910_112807", "GW190527_092055", "GW190421_213856"):
            lambda evnt: f"\nSkipping event {evnt} Because it does not actually match the configuration.\n",
        ("GW200115_042309", "GW191219_163120", "GW190425"):
            lambda evnt: f"\nSkipping event {evnt}, it is not a BBH system.\n"

    }

    directory = files_dir / "Estimation Data" / flow.name

    sample_catalog(catalog, flow, directory, exceptions=exceptions)
