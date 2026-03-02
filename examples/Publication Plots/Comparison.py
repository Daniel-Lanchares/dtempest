import json
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pesummary.gw.plots.latex_labels import GWlatex_labels

def get_mu_sigma(data):
    return data[1], (data[2] - data[0]) / 6

def get_gaussian_interval(data, percent=0.9):
    mu, sigma = get_mu_sigma(data)
    return norm.interval(percent, loc=mu, scale=sigma)

def generate_comparison_plot():

    select_params = ["chirp_mass_source", "luminosity_distance", "chi_eff"]

    # Flow and reference credible intervals can be computed from samples as:
    # samples.get_median_data(select_params, as_dict=True, quantiles=[0.05, 0.95])
    # on a dtempest.gw CBCComparisonSampleDict object.
    gwtc_data = json.load(open(estimation_dir / "GWTC-2.1_credible_intervals.json"))
    mydata = json.load(open(estimation_dir / "flow_credible_intervals.json"))
    base_p_data = json.load(open(estimation_dir / "alvares_et_al_3_sigma_data.json"))


    events = list(gwtc_data.keys())
    l = len(events)
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=plt.figaspect(0.5))
    for i, event in enumerate(events):
        for param, ax in zip(select_params, axs):
            base = base_p_data[event][param]
            ref = gwtc_data[event][param]
            ref = (ref["median"]-ref["lower"], ref["median"], ref["median"]+ref["upper"])
            mine = mydata[event][param]
            mine = (mine["median"]-mine["lower"], mine["median"], mine["median"]+mine["upper"])

            mu, sigma = get_mu_sigma(base)
            inter = get_gaussian_interval(base)

            rec = Rectangle((inter[0], (l-i)-0.25), inter[1] - inter[0], 0.5, fc="tab:purple", alpha=0.5)
            ref_rec = Rectangle((ref[0], (l - i) - 0.25), ref[2] - ref[0], 0.5, fc="tab:green", alpha=0.5)
            my_rec = Rectangle((mine[0], (l - i) - 0.25), mine[2] - mine[0], 0.5, fc="tab:blue", alpha=0.5)

            ax.add_patch(my_rec)
            ax.add_patch(ref_rec)
            ax.add_patch(rec)

            ax.scatter(mu, l-i, marker="o", color="b", alpha=0)
            ax.set_xlabel(GWlatex_labels[param])


    axs[0].set_yticks(range(1, l+1), labels=reversed(events))
    plt.tight_layout()
    fig.savefig(estimation_dir / "Estimation_comparison.pdf")


if __name__ == '__main__':
    """
    Comparison of credible intervals with references and previous work.
    """
    files_dir = Path('') # Main directory

    name = 'GP15_example'
    params = ["chirp_mass_source", "luminosity_distance", "chi_eff"]


    estimation_dir = files_dir / 'Estimation Data' / f'{name}'
    sample_dir = estimation_dir / 'refined samples'
    reference_dir = files_dir / "Public_Samples"

    nsamples = 10000

    generate_comparison_plot()