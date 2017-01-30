""" model output """

import logging
from pathlib import Path
import pickle

from hic import flow
import numpy as np
from sklearn.externals import joblib

from . import workdir, cachedir, systems, expt
from .design import Design


# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'


class ModelData:
    """
    Helper class for event-by-event model data.  Reads binary data files and
    computes centrality-binned observables.

    """
    dtype = np.dtype([
        ('initial_entropy', float_t),
        ('mult_factor', float_t),
        ('nsamples', int_t),
        ('dNch_deta', float_t),
        ('dN_dy', [(s, float_t) for s in ['pion', 'kaon', 'proton']]),
        ('mean_pT', [(s, float_t) for s in ['pion', 'kaon', 'proton']]),
        ('M', int_t),
        ('Qn', complex_t, 6),
    ])

    def __init__(self, *files):
        # read each file using the above dtype and treat each as a minimum-bias
        # event sample
        def load_events(f):
            logging.debug('loading %s', f)
            d = np.fromfile(str(f), dtype=self.dtype)
            d.sort(order='dNch_deta')
            return d

        self.events = [load_events(f) for f in files]

    def observables_like(self, data, *keys):
        """
        Compute the same centrality-binned observables as contained in `data`
        with the same nested dict structure.

        This function calls itself recursively, each time prepending to `keys`.

        """
        try:
            x = data['x']
            cent = data['cent']
        except KeyError:
            return {
                k: self.observables_like(v, k, *keys)
                for k, v in data.items()
            }

        def _compute_bin():
            """
            Choose a function to compute the current observable for a single
            centrality bin.

            """
            obs_stack = list(keys)
            obs = obs_stack.pop()

            if obs == 'dNch_deta':
                return lambda events: events[obs].mean()

            if obs == 'dN_dy':
                species = obs_stack.pop()
                return lambda events: events[obs][species].mean()

            if obs == 'mean_pT':
                species = obs_stack.pop()
                return lambda events: np.average(
                    events[obs][species],
                    weights=events['dN_dy'][species]
                )

            if obs == 'vn':
                n = obs_stack.pop()
                return lambda events: flow.Cumulant(
                    events['M'], *events['Qn'].T[1:]
                ).flow(n, 2, imaginary='zero')

        compute_bin = _compute_bin()

        def compute_all_bins(events):
            n = events.size
            bins = [
                events[int((1 - b/100)*n):int((1 - a/100)*n)]
                for a, b in cent
            ]

            return list(map(compute_bin, bins))

        return dict(
            x=x, cent=cent,
            Y=np.array(list(map(compute_all_bins, self.events))).squeeze()
        )


def observables(system, map_point=False):
    """
    Compute model observables for the given system to match the corresponding
    experimental data.

    """
    if map_point:
        files = [Path('map', system)]
        cachefile = Path(system + '_map')
    else:
        # expected filenames for each design point
        files = [Path(system, p) for p in Design(system).points]
        cachefile = Path(system)

    files = [workdir / 'model_output' / f.with_suffix('.dat') for f in files]
    cachefile = cachedir / 'model' / cachefile.with_suffix('.pkl')

    if cachefile.exists():
        # use the cache unless any of the model data files are newer
        # this DOES NOT check any other logical dependencies, e.g. the
        # experimental data
        # to force recomputation, delete the cache file
        mtime = cachefile.stat().st_mtime
        if all(f.stat().st_mtime < mtime for f in files):
            logging.debug('loading observables cache file %s', cachefile)
            return joblib.load(cachefile)
        else:
            logging.debug('cache file %s is older than event data', cachefile)
    else:
        logging.debug('cache file %s does not exist', cachefile)

    logging.info(
        'loading %s%s event data and computing observables',
        system,
        '_map' if map_point else ''
    )

    # identified particle data are not yet available for PbPb5020
    # create dummy entries for these observables so that they are computed for
    # the model
    if system == 'PbPb5020':
        system_data = expt.data[system].copy()
        for obs in ['dN_dy', 'mean_pT']:
            system_data[obs] = expt.data['PbPb2760'][obs]
    else:
        system_data = expt.data[system]

    data = ModelData(*files).observables_like(system_data)

    logging.info('writing cache file %s', cachefile)
    cachefile.parent.mkdir(exist_ok=True)
    joblib.dump(data, cachefile, protocol=pickle.HIGHEST_PROTOCOL)

    return data


data = {s: observables(s) for s in systems}
map_data = {s: observables(s, map_point=True) for s in systems}


if __name__ == '__main__':
    from pprint import pprint
    print('design:')
    pprint(data)
    print('map:')
    pprint(map_data)
