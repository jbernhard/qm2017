""" experimental data """

from collections import defaultdict
import logging
import pickle
from urllib.request import urlopen

import numpy as np
import yaml

from . import cachedir, systems


class HEPData:
    """
    Interface to a HEPData yaml file.

    """
    def __init__(self, inspire_rec, table, version=1):
        self._cent = None
        cachefile = (
            cachedir / 'hepdata' /
            'ins{}_table{}.pkl'.format(inspire_rec, table)
        )
        logging.debug('loading hepdata record %s table %s', inspire_rec, table)

        if cachefile.exists():
            logging.debug('reading from cache')
            with cachefile.open('rb') as f:
                self.data = pickle.load(f)
        else:
            logging.debug('not found in cache, downloading from hepdata.net')
            cachefile.parent.mkdir(exist_ok=True)
            with cachefile.open('wb') as f, urlopen(
                    'https://hepdata.net/download/table/'
                    'ins{}/Table{}/{}/yaml'.format(inspire_rec, table, version)
            ) as u:
                self.data = yaml.load(u)
                pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def x(self, name, case=True):
        """
        Get an independent variable ("x" data) with the given name.

        """
        for x in self.data['independent_variables']:
            x_name = x['header']['name']
            if (x_name if case else x_name.lower()) == name:
                return x['values']

    def y(self, name=None, **quals):
        """
        Get a dependent variable ("y" data) with the given name and qualifiers.

        """
        for y in self.data['dependent_variables']:
            if name is None or y['header']['name'] == name:
                y_quals = {q['name']: q['value'] for q in y['qualifiers']}
                if all(y_quals[k] == v for k, v in quals.items()):
                    return y['values']

    def cent(self):
        """
        Return a dict containing the centrality bins as a list of (low, high)
        tuples and the midpoints (x values) as a 1D np.array.

        """
        if self._cent is None:
            cent = [
                tuple(v[k] for k in ['low', 'high'])
                for v in self.x('centrality', case=False)
            ]
            self._cent = dict(
                cent=cent,
                x=np.array([(a + b)/2 for a, b in cent])
            )

        return self._cent

    def dataset(self, name=None, **quals):
        """
        Return a dict containing y values and errors along with centrality
        data.  Arguments are passed directly to self.y().

        """
        y = []
        yerr = defaultdict(list)

        for v in self.y(name, **quals):
            y.append(v['value'])
            for e in v['errors']:
                yerr[e.get('label', 'sum')].append(e['symerror'])

        return dict(
            y=np.array(y),
            yerr={k: np.array(v) for k, v in yerr.items()},
            **self.cent()
        )


def get_all_data():
    data = {s: {} for s in systems}

    # PbPb2760 and PbPb5020 dNch/deta
    for system, args, name in [
            ('PbPb2760', (880049, 1), 'D(N)/DETARAP'),
            ('PbPb5020', (1410589, 2),
             r'$\mathrm{d}N_\mathrm{ch}/\mathrm{d}\eta$'),
    ]:
        data[system]['dNch_deta'] = HEPData(*args).dataset(name)

    # PbPb2760 identified dN/dy and mean pT
    system = 'PbPb2760'

    for obs, table, combine_func in [
            ('dN_dy', 31, np.sum),
            ('mean_pT', 32, np.mean),
    ]:
        data[system][obs] = {}
        d = HEPData(1222333, table)
        for key, re_products in [
            ('pion', ['PI+', 'PI-']),
            ('kaon', ['K+', 'K-']),
            ('proton', ['P', 'PBAR']),
        ]:
            dsets = [
                d.dataset(RE='PB PB --> {} X'.format(i))
                for i in re_products
            ]

            data[system][obs][key] = dict(
                y=combine_func([d['y'] for d in dsets], axis=0),
                yerr={
                    e: combine_func([d['yerr'][e] for d in dsets], axis=0)
                    for e in dsets[0]['yerr']
                },
                **d.cent()
            )

    # PbPb2760 and PbPb5020 flows
    for system, tables in [
            ('PbPb5020', [1, 2, 2]),
            ('PbPb2760', [3, 4, 4]),
    ]:
        data[system]['vn'] = {}

        for n, t in enumerate(tables, start=2):
            data[system]['vn'][n] = HEPData(1419244, t).dataset(
                'V{}{{2, |DELTAETA|>1}}'.format(n)
            )

    return data


data = get_all_data()


def print_data(d, indent=0):
    """
    Pretty print the nested data dict.

    """
    prefix = indent * '  '
    for k in sorted(d):
        v = d[k]
        k = prefix + str(k)
        if isinstance(v, dict):
            print(k)
            print_data(v, indent + 1)
        else:
            if k.endswith('cent'):
                v = ' '.join(
                    str(tuple(int(j) if j.is_integer() else j for j in i))
                    for i in v
                )
            elif isinstance(v, np.ndarray):
                v = str(v).replace('\n', '')
            print(k, '=', v)


if __name__ == '__main__':
    print_data(data)
