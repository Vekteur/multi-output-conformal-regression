"""
Selects which models, methods and hyperparameters to run.
Models, methods or hyperparameters are declared using HP(name=value) or HP(name=[value1, value2, ...]).
Join combines sets of hyperparameters in a cartesian product (similar to a grid search).
Union concatenates sets of hyperparameters.
"""

import collections

from moc.utils.hparams import HP, Join, Union


def get_default_tuning(config):
    default_methods = [
        'M-CP',
        'DR-CP',
        'C-HDR', 
        'HDR-H', 
        'PCP',
        'HD-PCP',
        'C-PCP',
        'CopulaCPTS',
    ]
    mqf2_methods = default_methods + [
        'STDQR',
        'L-CP',
        'L-H',
    ]

    # Posthoc grid
    posthoc_grid = Join(
        HP(method=default_methods),
    )
    posthoc_grid_mqf2 = Join(
        HP(method=mqf2_methods),
    )

    # DRF + KDE
    drf_kde = Join(
        HP(model='DRF-KDE'),
        HP(posthoc_grid=posthoc_grid)
    )

    # MQF2
    mqf2 = Join(
        HP(model='MQF2'),
        HP(posthoc_grid=posthoc_grid_mqf2)
    )

    # Mixture
    mixture = Join(
        HP(model='Mixture'),
        HP(mixture_size=[1, 10]),
        HP(posthoc_grid=posthoc_grid)
    )

    return Union(
        mixture,
        drf_kde,
        mqf2,
    )


def get_tuning_glow(config):
    methods = [
        'M-CP',
        'DR-CP',
        'C-HDR', 
        'HDR-H', 
        'PCP',
        'HD-PCP',
        'C-PCP',
        'STDQR',
        'L-CP',
        'L-H',
        'CopulaCPTS',
    ]

    posthoc_grid = Join(
        HP(method=methods),
    )

    glow = Join(
        HP(model='Glow'),
        HP(posthoc_grid=posthoc_grid)
    )

    return glow


def get_hparams_tuning(config):
    posthoc_grid_mqf2 = Union(
        Join(HP(method='M-CP'), HP(correction_factor=[0, 0.2, 0.4, 0.6, 0.8, 1])),
        Join(HP(method='C-HDR'), HP(n_samples=[5, 10, 30, 100, 300])),
        Join(HP(method='PCP'), HP(n_samples=[5, 10, 30, 100, 300])),
        Join(HP(method='HD-PCP'), HP(n_samples=[5, 10, 30, 100, 300])),
        Join(
            HP(method='C-PCP'), 
            Union(
                Join(HP(n_samples_mc=[5, 10, 30, 100, 300]), HP(n_samples_ref=[100])), 
                Join(HP(n_samples_mc=[100]), HP(n_samples_ref=[5, 10, 30, 300]))
            )
        ),
    )

    mqf2 = Join(
        HP(model='MQF2'),
        HP(posthoc_grid=posthoc_grid_mqf2)
    )

    return Union(
        mqf2,
    )


def get_larger_mqf2_tuning(config):
    default_methods = [
        'M-CP',
        'DR-CP',
        'C-HDR', 
        'HDR-H', 
        'PCP',
        'HD-PCP',
        'C-PCP',
    ]
    mqf2_methods = default_methods + [
        'L-CP',
        'L-H',
    ]

    # Posthoc grid
    posthoc_grid_mqf2 = Join(
        HP(method=mqf2_methods),
    )

    # MQF2
    mqf2 = Join(
        HP(model='MQF2'),
        HP(icnn_hidden_size=40),
        HP(icnn_num_layers=5),
        HP(estimate_logdet=[True, False]),
        HP(posthoc_grid=posthoc_grid_mqf2),
    )

    return Union(
        mqf2,
    )


def _get_tuning(config):
    if config.tuning_type == 'default':
        return get_default_tuning(config)
    elif config.tuning_type == 'larger_mqf2':
        return get_larger_mqf2_tuning(config)
    elif config.tuning_type == 'glow':
        return get_tuning_glow(config)
    elif config.tuning_type == 'hparams':
        return get_hparams_tuning(config)
    raise ValueError('Invalid tuning type')


def duplicates(choices):
    frozendict = lambda d: frozenset(d.items())
    frozen_choices = map(frozendict, choices)
    return [choice for choice, count in collections.Counter(frozen_choices).items() if count > 1]


def remove_duplicates(seq_of_dicts):
    seen = set()
    deduped_seq = []
    
    for d in seq_of_dicts:
        t = tuple(frozenset(d.items()))
        if t not in seen:
            seen.add(t)
            deduped_seq.append(d)
            
    return deduped_seq


def get_tuning(config):
    tuning = _get_tuning(config)
    tuning = remove_duplicates(tuning)
    dup = duplicates(tuning)
    assert len(dup) == 0, dup
    return tuning

