from moc.utils.general import filter_dict


camehl_datasets = [
    'households',
]

cevid_datasets = [
    'air',
    'births1',
    'births2',
    'wage',
]

del_barrio_datasets = [
    'ansur2',
    'calcofi',
]

feldman_datasets = [
    'bio',
    'blog_data',
    'house',
    'meps_19',
    'meps_20',
    'meps_21',
]

mulan_datasets = [
    'atp1d',
    'atp7d',
    'oes97',
    'oes10',
    'rf1',
    'rf2',
    'scm1d',
    'scm20d',
    'edm',
    'sf1',
    'sf2',
    'jura',
    'wq',
    'enb',
    'slump',
    #'andro', # Too small
    'osales',
    'scpf',
]

wang_datasets = [
    #'energy', # Same as enb in MULAN
    'taxi',
]

toy_2dim_datasets = [
    #'toy1',
    #'bimodal',
    'unimodal_heteroscedastic_power_0',
    'unimodal_heteroscedastic_power_0.1',
    'unimodal_heteroscedastic_power_0.2',
    'unimodal_heteroscedastic_power_0.5',
    'unimodal_heteroscedastic_power_1',
    'unimodal_heteroscedastic_power_2',
    'unimodal_heteroscedastic_power_5',
    'bimodal_heteroscedastic_power_0',
    'bimodal_heteroscedastic_power_0.1',
    'bimodal_heteroscedastic_power_0.2',
    'bimodal_heteroscedastic_power_0.5',
    'bimodal_heteroscedastic_power_1',
    'bimodal_heteroscedastic_power_2',
    'bimodal_heteroscedastic_power_5',
    'toy_hallin',
    'toy_del_barrio',
]

toy_ndim_datasets = [
    'mvn_isotropic_1',
    'mvn_isotropic_3',
    'mvn_isotropic_10',
    'mvn_isotropic_30',
    'mvn_isotropic_100',
    'mvn_diagonal_1',
    'mvn_diagonal_3',
    'mvn_diagonal_10',
    'mvn_diagonal_30',
    'mvn_diagonal_100',
    'mvn_mixture_1_10',
    'mvn_mixture_3_10',
    'mvn_mixture_10_10',
    'mvn_mixture_30_10',
    #'mvn_mixture_100_10',
    'mvn_dependent_1',
    'mvn_dependent_3',
    'mvn_dependent_10',
    'mvn_dependent_30',
    'mvn_dependent_100',
]

real_dataset_groups = {
    'camehl': camehl_datasets,
    'cevid': cevid_datasets,
    'del_barrio': del_barrio_datasets,
    'feldman': feldman_datasets,
    'mulan': mulan_datasets,
    'wang': wang_datasets,
}

toy_dataset_groups = {
    'toy_2dim': toy_2dim_datasets,
    'toy_ndim': toy_ndim_datasets,
}

subset_datasets = {
    'mulan': ['slump', 'sf1', 'jura', 'enb', 'scpf', 'scm20d', 'scm1d'],
    'cevid': ['births1'],
    'feldman': ['meps_19', 'bio'],
    'wang': ['taxi'],
}

def get_dataset_groups(key):
    if key == 'default':
        key = 'real'
    if key == 'all':
        return real_dataset_groups | toy_dataset_groups
    elif key == 'real':
        return real_dataset_groups
    elif key == 'toy':
        return toy_dataset_groups
    elif key == 'toy_2dim':
        return filter_dict(toy_dataset_groups, ['toy_2dim'])
    elif key == 'toy_ndim':
        return filter_dict(toy_dataset_groups, ['toy_ndim'])
    elif key == 'mulan':
        return filter_dict(real_dataset_groups, ['mulan'])
    elif key == 'cifar10':
        return {
            'cifar10': ['cifar10']
        }
    elif key == 'subset':
        return subset_datasets
    elif key == 'test':
        return {
            'mulan': ['sf1']
        }
    raise ValueError(f'Unknown dataset group {key}')
