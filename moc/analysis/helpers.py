main_metrics = ['coverage', 'median_region_size', 'region_size', 'cond_cov_x_error', 'cond_cov_z_error', 'wsc', 'test_coverage_time', 'score_time', 'total_time']
other_metrics = ['log_region_size', 'log_exact_region_size', 'q']

conformal_methods = ['M-CP', 'CopulaCPTS', 'DR-CP', 'C-HDR', 'PCP', 'HD-PCP', 'STDQR', 'C-PCP', 'L-CP']


def create_name_from_dict(d, config):
    name = d['posthoc_method']
    if config.name == 'hparams':
        if name == 'M-CP':
            correction_factor = d['posthoc_correction_factor']
            name = f'{name}-{correction_factor}'
        elif name in ['C-HDR', 'PCP', 'HD-PCP']:
            n_samples = d['posthoc_n_samples']
            name = f'{name}-{n_samples}'
        elif name == 'C-PCP':
            n_samples_mc = d['posthoc_n_samples_mc']
            n_samples_ref = d['posthoc_n_samples_ref']
            name = f'{name}-{n_samples_mc}'
            name = f'{name}-{n_samples_ref}'
    return name


def get_metric_name(metric):
    return {
        'coverage': 'Marginal coverage',
        'region_size': 'Mean Region Size',
        'log_region_size': 'G. Size',
        'cond_cov_x_error': 'CEC-$X$ (x100)',
        'cond_cov_z_error': 'CEC-$Z$ (x100)',
        'wsc': 'WSC',
        'score_time': 'Score Time',
        'test_coverage_time': 'Test coverage Time',
        'total_time': 'Time (s)',
        'log_exact_region_size': 'Geometric Mean Exact Region Size',
        'median_region_size': 'Median Region Size'
    }[metric]
