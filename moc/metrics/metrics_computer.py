import logging
from collections import defaultdict

import torch

from .conditional_coverage_metrics import wsc_unbiased, ConditionalCoverageComputerForX, ConditionalCoverageComputerForZ
from .cache import Cache, EmptyCache
from moc.utils.general import elapsed_timer

log = logging.getLogger('moc')


def compute_coverage_indicator(conformalizer, alpha, x, y, cache={}):
    return conformalizer.is_in_region(x, y, alpha, cache).float()


def compute_cum_region_size(conformalizer, model, alpha, x, n_samples, cache_region_size={}, cache_test={}):
    samples = cache_region_size.get('samples')
    if samples is None:
        dist = model.predict(x)
        samples = dist.sample((n_samples,))
    log_probs = cache_region_size.get('log_probs')
    if log_probs is None:
        log_probs = dist.log_prob(samples).detach()
    terms = conformalizer.is_in_region(x, samples, alpha, cache_test).float() / torch.exp(log_probs)
    return torch.cumsum(terms, dim=0) / torch.arange(1, n_samples + 1, device=x.device)[:, None]


def compute_cum_log_region_size(conformalizer, model, alpha, x, n_samples=100, cache_region_size={}, cache_test={}):
    # Compute log region size for each instance in the batch and each sample size
    # between 0 and n_samples
    samples = cache_region_size.get('samples')
    if samples is None:
        dist = model.predict(x)
        samples = dist.sample((n_samples,))
    log_probs = cache_region_size.get('log_probs')
    if log_probs is None:
        log_probs = dist.log_prob(samples).detach()

    n_samples = samples.shape[0]
    assert samples.shape[:-1] == log_probs.shape
    
    # Use log-sum-exp trick
    max_logpdf = torch.max(log_probs, dim=0).values
    log_terms = torch.where(
        conformalizer.is_in_region(x, samples, alpha, cache_test),
        -log_probs + max_logpdf, 
        -torch.inf
    )
    
    log_cumsum = torch.logcumsumexp(log_terms, dim=0)
    log_means = log_cumsum - torch.log(torch.arange(1, n_samples + 1, device=x.device))[:, None]
    log_region_size = log_means - max_logpdf
    assert log_region_size.shape == (n_samples, x.shape[0],)
    return log_region_size


def compute_log_region_size(*args, **kwargs):
    return compute_cum_log_region_size(*args, **kwargs)[-1]


def compute_exact_log_region_size(conformalizer, alpha, x, cache_test={}):
    if not hasattr(conformalizer, 'get_exact_log_region_size'):
        return torch.full((x.shape[0],), torch.nan, device=x.device)
    return conformalizer.get_exact_log_region_size(x, alpha, cache_test)


class MetricsComputer:
    def __init__(self, model, alpha, datamodule, n_samples_for_region_size=100, nb_partitions=5, use_cache=True):
        self.model = model
        self.alpha = alpha
        self.datamodule = datamodule
        self.n_samples_for_region_size = n_samples_for_region_size
        # Check that the dataloader has not shuffling enabled
        assert type(datamodule.test_dataloader().sampler) == torch.utils.data.sampler.SequentialSampler
        # n_samples = 300
        # print('Warning: n_samples is hardcoded to 300')
        n_samples = 100
        if use_cache:
            log.info('Computing cache for calibration')
            self.cache_calib = Cache(model, datamodule.calib_dataloader(), n_samples, add_second_sample=True)
            log.info('Computing cache for test')
            self.cache_test = Cache(model, datamodule.test_dataloader(), n_samples, add_second_sample=True)
            log.info('Computing cache for region size')
            self.cache_region_size = Cache(model, datamodule.test_dataloader(), n_samples_for_region_size)
        else:
            self.cache_calib = EmptyCache(datamodule.calib_dataloader())
            self.cache_test = EmptyCache(datamodule.test_dataloader())
            self.cache_region_size = EmptyCache(datamodule.test_dataloader())

        self.cond_cov_x = ConditionalCoverageComputerForX(model, alpha, datamodule, nb_partitions=nb_partitions)
        if self.model.output_type() == 'distribution':
            log.info('Computing partition for Z')
            self.cond_cov_z = ConditionalCoverageComputerForZ(model, alpha, datamodule, nb_partitions=nb_partitions, cache_test=self.cache_test)
    
    def compute_metrics_on_batch(self, conformalizer, x, y, cache_test={}, cache_region_size={}):
        with elapsed_timer() as timer:
            coverage_indicator = compute_coverage_indicator(conformalizer, self.alpha, x, y, cache=cache_test)
        test_coverage_time = timer()
        if self.model.output_type() == 'distribution':
            log_region_size = compute_log_region_size(
                conformalizer, 
                self.model, 
                self.alpha, 
                x, 
                self.n_samples_for_region_size,
                cache_region_size=cache_region_size,
                cache_test=cache_test,
            )
            log_region_size = torch.maximum(log_region_size, torch.tensor(-1e30))  # Avoid -inf
            region_size = log_region_size.exp()
        else:
            log_region_size = torch.full((x.shape[0],), torch.nan, device=x.device)
            region_size = torch.full((x.shape[0],), torch.nan, device=x.device)
        log_exact_region_size = compute_exact_log_region_size(conformalizer, self.alpha, x, cache_test=cache_test)
        log_exact_region_size = torch.maximum(log_exact_region_size, torch.tensor(-1e30))  # Avoid -inf
        exact_region_size = log_exact_region_size.exp()
        
        return {
            'coverage': coverage_indicator,
            'log_region_size': log_region_size,
            'region_size': region_size,
            'log_exact_region_size': log_exact_region_size,
            'exact_region_size': exact_region_size,
        }, test_coverage_time
    
    def compute_conditional_coverage(self, coverage):
        metrics = {}

        x = torch.cat([x for x, y in self.datamodule.test_dataloader()], dim=0)
        metrics['wsc'] = wsc_unbiased(x.numpy(), coverage.numpy(), delta=0.2).item()

        cond_coverages = self.cond_cov_x.compute_cond_coverages(coverage)
        metrics['cond_cov_x_error'] = self.cond_cov_x.compute_error(cond_coverages).item()

        if self.model.output_type() == 'distribution':
            cond_coverages = self.cond_cov_z.compute_cond_coverages(coverage)
            metrics['cond_cov_z_error'] = self.cond_cov_z.compute_error(cond_coverages).item()
        
        return metrics
    
    def compute_all_metrics(self, conformalizer):
        metrics_to_average = defaultdict(list)
        test_coverage_time = 0
        for (x, y), cache_test, cache_region_size in zip(self.datamodule.test_dataloader(), self.cache_test, self.cache_region_size):
            x, y = x.to(self.model.device), y.to(self.model.device)
            metrics_on_batch, test_coverage_time_on_batch = self.compute_metrics_on_batch(conformalizer, x, y, cache_test, cache_region_size)
            for name, values in metrics_on_batch.items():
                metrics_to_average[name].append(values)
            test_coverage_time += test_coverage_time_on_batch
        metrics_to_average = {
            name: torch.cat(values).float().cpu() for name, values in metrics_to_average.items()
        }
        metrics = {name: values.mean().item() for name, values in metrics_to_average.items()}
        if self.model.output_type() == 'distribution':
            metrics['median_region_size'] = metrics_to_average['region_size'].median().item()

        coverage = metrics_to_average['coverage']
        metrics.update(self.compute_conditional_coverage(coverage))

        return metrics, test_coverage_time

    def compute_metrics(self, conformalizer):
        metrics, test_coverage_time = self.compute_all_metrics(conformalizer)
        metrics['test_coverage_time'] = test_coverage_time
        metrics['test_coverage_time'] += self.cache_test.get_time(conformalizer.used_cache_keys())
        metrics['q'] = conformalizer.get_q(self.alpha).detach().cpu().numpy()
        return metrics
