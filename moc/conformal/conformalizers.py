import torch
from torch.utils.data import DataLoader, random_split

from scipy.stats import chi2

from moc.metrics.distribution_metrics import nll, kernel_score_from_samples
from .base_conformalizer import RegionPredictorBase, ConformalizerBase, ConformalizerBaseAlphaInvariant
from .utils import (
    get_samples,
    get_samples_and_log_probs, 
    fast_empirical_cdf, 
    to_latent_space, 
    latent_distance,
    distance_to_closest_point, 
    pairwise_distance_to_closest_point, 
)


class M_CP(ConformalizerBase):
    def __init__(self, dl_calib, model, n_samples=100, correction_factor=0, **kwargs):
        self.n_samples = n_samples
        _, first_y = next(iter(dl_calib))
        self.d = first_y.shape[-1]
        self.correction_factor = correction_factor
        super().__init__(dl_calib, model, **kwargs)

    def get_region_bounds(self, x, alpha, cache={}):
        samples = get_samples(self.model, x, self.n_samples, cache, 'samples')
        # Changing the correction factor from 0 to 1 allows to change the targeted coverage from 1 - alpha to 1 - alpha / d.
        alpha /= (1 + self.correction_factor * (self.d - 1))
        return torch.quantile(samples, torch.tensor([alpha / 2, 1 - alpha / 2], device=x.device), dim=0)

    def get_score(self, x, y, alpha, cache={}):
        ql, qh = self.get_region_bounds(x, alpha, cache)
        cqr_score = torch.maximum(ql - y, y - qh)
        assert cqr_score.shape == y.shape
        simultaneous_score = torch.max(cqr_score, dim=-1).values
        return simultaneous_score
    
    def get_exact_log_region_size(self, x, alpha, cache={}):
        ql, qh = self.get_region_bounds(x, alpha, cache)
        q = self.get_q(alpha)
        ql, qh = ql - q, qh + q
        return torch.maximum(qh - ql, torch.tensor(1e-50, device=x.device)).log().sum(dim=-1)
    
    def used_cache_keys(self):
        return ['samples']


class DR_CP(ConformalizerBaseAlphaInvariant):
    def get_score(self, x, y, cache={}):
        return nll(self.model, x, y)


def hpd(model, x, y, n_samples, cache={}):
    """
    Returns the highest predictive density region of y conditionally to x.
    x is a tensor of shape (b, d_x).
    y is a tensor of shape (..., b, d_y), where the first dimensions are arbitrary 
    and will be evaluated for the same x.
    """
    y_sample_shape = y.shape[:-2]
    assert x.dim() == 2 and x.shape[-2] == y.shape[-2]
    dist = model.predict(x)
    samples, log_probs_samples = get_samples_and_log_probs(
        model, 
        x, 
        n_samples, 
        cache, 
        'samples',
        'log_probs',
    )
    assert samples.shape == (n_samples, x.shape[0], y.shape[-1])
    assert log_probs_samples.shape == (n_samples, x.shape[0])
    log_probs_y = dist.log_prob(y).detach()
    assert log_probs_y.shape == y_sample_shape + (x.shape[0],)
    cdf = fast_empirical_cdf(log_probs_samples, log_probs_y)
    assert cdf.shape == y_sample_shape + (x.shape[0],)
    hpd = 1 - cdf
    return hpd


class HDR_CP(ConformalizerBaseAlphaInvariant):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        self.n_samples = n_samples
        super().__init__(dl_calib, model, **kwargs)

    def get_score(self, x, y, cache={}):
        return hpd(self.model, x, y, n_samples=self.n_samples, cache=cache)
    
    def used_cache_keys(self):
        return ['samples', 'log_probs']


class HDR_H(RegionPredictorBase):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        self.n_samples = n_samples
        super().__init__(dl_calib, model, **kwargs)
    
    def is_in_region(self, x, y, alpha, cache={}):
        return hpd(self.model, x, y, n_samples=self.n_samples, cache=cache) <= 1 - alpha
    
    def used_cache_keys(self):
        return ['samples', 'log_probs']


class L_CP(ConformalizerBaseAlphaInvariant):
    def get_score(self, x, y, cache={}):
        return latent_distance(to_latent_space(self.model, x, y))


class L_H(RegionPredictorBase):
    def is_in_region(self, x, y, alpha, cache={}):
        return latent_distance(to_latent_space(self.model, x, y)) <= chi2(y.shape[-1]).ppf(1 - alpha)


class PCP(ConformalizerBaseAlphaInvariant):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        self.n_samples = n_samples
        super().__init__(dl_calib, model, **kwargs)
    
    def get_score(self, x, y, cache={}):
        samples = get_samples(self.model, x, self.n_samples, cache, 'samples2')
        assert samples.shape == (self.n_samples, x.shape[0], y.shape[-1])
        return distance_to_closest_point(samples, y)
    
    def used_cache_keys(self):
        return ['samples2']


class HD_PCP(ConformalizerBase):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        self.n_samples = n_samples
        super().__init__(dl_calib, model, **kwargs)
    
    def get_score(self, x, y, alpha, cache={}):
        samples, log_probs = get_samples_and_log_probs(
            self.model, 
            x, 
            self.n_samples, 
            cache, 
            'samples2',
            'log_probs2',
        )
        assert samples.shape == (self.n_samples, x.shape[0], y.shape[-1])
        assert log_probs.shape == (self.n_samples, x.shape[0])
        # Select the k samples with the highest log probability
        k = torch.tensor((1 - alpha) * (self.n_samples)).int()
        indices = torch.argsort(log_probs, dim=0, descending=True)[:k]
        assert indices.shape == (k, x.shape[0])
        samples = samples[indices, torch.arange(x.shape[0])]
        assert samples.shape == (k, x.shape[0], y.shape[-1])
        # Compute the distance to the closest point
        return distance_to_closest_point(samples, y)
    
    def used_cache_keys(self):
        return ['samples2', 'log_probs2']


class CP2_PCP_Linear(ConformalizerBase):
    """
    This class implements the method of Vincent Plassier et al with a linear adjustment function.
    """
    def __init__(self, dl_calib, model, n_samples_mc=100, n_samples_ref=100, **kwargs):
        self.n_samples_mc = n_samples_mc
        self.n_samples_ref = n_samples_ref
        super().__init__(dl_calib, model, **kwargs)
    
    def get_score(self, x, y, alpha, cache={}):
        y_sample_shape = y.shape[:-2]
        samples_mc = get_samples(self.model, x, self.n_samples_mc, cache, 'samples')
        samples_ref = get_samples(self.model, x, self.n_samples_ref, cache, 'samples2')

        min_norm_y = distance_to_closest_point(samples_ref, y)
        assert min_norm_y.shape == y_sample_shape + (x.shape[0],)

        norm_samples = torch.linalg.norm(samples_mc[:, None, :, :] - samples_ref[None, :, :, :], 2, dim=-1)
        min_norm_samples = torch.min(norm_samples, dim=-2).values
        assert min_norm_samples.shape == (self.n_samples_mc, x.shape[0])
        min_norm_samples_quantile = torch.quantile(min_norm_samples, 1 - alpha, dim=0)
        assert min_norm_samples_quantile.shape == (x.shape[0],)

        score = min_norm_y / min_norm_samples_quantile
        assert score.shape == y_sample_shape + (x.shape[0],)
        return score
    
    def used_cache_keys(self):
        return ['samples', 'samples2']


class CDF_Based(ConformalizerBaseAlphaInvariant):
    def __init__(self, dl_calib, model, base_conformalizer, n_samples_mc=100, **kwargs):
        self.n_samples_mc = n_samples_mc
        self.base_conformalizer = base_conformalizer
        super().__init__(dl_calib, model, **kwargs)
    
    def get_score(self, x, y, cache={}):
        y_sample_shape = y.shape[:-2]

        score_y = self.base_conformalizer.get_score(x, y, cache)
        samples_mc = get_samples(self.model, x, self.n_samples_mc, cache, 'samples')
        assert samples_mc.shape == (self.n_samples_mc, x.shape[0], y.shape[-1])
        score_samples = self.base_conformalizer.get_score(x, samples_mc, cache)

        score = fast_empirical_cdf(score_samples, score_y)
        assert score.shape == y_sample_shape + (x.shape[0],)
        return score
    
    def used_cache_keys(self):
        return {'samples'} | set(self.base_conformalizer.used_cache_keys())


class C_PCP(CDF_Based):
    def __init__(self, dl_calib, model, n_samples_mc=100, n_samples_ref=100, **kwargs):
        base_conformalizer = PCP(dl_calib, model, n_samples_ref, **kwargs)
        super().__init__(dl_calib, model, base_conformalizer, n_samples_mc, **kwargs)
