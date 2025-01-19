from abc import abstractmethod
from moc.metrics.cache import EmptyCache

import torch


class RegionPredictorBase:
    def __init__(self, dl_calib, model, cache_calib=None, rc=None):
        self.dl_calib = dl_calib
        self.model = model
        self.rc = rc
    
    def get_q(self, alpha):
        """
        Returns the quantile of the score distribution at level 1 - alpha, or nan if not applicable.
        """
        return torch.tensor(torch.nan, device=self.model.device)
    
    @abstractmethod
    def is_in_region(self, x, y, alpha, cache={}):
        pass

    def used_cache_keys(self):
        """
        This method indicates which keys of the cache are used by the conformalizer. This is useful
        to determine the time required to compute the score without the speedup from caching.
        """
        return []


def conformal_quantile(scores, alpha):
    n = scores.shape[0]
    scores = torch.cat([scores, torch.full((1,) + scores.shape[1:], torch.inf, device=scores.device)], dim=0)
    level = torch.tensor((1 - alpha) * (n + 1)) / (n + 1)
    level = level.type(scores.dtype).to(scores.device)
    return torch.quantile(scores, level, interpolation='higher', dim=0)


class ConformalizerBase(RegionPredictorBase):
    """
    Conformalizer base class.
    To match the interface of ConformalizerBaseAlphaInvariant, alpha is only provided to `is_in_region`.
    This requires caching the computed conformal quantiles in `q_dict` for efficiency.
    """
    def __init__(self, dl_calib, model, cache_calib=None, **kwargs):
        super().__init__(dl_calib, model, cache_calib, **kwargs)
        self.cache_calib = EmptyCache(dl_calib) if cache_calib is None else cache_calib
        self.scores_dict = {}

    @abstractmethod
    def get_score(self, x, y, alpha, cache={}):
        pass

    def get_calib_scores(self, alpha):
        if alpha not in self.scores_dict:
            scores = [
                self.get_score(x.to(self.model.device), y.to(self.model.device), alpha, cache_cal)
                for (x, y), cache_cal in zip(self.dl_calib, self.cache_calib)
            ]
            scores = torch.concat(scores, dim=0)
            self.scores_dict[alpha] = scores
        return self.scores_dict[alpha]
        
    def get_q(self, alpha):
        return conformal_quantile(self.get_calib_scores(alpha), alpha)

    def is_in_region(self, x, y, alpha, cache={}):
        return self.get_score(x, y, alpha, cache) <= self.get_q(alpha)


class ConformalizerBaseAlphaInvariant(RegionPredictorBase):
    """
    Conformalizer base class where the scores do not depend on alpha.
    This is useful e.g. to avoid recomputing the scores on toy graphics with different alpha.
    """
    def __init__(self, dl_calib, model, cache_calib=None, **kwargs):
        super().__init__(dl_calib, model, **kwargs)
        cache_calib = EmptyCache(dl_calib) if cache_calib is None else cache_calib
        self.calib_scores = [
            self.get_score(x.to(model.device), y.to(model.device), cache)
            for (x, y), cache in zip(dl_calib, cache_calib)
        ]
        self.calib_scores = torch.concat(self.calib_scores)

    @abstractmethod
    def get_score(self, x, y, cache={}):
        pass

    def get_q(self, alpha):
        return conformal_quantile(self.calib_scores, alpha)

    def is_in_region(self, x, y, alpha, cache={}):
        return self.get_score(x, y, cache) <= self.get_q(alpha)
    