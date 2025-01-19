import logging

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

log = logging.getLogger('moc')

# wsc is adapted from https://github.com/msesia/chr/blob/master/chr/coverage.py

def wsc(reprs, coverages, delta, M=1000):
    def wsc_v(reprs, cover, delta, v):
        n = len(cover)
        z = np.dot(reprs, v)
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0 - delta) * n))
        ai_best, bi_best = 0, n - 1
        cover_min = cover.mean()
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
            coverage[np.arange(0, bi_min - ai)] = 1
            bi_star = ai + np.argmin(coverage)
            cover_star = coverage[bi_star - ai]
            if cover_star < cover_min:
                ai_best, bi_best = ai, bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def wsc_v_fully_vectorized(reprs, cover, delta, v):
        # Ensure inputs are numpy arrays
        reprs = np.atleast_2d(reprs)
        cover = np.atleast_1d(cover)
        v = np.atleast_1d(v)

        # Calculate z and sort
        z = np.dot(reprs, v)
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]

        # Calculate parameters
        n = len(cover)
        ai_max = int(np.round((1.0 - delta) * n))
        delta_n = int(np.round(delta * n))

        # Prepare arrays for vectorized operations
        ai_range = np.arange(ai_max)
        bi_range = np.arange(n)
        cumsum_cover = np.cumsum(cover_ordered)

        # Create meshgrid for all possible (ai, bi) pairs
        ai_mesh, bi_mesh = np.meshgrid(ai_range, bi_range, indexing='ij')

        # Calculate coverage for all pairs
        denominator = bi_mesh - ai_mesh + 1
        numerator = cumsum_cover[bi_mesh] - np.where(ai_mesh > 0, cumsum_cover[ai_mesh - 1], 0)
        
        coverage = np.full_like(denominator, np.inf, dtype=float)
        valid_mask = (bi_mesh >= ai_mesh) & (denominator > 0)
        coverage[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

        # Set coverage to 1 for invalid pairs (bi < ai + delta_n)
        coverage[bi_mesh < ai_mesh + delta_n] = 1

        # Find the minimum coverage and corresponding indices
        min_coverage = np.min(coverage)
        ai_best, bi_best = np.unravel_index(np.argmin(coverage), coverage.shape)

        return min_coverage, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = np.random.randn(p, n)
        v /= np.linalg.norm(v, axis=0)
        return v.T

    def sample_from_normal_approximation(n, X):
        mean = np.mean(X, axis=0)
        covariance = np.cov(X, rowvar=False)
        return np.random.multivariate_normal(mean, covariance, size=n)

    V = sample_sphere(M, reprs.shape[1])
    # V = sample_from_normal_approximation(M, reprs)
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    for m in range(M):
        wsc_list[m], a_list[m], b_list[m] = wsc_v_fully_vectorized(reprs, coverages, delta, V[m])

    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star


def wsc_unbiased(reprs, coverages, delta, M=1000, test_size=0.75, random_state=0):
    def wsc_vab(reprs, cover, v, a, b):
        n = len(reprs)
        z = np.dot(reprs, v)
        idx = np.where((a <= z) * (z <= b))
        coverage = np.mean(cover[idx])
        return coverage

    (
        reprs_train,
        reprs_test,
        coverages_train,
        coverages_test,
    ) = train_test_split(reprs, coverages, test_size=test_size, random_state=random_state)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(reprs_train, coverages_train, delta=delta, M=M)
    # print(wsc_star, v_star, a_star, b_star)
    # Estimate coverage
    coverage = wsc_vab(reprs_test, coverages_test, v_star, a_star, b_star)
    return coverage


class ConditionalCoverageComputer:
    def __init__(self, model, alpha, datamodule, nb_partitions, cache_val=None, cache_test=None):
        self.model = model
        self.alpha = alpha
        self.kmeans = self.compute_partition(datamodule.val_dataloader(), nb_partitions, cache=cache_val)
        self.test_partition = self.get_partition(datamodule.test_dataloader(), cache=cache_test)

    def get_partition_features(self, dl, cache=None):
        pass

    def compute_partition(self, dl_val, nb_partitions, cache=None):
        features = self.get_partition_features(dl_val, cache=cache)
        return KMeans(n_clusters=nb_partitions, n_init='auto').fit(features)
    
    def get_partition(self, dl_test, cache=None):
        features = self.get_partition_features(dl_test, cache=cache)
        return self.kmeans.predict(features)

    def compute_cond_coverages(self, coverages):
        cond_coverage_list = []
        for i in range(self.kmeans.n_clusters):
            cond_coverages = coverages[self.test_partition == i]
            if len(cond_coverages) == 0:   # Special case: no sample in the partition
                cond_coverage = 0.5
            else:
                cond_coverage = cond_coverages.mean()
            cond_coverage_list.append(cond_coverage)
        return torch.tensor(cond_coverage_list)

    def compute_error(self, cond_coverages):
        labels = self.kmeans.labels_
        weights = np.bincount(labels, minlength=self.kmeans.n_clusters) / len(labels)
        error = ((cond_coverages - (1 - self.alpha)).square() * weights).sum()
        return error


class ConditionalCoverageComputerForX(ConditionalCoverageComputer):
    def get_partition_features(self, dl, cache=None):
        return torch.cat([x for x, y in dl], dim=0)
    

class ConditionalCoverageComputerForZ(ConditionalCoverageComputer):
    def get_partition_features(self, dl, n_samples=100, cache=None):
        logz_list = []
        for batch_idx, (x, y) in enumerate(dl):
            x, y = x.to(self.model.device), y.to(self.model.device)
            logz_i = None
            if cache is not None:
                logz_i = cache[batch_idx].get('log_probs')[:n_samples]
            if logz_i is None:
                dist = self.model.predict(x)
                sample = dist.sample((n_samples,))
                logz_i = dist.log_prob(sample).detach()
            logz_list.append(logz_i.cpu())
        logz = torch.cat(logz_list, dim=1)
        logz = logz.permute(1, 0)
        assert logz.shape[1] == n_samples, logz.shape

        n_nan = torch.isnan(logz).sum().item()
        if n_nan > 0:
            log.info(f'Warning: There are {n_nan} NaNs in the tensor')
        n_inf = torch.isinf(logz).sum().item()
        if n_inf > 0:
            log.info(f'There are {n_inf} infinite values in the tensor')
        n_big = torch.abs(logz) > 1e100
        if n_big.sum().item() > 0:
            log.info(f'There are {n_big.sum().item()} values with magnitude greater than 1e30 in the tensor')
        
        # Replace NaNs and infinite values with 0
        logz[torch.isnan(logz)] = 0
        logz[torch.isinf(logz)] = 0

        return logz
