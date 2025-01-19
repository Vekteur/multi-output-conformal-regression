from collections import defaultdict

from moc.utils.general import elapsed_timer


class Cache:
    def __init__(self, model, dataloader, n_samples, add_second_sample=False):
        self.cache = {}
        self.time = {}
        cache, time = self.compute_cache(dataloader, model, n_samples)
        self.cache.update(cache)
        self.time.update(time)
        if add_second_sample:
            cache, time = self.compute_cache(dataloader, model, n_samples)
            self.cache['samples2'] = cache['samples']
            self.cache['log_probs2'] = cache['log_probs']
            self.time['samples2'] = time['samples']
            self.time['log_probs2'] = time['log_probs']

    def compute_cache(self, dl, model, n_samples):
        cache = defaultdict(list)
        time = defaultdict(float)

        for x, y in dl:
            x, y = x.to(model.device), y.to(model.device)
            with elapsed_timer() as timer:
                dist = model.predict(x)
                sample = dist.sample((n_samples,))
            cache['samples'].append(sample)
            time['samples'] += timer()
            d = sample.shape[2]
            assert sample.shape == (n_samples, x.shape[0], d)
            with elapsed_timer() as timer:
                log_prob = dist.log_prob(sample).detach()
            cache['log_probs'].append(log_prob)
            time['log_probs'] += timer()
            assert log_prob.shape == (n_samples, x.shape[0])
        return cache, time
    
    def __getitem__(self, batch_idx):
        return {
            key: self.cache[key][batch_idx]
            for key in self.cache
        }
    
    def __len__(self):
        return len(self.cache['samples'])
    
    def get_time(self, keys):
        return sum(self.time[key] if key in self.time else 0 for key in keys)


class EmptyCache:
    def __init__(self, dataloader):
        self.len = len(dataloader)

    def __getitem__(self, key):
        return {}
    
    def __len__(self):
        return self.len

    def get_time(self, keys):
        return 0
