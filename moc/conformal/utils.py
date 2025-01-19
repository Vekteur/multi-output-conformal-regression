import torch


def get_samples(model, x, n_samples, cache={}, samples_key=None):
    """
    Helper function to get samples from the model or the cache if available.
    """
    samples = cache.get(samples_key)
    if samples is None:
        dist = model.predict(x)
        samples = dist.sample((n_samples,))
    else:
        samples = samples[:n_samples]
    return samples


def get_samples_and_log_probs(model, x, n_samples, cache={}, samples_key=None, log_probs_key=None):
    """
    Helper function to get samples with their log probabilities from the model or the cache if available.
    """
    samples = get_samples(model, x, n_samples, cache, samples_key)
    log_probs = cache.get(log_probs_key)
    if log_probs is None:
        dist = model.predict(x)
        log_probs = dist.log_prob(samples).detach()
    else:
        log_probs = log_probs[:n_samples]
    return samples, log_probs


def naive_empirical_cdf(a, b):
    """    
    Returns the empirical CDF of a at b.
    The first dimension of a represents the samples of the CDF.
    a is a tensor of shape (s, n).
    b is a tensor of shape (..., n).
    """
    assert a.dim() == 2 and b.dim() >= 1 and a.shape[-1] == b.shape[-1]
    b_shape = b.shape
    b = b.unsqueeze(b.dim() - 1)
    cdf = (a <= b).float().mean(dim=-2)
    assert cdf.shape == b_shape
    return cdf


def fast_empirical_cdf(a, b):
    """
    Returns the empirical CDF of a at b.
    The first dimension of a represents the samples of the CDF.
    a is a tensor of shape (s, n).
    b is a tensor of shape (..., n).
    This implementation is faster and requires less memory than naive_cdf.
    """
    assert a.dim() == 2 and b.dim() >= 1 and a.shape[-1] == b.shape[-1]
    b_shape = b.shape
    # We move n to the first dimension.
    # This will be useful because searchsorted has to be applied on the last dimension of a.
    a = a.movedim(-1, 0)
    b = b.movedim(-1, 0)

    if b.dim() == 1:
        # The naive implementation is faster for this case.
        cdf = (a <= b[:, None]).float().mean(dim=-1)
    else:
        a = torch.sort(a, dim=1)[0]
        # These operations are needed because the first N - 1 dimensions of a and b
        # have to be the same.
        view = (a.shape[0],) + (1,) * len(b.shape[1:-1]) + (a.shape[1],)
        repeat = (1,) + b.shape[1:-1] + (1,)
        a = a.view(*view).repeat(*repeat)
        cdf = torch.searchsorted(a, b.contiguous(), side='right') / a.shape[-1]
    
    # We move n back to the last dimension.
    cdf = cdf.movedim(0, -1)
    assert cdf.shape == b_shape
    return cdf


def to_latent_space(model, x, y):
    """
    Converts values from the output space to the latent space.
    This transformation is model-dependent.
    """
    from moc.models.mqf2.lightning_module import MQF2LightningModule
    from moc.models.glow.glow import GlowPreTrained

    if isinstance(model, MQF2LightningModule):
        z = model.model.flow.forward(y, x)
    elif isinstance(model, GlowPreTrained):
        front_shape = y.shape[:-1]
        y = y.reshape((front_shape.numel(),) + model.output_shape)
        z = model.model.inverse_and_log_det(y)[0]
    else:
        raise ValueError(f'Unsupported model type: {type(model)}')
    return z


def latent_distance(z):
    """
    Returns the distance from the origin in the latent space.
    """
    if isinstance(z, torch.Tensor): # Latent space of MQF2
        return torch.linalg.norm(z, dim=-1)
    elif isinstance(z, list): # Latent space of Glow
        s = []
        for zi in z:
            si = torch.linalg.norm(zi, dim=(-3, -2, -1))
            s.append(si)
        s = torch.stack(s, dim=-1)
        s = torch.max(s, dim=-1).values
        return s
    raise ValueError(f'Unsupported type: {type(z)}')


def to_output_space(model, x, z):
    from moc.models.mqf2.lightning_module import MQF2LightningModule

    if isinstance(model, MQF2LightningModule):
        pass


def distance_to_closest_point(points, y):
    """
    Returns the distance from `y` to the closest points in `points`.
    `points` is a tensor of shape (n_points, b, d).
    `y` is a tensor of shape (..., b, d), where the first dimensions are arbitrary 
    and will be evaluated for the same x.
    """
    n_points, b, d = points.shape
    *y_sample_shape, by, dy = y.shape
    assert (b, d) == (by, dy)
    y = y.unsqueeze(y.dim() - 2)
    norm = torch.linalg.norm(points - y, 2, dim=-1)
    assert norm.shape == tuple(y_sample_shape) + (n_points, b)
    distance = torch.min(norm, dim=-2).values
    assert distance.shape == tuple(y_sample_shape) + (b,)
    return distance


def pairwise_distance_to_closest_point(y):
    """
    Returns the distance from `y` to the closest points in `y`, except the point itself.
    `y` is a tensor of shape (n_points, ..., d).
    """
    n_points, *remaining, d = y.shape
    # Compute the pairwise distances
    norm = torch.linalg.norm(y.unsqueeze(0) - y.unsqueeze(1), 2, dim=-1)
    assert norm.shape == (n_points, n_points, *remaining)
    # Exclude the point itself
    mask = ~torch.eye(y.shape[0], dtype=bool)
    mask = mask[:, :, None].repeat(1, 1, *remaining)
    norm = norm[mask].reshape(n_points, n_points - 1, *remaining)
    # Compute the distance to the closest point
    distance = torch.min(norm, dim=-2).values
    assert distance.shape == (n_points, *remaining)
    return distance
