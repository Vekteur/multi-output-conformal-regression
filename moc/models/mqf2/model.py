# Adapted from https://github.com/awslabs/gluonts/tree/dev/src/gluonts/torch/model/mqf2
# Copyright (c) 2021 Chin-Wei Huang

from typing import Optional, Tuple, List, Union

from cpflows.flows import SequentialFlow, DeepConvexFlow, ActNorm

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal


class MyDeepConvexFlow(DeepConvexFlow):
    def __init__(
        self,
        picnn: torch.nn.Module,
        dim: int,
        is_energy_score: bool = False,
        estimate_logdet: bool = False,
        m1: int = 10,
        m2: Optional[int] = None,
        rtol: float = 0.0,
        atol: float = 1e-3,
    ) -> None:
        super().__init__(
            picnn,
            dim,
            m1=m1,
            m2=m2,
            rtol=rtol,
            atol=atol,
        )

        self.picnn = self.icnn
        self.is_energy_score = is_energy_score
        self.estimate_logdet = estimate_logdet

    def get_potential(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        n = x.size(0)
        output = self.picnn(x, context)

        return (
            F.softplus(self.w1) * output
            + F.softplus(self.w0)
            * (x.view(n, -1) ** 2).sum(1, keepdim=True)
            / 2
        )

    def forward_transform(
        self,
        x: torch.Tensor,
        logdet: Optional[Union[float, torch.Tensor]] = 0.0,
        context: Optional[torch.Tensor] = None,
        extra: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.estimate_logdet:
            return self.forward_transform_stochastic(
                x, logdet, context=context, extra=extra
            )
        else:
            return self.forward_transform_bruteforce(
                x, logdet, context=context
            )


class MySequentialFlow(SequentialFlow):
    def __init__(self, flows: List[torch.nn.Module], dim: int) -> None:
        super().__init__(flows)
        self.dim = dim

    def forward(
        self, y: torch.Tensor, x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        sample_shape = y.shape[:-(len(batch_shape) + 1)]
        x_repeat = x.view((1,) * len(sample_shape) + x.shape).expand(sample_shape + x.shape)
        y_flat, x_repeat_flat = y.reshape(-1, y.shape[-1]), x_repeat.reshape(-1, x_repeat.shape[-1])

        for flow in self.flows:
            if isinstance(flow, MyDeepConvexFlow):
                y_flat = flow.forward(y_flat, context=x_repeat_flat)
            elif isinstance(flow, ActNorm):
                y_flat = flow.forward_transform(y_flat)[0]
        
        return y_flat.reshape(sample_shape + batch_shape + (self.dim,))


class ReverseSequentialFlow(MySequentialFlow):
    def forward_transform(self, *args, **kwargs):
        return super().reverse(*args, **kwargs)

    def reverse(self, *args, **kwargs):
        return super().forward_transform(*args, **kwargs)
