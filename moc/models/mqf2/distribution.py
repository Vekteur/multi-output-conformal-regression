# Adapted from https://github.com/awslabs/gluonts/tree/dev/src/gluonts/torch/model/mqf2

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch


class MQF2Distribution(torch.distributions.Distribution):
    def __init__(
        self,
        model,
        x: torch.Tensor,
        threshold_input: float = 100.0,
        is_energy_score: bool = False,
        validate_args: bool = False,
    ) -> None:
        self.model = model
        self.x = x
        self.threshold_input = threshold_input
        self.is_energy_score = is_energy_score

        super().__init__(
            batch_shape=self.batch_shape, validate_args=validate_args
        )

    def log_prob(self, y):
        return self.model.log_prob(y, self.x)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        z = self.model.latent_dist(self.batch_shape).sample(sample_shape)
        return self.model.forward(z, self.x)

    @property
    def batch_shape(self) -> torch.Size:
        # last dimension is the hidden state size
        return self.x.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size((self.flow.dim,))
