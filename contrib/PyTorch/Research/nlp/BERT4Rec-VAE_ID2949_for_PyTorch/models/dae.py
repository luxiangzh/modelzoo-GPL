# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAEModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        # Input dropout
        self.input_dropout = nn.Dropout(p=args.dae_dropout)

        # Construct a list of dimensions for the encoder and the decoder
        dims = [args.dae_hidden_dim] * 2 * args.dae_num_hidden
        dims = [args.num_items] + dims + [args.dae_latent_dim]

        # Stack encoders and decoders
        encoder_modules, decoder_modules = [], []
        for i in range(len(dims)//2):
            encoder_modules.append(nn.Linear(dims[2*i], dims[2*i+1]))
            decoder_modules.append(nn.Linear(dims[-2*i-1], dims[-2*i-2]))
        self.encoder = nn.ModuleList(encoder_modules)
        self.decoder = nn.ModuleList(decoder_modules)

        # Initialize weights
        self.encoder.apply(self.weight_init)
        self.decoder.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.normal_(0.0, 0.001)

    @classmethod
    def code(cls):
        return 'dae'

    def forward(self, x):
        x = F.normalize(x)
        x = self.input_dropout(x)
        
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            x = torch.tanh(x)
        
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i != len(self.decoder)-1:
                x = torch.tanh(x)

        return x

