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

def set_template(args):
    
    args.mode = 'train'

    args.dataset_code = 'ml-1m' 
    args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
    args.min_uc = 5
    args.min_sc = 0
    args.split = 'leave_one_out'

    args.dataloader_code = 'bert'
    batch = 128
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    args.train_negative_sampler_code = 'random'
    args.train_negative_sample_size = 0
    args.train_negative_sampling_seed = 0
    args.test_negative_sampler_code = 'random'
    args.test_negative_sample_size = 100
    args.test_negative_sampling_seed = 98765

    args.trainer_code = 'bert'
    args.device = 'npu'
    args.device_idx = '0'
    args.optimizer = 'Adam'
    args.lr = 0.001
    args.enable_lr_schedule = True
    args.decay_step = 25
    args.gamma = 1.0
    args.num_epochs=100
    args.metric_ks = [1, 5, 10, 20, 50, 100]
    args.best_metric = 'NDCG@10'

    args.model_code = 'bert'
    args.model_init_seed = 0

    args.bert_dropout = 0.1
    args.bert_hidden_units = 256
    args.bert_mask_prob = 0.15
    args.bert_max_len = 100
    args.bert_num_blocks = 2
    args.bert_num_heads = 4
    
    
