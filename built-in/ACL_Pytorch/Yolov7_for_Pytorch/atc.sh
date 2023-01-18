# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

atc --framework=5 --model=$1 --output=$2 --input_format=NCHW --input_shape="images:"$3", 3, 640, 640" --log=error \
--soc_version=$4 --insert_op_conf=aipp.cfg --optypelist_for_implmode="Sigmoid" --op_select_implmode=high_performance


