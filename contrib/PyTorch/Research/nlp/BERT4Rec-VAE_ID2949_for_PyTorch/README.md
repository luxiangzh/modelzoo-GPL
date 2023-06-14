Sequential Recommendation with Bidirectional Encoder Representations from Transformer

发布者（Publisher）：王媛媛

应用领域（Application Domain）：NLP

版本（Version）：1

修改时间（Modified） ：2022.8.29

框架（Framework）：Pytorch 1.3

处理器（Processor）：昇腾910

应用级别（Categories）：Official

描述（Description）：基于Bert的推荐系统

# Introduction

This repository implements models from the following paper:

> **BERT4Rec: Sequential Recommendation with BERT (Sun et al.)**  

and lets you train them on MovieLens-1m and MovieLens-20m.

#requirements

wget==3.2
tqdm==4.36.1
numpy==1.16.2
torch==1.3.0
tb-nightly==2.1.0a20191121
pandas==0.25.0
scipy==1.3.2
future==0.18.2

# Usage

Train BERT4Rec on ML-1m and run test set inference after training

   ```bash
   printf '1\ny\n' | python main.py --template train_bert
   ```
# 公网地址说明

代码涉及公网地址参考 public_address_statement.md



  

