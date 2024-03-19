# ModelZoo-GPL

# 简介

ModelZoo-GPL，昇腾旗下的开源AI模型平台，涵盖计算机视觉、自然语言处理、语音、推荐、多模态、大语言模型等方向的AI模型及其基于昇腾机器实操案例。平台的每个模型都有详细的使用指导，为方便更多开发者使用ModelZoo-GPL，我们将持续增加典型网络和相关预训练模型。如果您有任何需求，请在[Gitee](https://gitee.com/ascend/modelzoo-GPL/issues)或[ModelZoo](https://bbs.huaweicloud.com/forum-726-1.html)提交issue，我们会及时处理。

# 目录

## PyTorch

|  一级目录  |  二级目录  |  说明  |
| ---------- | ----------- | ----------- |
| [built-in](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in) | [ACL_Pytorch](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch) <br> [AscendIE](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/AscendIE) <br> [PyTorch](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/PyTorch)| 基于昇腾芯片的推理模型参考 <br> 基于昇腾芯片的推理引擎模型参考 <br> 基于昇腾芯片的训练模型参考 |
| [contrib](https://gitee.com/ascend/modelzoo-GPL/tree/master/contrib) | [PyTorch](https://gitee.com/ascend/modelzoo-GPL/tree/master/contrib/PyTorch/) | 基于昇腾芯片的生态贡献训练模型参考 |


# 声明

本仓仅适用于GPL类许可证下的模型，请访问[modelzoo](https://gitee.com/ascend/modelzoo)获取其他的模型。


# 如何贡献

在开始贡献之前，请先阅读[notice](https://gitee.com/ascend/modelzoo/blob/master/contrib/CONTRIBUTING.md)。谢谢！
					
  
# 安全声明

## 运行用户建议

出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用。

## 文件权限控制

1. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
2. 建议用户对个人数据、商业资产、源文件、训练过程中保存的各类文件等敏感内容做好权限管控，管控权限可参考表1进行设置。

    表1 文件（夹）各场景权限管控推荐最大值

    | 类型           | linux权限参考最大值 |
    | -------------- | ---------------  |
    | 用户主目录                        |   750（rwxr-x---）            |
    | 程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）             |
    | 程序文件目录                      |   550（r-xr-x---）            |
    | 配置文件                          |  640（rw-r-----）             |
    | 配置文件目录                      |   750（rwxr-x---）            |
    | 日志文件(记录完毕或者已经归档)        |  440（r--r-----）             | 
    | 日志文件(正在记录)                |    640（rw-r-----）           |
    | 日志文件目录                      |   750（rwxr-x---）            |
    | Debug文件                         |  640（rw-r-----）         |
    | Debug文件目录                     |   750（rwxr-x---）  |
    | 临时文件目录                      |   750（rwxr-x---）   |
    | 维护升级文件目录                  |   770（rwxrwx---）    |
    | 业务数据文件                      |   640（rw-r-----）    |
    | 业务数据文件目录                  |   750（rwxr-x---）      |
    | 密钥组件、私钥、证书、密文文件目录    |  700（rwx—----）      |
    | 密钥组件、私钥、证书、加密密文        | 600（rw-------）      |
    | 加解密接口、加解密脚本            |   500（r-x------）        |


## 运行安全声明

建议用户结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。

## 公网地址声明

详见各模型目录下的public_address_statement.md


# 免责声明

### 致ModelZoo-GPL使用者
1. ModelZoo-GPL提供的模型仅供您用于非商业目的。
2. 对于各模型，ModelZoo-GPL平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用ModelZoo-GPL模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。

### 致数据集所有者
如果您不希望您的数据集在ModelZoo-GPL中的模型被提及，或希望更新ModelZoo-GPL中的模型关于您的数据集的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对ModelZoo-GPL的理解和贡献。

### License声明
Ascend ModelZoo-GPL提供的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以GPL-3.0许可证许可，对应许可证文本可查阅Ascend ModelZoo-GPL根目录。