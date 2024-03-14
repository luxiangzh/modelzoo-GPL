# 欢迎使用Ascend ModelZoo-GPL

为方便更多开发者使用Ascend ModelZoo，我们将持续增加典型网络和相关预训练模型。如果您有任何需求，请在[Gitee](https://gitee.com/ascend/modelzoo-GPL/issues)或[ModelZoo](https://bbs.huaweicloud.com/forum-726-1.html)提交issue，我们会及时处理。


## 声明

本仓仅适用于GPL类许可证下的模型，请访问[modelzoo](https://gitee.com/ascend/modelzoo)获取其他的模型。


## 如何贡献

在开始贡献之前，请先阅读[notice](https://gitee.com/ascend/modelzoo/blob/master/contrib/CONTRIBUTING.md)。谢谢！


## 目录

### PyTorch

| 目录                                                         | 说明                       |
| ------------------------------------------------------------ | -------------------------- |
| [built-in](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in) | 规范模型 |
| [contrib](https://gitee.com/ascend/modelzoo-GPL/tree/master/contrib) | 生态贡献模型 |
					
  
## 安全声明

### 运行用户建议

出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用。

### 文件权限控制

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


### 运行安全声明

1. 建议用户结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。


### 公网地址声明

详见各模型目录下的public_address_statement.md


## 免责声明

Ascend ModelZoo仅提供公共数据集下载和预处理脚本。这些数据集不属于ModelZoo，ModelZoo也不对其质量或维护负责。请确保您具有这些数据集的使用许可。基于这些数据集训练的模型仅可用于非商业研究和教育。

致数据集所有者：

如果您不希望您的数据集公布在ModelZoo上或希望更新ModelZoo中属于您的数据集，请在Github/Gitee提交issue，我们将根据您的issue删除或更新您的数据集。衷心感谢您对我们社区的理解和贡献。

Ascend ModelZoo-GPL的license是GPL-3.0具体内容，请参见LICENSE文件。

