import sys
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from torch.utils.data.dataloader import _utils


class _MultiProcessingDataLoaderIterFixed(_MultiProcessingDataLoaderIter):

    def _shutdown_workers(self):
        if _utils is None or _utils.python_exit_status is True or _utils.python_exit_status is None:
            return
        super()._shutdown_workers()


for k, v in sys.modules.items():
    if 'torch.utils.data' in k and hasattr(v,
                                           '_MultiProcessingDataLoaderIter'):
        setattr(v, '_MultiProcessingDataLoaderIter',
                _MultiProcessingDataLoaderIterFixed)
