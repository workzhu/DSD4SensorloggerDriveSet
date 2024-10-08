import os
import torch
from models import Autoformer, Transformer, TimesNet, TimesNetMod, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, Koopa, LSTM, \
    TimesNetFCN, LSTM2, TimesNet3, CNNLSTM, FCN, CNN, GRU, LSTMFCN


# exp基类
class Exp_Basic(object):

    # 初始化
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model_dict = {
            'TimesNetMod': TimesNetMod,
            'TimesNet': TimesNet,
            'TimesNetFCN': TimesNetFCN,
            'TimesNet3': TimesNet3,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'LSTM': LSTM,
            'LSTM2': LSTM2,
            'CNNLSTM': CNNLSTM,
            'FCN': FCN,
            'CNN': CNN,
            'GRU': GRU,
            'LSTMFCN': LSTMFCN
        }

        self.dataset, self.train_dataset, self.valid_dataset, self.test_dataset, self.train_loader, self.valid_loader, \
            self.test_loader = self._get_data()

        # print('train_data windows', self.train_data.windows)
        # print('train_data samples', self.test_data.samples)
        # 转去exp_classification的_build_model()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
