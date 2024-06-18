import numpy as np
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from .model import ResNet18

# set manual seed for reproducibility
seed = 1234
# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, num_layer="2_layer"):
        super().__init__()
        self.in_features = dim
        if num_layer == "1_layer":
            self.net = nn.Sequential(
                nn.Linear(dim, projection_size),
            )
        elif num_layer == "2_layer":
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size),
            )
        else:
            raise NotImplementedError(f"Not defined MLP: {num_layer}")

    def forward(self, x):
        return self.net(x)

class BYOL_Client(nn.Module):
    def __init__(
        self,
        net=ResNet18(),
        image_size=32,
        projection_size=2048,
        projection_hidden_size=4096,
        moving_average_decay=0.99,
        stop_gradient=True,
        has_predictor=True,
        predictor_network="2_layer",
    ):
        super().__init__()

        self.online_encoder = net
        if not hasattr(net, 'feature_dim'):
            feature_dim = list(net.children())[-1].in_features
        else:
            feature_dim = net.feature_dim
        self.online_encoder.fc = MLP(feature_dim, projection_size, projection_hidden_size)  # projector

        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.stop_gradient = stop_gradient
        self.has_predictor = has_predictor
        
        # debug purpose
        # self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))
        # self.reset_moving_average()
        
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, image_one, image_two):
        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)

        # online_pred_one = self.online_predictor(online_proj_one)
        # online_pred_two = self.online_predictor(online_proj_two)

        if self.stop_gradient:
            with torch.no_grad():
                if self.target_encoder is None:
                    self.target_encoder = self._get_target_encoder()
                target_proj_one = self.target_encoder(image_one)
                target_proj_two = self.target_encoder(image_two)

                target_proj_one = target_proj_one.detach()
                target_proj_two = target_proj_two.detach()


        # loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        # loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        # loss = loss_one + loss_two
        return online_proj_one, online_proj_two, target_proj_one, target_proj_two

class BYOL_Server(nn.Module):
    def __init__(
        self,
        projection_size=2048,
        projection_hidden_size=4096,
        moving_average_decay=0.99,
        predictor_network="2_layer",
    ):
        super().__init__()

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, predictor_network)

    def forward(self, online_proj_one, online_proj_two, target_proj_one, target_proj_two):

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)


        loss_one = byol_loss_fn(online_pred_one, target_proj_two)
        loss_two = byol_loss_fn(online_pred_two, target_proj_one)
        loss = loss_one + loss_two
        
        return loss.mean()
