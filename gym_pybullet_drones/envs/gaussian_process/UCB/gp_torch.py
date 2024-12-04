from collections import deque
import gpytorch.constraints
import torch
import numpy as np
from itertools import product
from gym_pybullet_drones.utils.utils import *
from sklearn.preprocessing import StandardScaler


class GaussianProcessTorch():

    def __init__(self, id=0, other_id=0):
        self.id = id
        self.other_id = other_id
        self.device = "cuda"
        self.gp_torch = GaussianProcessRegressorTorch(train_inputs=None,
                                                      train_targets=None).to(
                                                          self.device)
        self.observed_points: deque[np.ndarray] = deque()
        self.observed_value: deque[np.ndarray] = deque()

        self.curr_t = -1.0
        self.cache = {}
        # config
        self.scale_t = 3.0  # 指定 kernel lenghtscale of time.
        self.grid_size = 40
        self.grid = np.array(
            list(
                product(
                    np.linspace(-1, 1, self.grid_size),
                    np.linspace(-1, 1, self.grid_size),
                )))

    def add_observed_point(self, point: np.ndarray, value: np.ndarray):
        '''
        Add new data, and deltet old data
        Args
            point: [x,y,time]
            value: 0 or 1
        '''
        self.observed_points.append(point)
        self.observed_value.append(value)

        # Matern1.5: 2.817: 0.1%; 1.993: 1%; 1.376: 5%; 1.093: 10%
        dt = 1.993 * self.scale_t
        curr_t = point.squeeze()[-1]
        while curr_t - self.observed_points[0].squeeze()[-1] >= dt:
            self.observed_points.popleft()
            self.observed_value.popleft()

    def update_gp(self):
        if len(self.observed_points):
            # 更新数据即可
            train_x = torch.from_numpy(
                np.asarray(self.observed_points).reshape(-1,
                                                         3)).to(self.device)
            # train_y 需要一维度张量
            train_y = torch.from_numpy(
                np.asarray(self.observed_value).reshape(-1, )).to(self.device)

            self.gp_torch.train_inputs = [train_x]
            self.gp_torch.train_targets = train_y
        else:
            pass

    def update_grid(self, t):
        '''
        predict using gaussian process with grid, and storage it.
        '''
        if self.curr_t == t:
            return self.cache["y_pred_at_grid"], self.cache["std_at_grid"]

        self.curr_t = t
        gird_with_t = torch.from_numpy(add_t(self.grid, t)).to(self.device)
        predict_grid = self.gp_torch.predict(gird_with_t)
        mean = predict_grid.mean.cpu().detach().numpy()
        std = predict_grid.stddev.cpu().detach().numpy()

        self.cache["y_pred_at_grid"] = mean
        self.cache["std_at_grid"] = std
        return self.cache["y_pred_at_grid"], self.cache["std_at_grid"]

    def evaluate_unc(self, idx=None, t=None):
        '''
        在 t 时刻评估 unc, t 必须为最新
        '''
        if t is None or t != self.curr_t:
            raise ValueError
        if idx is not None:
            X = self.cache["std_at_grid"][idx]
            return np.mean(X)
        else:
            return np.mean(self.cache["std_at_grid"])


class GaussianProcessRegressorTorch(gpytorch.models.ExactGP):

    def __init__(self, train_inputs, train_targets):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-11)).cuda()
        # default noise of sklearn gaussian process
        likelihood.noise_covar.noise = torch.tensor(1e-10).cuda()
        likelihood.noise_covar.raw_noise.requires_grad = False
        super().__init__(train_inputs, train_targets, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5,
                                                          ard_num_dims=3)
        # 固定核参数
        self.covar_module.raw_lengthscale = torch.nn.Parameter(
            torch.tensor([0.1, 0.1, 3]), requires_grad=False)
        for param in self.covar_module.parameters():
            param.requires_grad = False
        self.covar_module.register_constraint(
            "raw_lengthscale",
            gpytorch.constraints.Interval(0.1,
                                          3,
                                          transform=None,
                                          inv_transform=None))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x,
                                                         covar_x,
                                                         validate_args=True)

    def fit(self, train_x, train_y, verbose=False):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        # loss of gp
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        train_iter = 1

        for i in range(train_iter):
            optimizer.zero_grad()
            ouput = self(train_x)
            loss = -mll(ouput, train_y)
            loss.backward()
            optimizer.step()

    def predict(self, x):
        '''
        Args
            x: 为带有 t 的 heatmap
        '''
        self.eval()
        self.likelihood.eval()

        # 清空训练数据，禁用联合推断
        self.set_train_data(inputs=None, targets=None, strict=False)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            t = self(x)
            observed_pred = self.likelihood(t)
        return observed_pred
