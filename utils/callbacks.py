import deepxde as dde
from deepxde.config import real
import numpy as np


class IntervalWithSmartResampling(dde.geometry.Interval):
    def __init__(self, l, r, model, num_oxides):
        super().__init__(l, r)
        self.model = model
        self.num_oxides = num_oxides

    def random_points(self, n, random="pseudo"):
        if random == "residual-based":
            uniform_grid = super().uniform_points(n * 10, False)
            full_error = self.model.predict(uniform_grid, operator=self.model.data.pde)
            squared_full_error = [error ** 2 for error in full_error]
            average_pde_error = np.sqrt(np.mean(squared_full_error[:self.num_oxides], axis=0))
            number_of_groups = self.r // 10
            total_error = average_pde_error.sum()
            if total_error == 0:
                return super().uniform_points(n, False)
            resulting_points = np.array([]).reshape(-1, 1)
            points_groups = np.array_split(uniform_grid, number_of_groups)
            error_groups = np.array_split(average_pde_error, number_of_groups)
            for interval, interval_error in zip(points_groups, error_groups):
                number_of_points_on_interval = int((n - number_of_groups) * (interval_error.sum() / total_error))
                left = interval[0]
                right = interval[-1]
                sampled_points_on_interval = np.linspace(left, right, num=number_of_points_on_interval + 1,
                                                         endpoint=True, dtype=dde.config.real(np)).reshape(-1, 1)
                resulting_points = np.concatenate([resulting_points, sampled_points_on_interval])

            return resulting_points
        else:
            return super().random_points(n, random)


class SolutionHistory(dde.callbacks.Callback):

    def __init__(
        self,
        filename,
        x1,
        x2,
        num_points=100,
        period=1,
    ):
        super().__init__()
        self.filename = filename
        x1 = np.array(x1)
        x2 = np.array(x2)
        self.x = (x1 + (x2 - x1) / (num_points - 1) * np.arange(num_points)[:, None]).astype(dtype=dde.config.real(np))
        self.period = period
        self.y = []
        self.epochs_since_last_save = 0

    def on_train_begin(self):
        self.y.append(self.model.predict(self.x)[:, :])

    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.on_train_begin()

    def on_train_end(self):
        fname_x = self.filename + "_x.txt"
        fname_y = self.filename + "_y.txt"
        self.y = np.array(self.y)
        np.savetxt(fname_x, self.x)
        np.savetxt(fname_y, np.array(self.y).reshape(self.y.shape[0], -1))


class ResidualBasedResampler(dde.callbacks.Callback):

    def __init__(self, period=100, n=1000, num_oxides=None):
        super().__init__()
        self.period = period
        self.epochs_since_last_resample = 0
        self.n = n
        self.num_oxides = num_oxides

    def on_epoch_end(self):
        self.epochs_since_last_resample += 1
        if self.epochs_since_last_resample < self.period:
            return
        self.epochs_since_last_resample = 0
        uniform_grid = self.model.data.geom.uniform_points(self.n * 10, False)
        full_error = self.model.predict(uniform_grid, operator=self.model.data.pde)
        squared_full_error = [error ** 2 for error in full_error]
        average_pde_error = np.sqrt(np.mean(squared_full_error[:self.num_oxides], axis=0))
        number_of_groups = self.model.data.geom.r // 10
        total_error = average_pde_error.sum()
        if total_error == 0:
            self.model.data.replace_with_anchors(self.model.data.geom.uniform_points(self.n, False))
        resulting_points = np.array([]).reshape(-1, 1)
        points_groups = np.array_split(uniform_grid, number_of_groups)
        error_groups = np.array_split(average_pde_error, number_of_groups)
        for interval, interval_error in zip(points_groups, error_groups):
            number_of_points_on_interval = int((self.n - number_of_groups) * (interval_error.sum() / total_error))
            left = interval[0]
            right = interval[-1]
            sampled_points_on_interval = np.linspace(left, right, num=number_of_points_on_interval + 1,
                                                     endpoint=True, dtype=dde.config.real(np)).reshape(-1, 1)
            resulting_points = np.concatenate([resulting_points, sampled_points_on_interval])
        self.model.data.replace_with_anchors(resulting_points)

