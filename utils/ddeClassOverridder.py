import deepxde as dde
import torch
from deepxde.utils import run_if_all_none


class OxSepModel(dde.Model):

    def __init__(self, data, net, max_value):
        self.max_value = max_value
        super().__init__(data, net)

    def predict(self, x, operator=None, callbacks=None):
        output = super().predict(x, operator=operator, callbacks=callbacks)
        if operator is None:
            output *= self.max_value
        return output


class ODEWithReferences(dde.data.PDE):

    def __init__(self, geometry, pde, bcs, num_domain=0, num_boundary=0, train_distribution="Hammersley", anchors=None,
                 exclusions=None, solution=None, num_test=None, auxiliary_var_function=None,
                 reference_grid=None, reference_values=None, mpeak_beg=None, mpeak_end=None):

        self.reference_grid = torch.Tensor(reference_grid)
        self.reference_values = torch.Tensor(reference_values)
        self.mpeak_beg = mpeak_beg
        self.mpeak_end = mpeak_end
        self.reference_len = len(reference_grid)
        self.num_ref = len(reference_values)

        super().__init__(geometry, pde, bcs, num_domain, num_boundary, train_distribution, anchors, exclusions,
                         solution, num_test, auxiliary_var_function)

    @run_if_all_none("train_x", "train_y", "train_aux_vars")
    def train_next_batch(self, batch_size=None):
        self.train_x_all = torch.Tensor(self.train_points())
        self.train_x = self.reference_grid
        self.train_x = torch.vstack((self.train_x, self.train_x_all))
        return self.train_x, self.train_y, self.train_aux_vars

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):

        outputs_pde = outputs
        f = self.pde(inputs, outputs_pde)
        error_f = [fi[self.reference_len:] for fi in f]

        if not isinstance(loss_fn, (list, tuple)):
            loss_fn = [loss_fn] * (len(f) + self.num_ref * 2)
        elif len(loss_fn) != len(f) + len(self.bcs):
            raise ValueError(
                "There are {} errors, but only {} losses.".format(
                    len(f) + len(self.bcs), len(loss_fn)
                )
            )

        losses = [loss_fn[i](torch.zeros_like(error), error) for i, error in enumerate(error_f)]
        for i, reference in enumerate(self.reference_values):
            error = outputs[0:self.reference_len, -1] - reference
            losses.append(loss_fn[len(error_f) + i](torch.zeros_like(error), error))
            error_mpeak = error[self.mpeak_beg:self.mpeak_end]
            losses.append(loss_fn[len(error_f) + i](torch.zeros_like(error_mpeak), error_mpeak))
        return losses
