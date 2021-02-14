"""Expriment with learning an optimizer for LP problem as a Neural network that performes primal-dual minimization iterates."""

import datetime
import os

import matplotlib.pyplot as plt

import numpy as np

from pysparselp.randomLP import generate_random_lp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import inf


def scipy_sparse_to_torch(sparse_matrix):
    coo = sparse_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices.astype(np.int32))
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.DoubleTensor(i, v, torch.Size(shape))


def generate_random_lps(nb_problems=10, get_dual_sol=True):

    # generate set of problems
    lp_with_sol = []

    for problem_id in range(nb_problems):
        print(f"generating problem {problem_id}")
        lp, feasible_x, scipy_sol = generate_random_lp(
            nbvar=5, n_eq=0, n_ineq=5, sparsity=0.8, seed=None
        )
        lp.convert_to_all_inequalities()

        if get_dual_sol:
            lp.convert_to_single_inequalities_without_bounds()  # simpler to get a dual problem
            lp_dual = lp.get_dual()
            y_ineq, _ = lp_dual.solve(method="scipy_revised_simplex")
            assert (
                abs(lp.costsvector.dot(scipy_sol) + lp_dual.costsvector.dot(y_ineq))
                < 1e-8
            )
            lp.dual_solution = y_ineq
        else:
            lp.convert_to_one_sided_inequality_system()

        lp.solution = scipy_sol
        lp_with_sol.append(lp)

    return lp_with_sol


class DeepPrimalDual(nn.Module):
    """Class to learn DNN that does primal dual optimization steps."""

    def __init__(
        self,
        nch_primal=(4, 4),
        nch_dual=(4, 4),
        has_bounds=False,
        coef_increase_primal=1,
        coef_increase_dual=1,
    ):
        super(DeepPrimalDual, self).__init__()

        self.primal_convs = nn.ModuleList()
        self.primal_conv_bns = nn.ModuleList()

        self.dual_convs = nn.ModuleList()
        self.dual_conv_bns = nn.ModuleList()

        self.nch_dual = nch_dual
        self.nch_primal = nch_primal
        self.has_bounds = has_bounds

        self.coef_increase_primal = coef_increase_primal
        self.coef_increase_dual = coef_increase_dual

        if has_bounds:
            nch_add_primal = 4
            nch_primal_input = nch_primal[-1] + nch_dual[-1]
        else:
            nch_add_primal = 2
            nch_primal_input = nch_primal[-1] + nch_dual[-1]

        nc_primal = (nch_primal_input, *nch_primal)

        nch_add_dual = 2

        nch_dual_input = nch_primal[-1] + nch_dual[-1]
        nc_dual = (nch_dual_input, *nch_dual)

        for i in range(len(nc_primal) - 1):

            self.primal_convs.append(
                nn.Conv1d(nc_primal[i] + nch_add_primal, nc_primal[i + 1], 1)
            )

        for i in range(len(nc_dual) - 1):
            self.dual_convs.append(
                nn.Conv1d(nc_dual[i] + nch_add_dual, nc_dual[i + 1], 1)
            )

        self.dual_update = nn.Conv1d(nc_dual[-1] + nch_add_dual, 1, 1)
        self.primal_update = nn.Conv1d(nc_primal[-1] + nch_add_primal, 1, 1)

    def forward(self, lp, nb_max_iter, loss_decay=0.90, beta=1, cheat=False, std_y=1.0):

        assert lp.a_equalities is None
        assert lp.b_lower is None
        # assert(lp.has_single_inequalities_without_bounds())

        a = scipy_sparse_to_torch(lp.a_inequalities)
        b = torch.FloatTensor(lp.b_upper[:, None])
        c = torch.FloatTensor(lp.costsvector[:, None])

        if self.has_bounds:
            assert lp.all_bounded()
            lb = torch.FloatTensor(lp.lower_bounds[:, None])
            ub = torch.FloatTensor(lp.upper_bounds[:, None])
            x = beta * (torch.rand(a.shape[1], 1, dtype=torch.float32) - 0.5) * (
                ub - lb
            ) + 0.5 * (ub + lb)
        else:
            x = torch.randn(a.shape[1], 1, dtype=torch.float32) * beta

        y = torch.randn(a.shape[0], 1, dtype=torch.float32) * std_y

        if cheat:
            alpha = np.random.rand()
            solution_pytorch = torch.FloatTensor(lp.solution[:, None])
            x = alpha * solution_pytorch + (1 - alpha) * x

            if lp.dual_solution is not None:
                y = (
                    alpha * (torch.FloatTensor(lp.dual_solution[:, None]))
                    + (1 - alpha) * y
                )

        # x = torch.zeros(a.shape[1], 1, dtype=torch.float32)

        y_h = torch.zeros(self.nch_dual[-1], a.shape[0], dtype=torch.float32)

        x_h = torch.zeros(self.nch_primal[-1], a.shape[1], dtype=torch.float32)

        at = a.transpose(1, 0)

        distances_primal_list = []
        distances_dual_list = []

        for _ in range(nb_max_iter):

            # Update the primal variables

            # compute constraint violations
            r = a.matmul(x) - b
            a_t_x_h = a.matmul(x_h.T)

            for i in range(len(self.dual_convs)):
                if i == 0:
                    y_h = torch.cat((y_h, a_t_x_h.T), 0)
                y_h = torch.cat((y_h, y.T, r.T), 0)
                y_h = F.relu(self.dual_convs[i](y_h[None, :, :])).squeeze(dim=0)

            y_h2 = torch.cat((y_h, y.T, r.T), 0)

            y = y + self.dual_update(y_h2[None, :, :]).squeeze(dim=0).T

            # compute modified cost
            d = c + at.matmul(y)
            at_y_h = at.matmul(y_h.T)

            for i in range(len(self.primal_convs)):
                if i == 0:
                    x_h = torch.cat((x_h, at_y_h.T))
                if self.has_bounds:
                    x_h = torch.cat((x_h, x.T, d.T, lb.T, ub.T))
                else:
                    x_h = torch.cat((x_h, x.T, d.T))

                x_h = F.relu(self.primal_convs[i](x_h[None, :, :])).squeeze(dim=0)

            if self.has_bounds:
                x_h2 = torch.cat((x_h, x.T, d.T, lb.T, ub.T))
            else:
                x_h2 = torch.cat((x_h, x.T, d.T))

            x = x + self.primal_update(x_h2[None, :, :]).squeeze(dim=0).T

            if lp.all_bounded():
                # may cause problem due to gradient zeroing
                x = torch.max(torch.min(x, ub), lb)

            squared_distance_primal = torch.sum(
                (x - torch.FloatTensor(lp.solution[:, None])) ** 2
            )
            distance_primal = squared_distance_primal.sqrt()
            if lp.dual_solution is not None:
                squared_distance_dual = torch.sum(
                    (y - torch.FloatTensor(lp.dual_solution[:, None])) ** 2
                )
                distance_dual = squared_distance_dual.sqrt()
                distances_dual_list.append(distance_dual)

            distances_primal_list.append(distance_primal)

        distances_primal = torch.stack(distances_primal_list)

        mean_loss_primal = torch.mean(distances_primal)
        increases_primal = torch.clamp_min(
            distances_primal[1:] - distances_primal[:-1], 0
        )
        loss_primal = mean_loss_primal + self.coef_increase_primal * torch.mean(
            increases_primal
        )

        if lp.dual_solution is not None:
            distances_dual = torch.stack(distances_dual_list)

            mean_loss_dual = torch.mean(distances_dual)
            increases_dual = torch.clamp_min(
                distances_dual[1:] - distances_dual[:-1], 0
            )
            loss_dual = mean_loss_dual + self.coef_increase_dual * torch.mean(
                increases_dual
            )

        else:
            loss_dual = 0
            distances_dual = None

        loss = loss_primal + loss_dual

        return x, y, loss, distances_primal, distances_dual


def train(
    net,
    max_iter=1000,
    outputs_folder="",
    use_dual_solution=True,
    nb_lp_per_batch=5,
    nb_batch=1,
    nb_new_batch=2,
    lp_renewal_frequency=100000,
    clip_grad_norm_threshold=None,
    nb_test=5,
    nb_max_iter_per_lp_train=20,
    cheat_train=False,
):

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    test_lps = generate_random_lps(nb_test, get_dual_sol=use_dual_solution)

    test_losses_curve = []

    fig_train_batch_visu = plt.figure()
    ax_train_batch_visu = fig_train_batch_visu.add_subplot(111)

    fig_test_batch_visu = plt.figure()
    ax_test_batch_visu = fig_test_batch_visu.add_subplot(111)

    all_train_lps = generate_random_lps(
        nb_batch * nb_lp_per_batch, get_dual_sol=use_dual_solution
    )

    os.makedirs(os.path.join(outputs_folder, "train_primal"))
    os.makedirs(os.path.join(outputs_folder, "train_dual"))
    os.makedirs(os.path.join(outputs_folder, "test_primal"))
    os.makedirs(os.path.join(outputs_folder, "test_dual"))

    for n_epoch in range(max_iter):

        if n_epoch % lp_renewal_frequency == 0:
            nb_new_lps = nb_new_batch * nb_lp_per_batch
            new_train_lps = generate_random_lps(
                nb_new_lps, get_dual_sol=use_dual_solution
            )

            # remove the oldest lps, maybe should instead keep the hard problems
            all_train_lps = all_train_lps[nb_new_lps:]
            # add new lps
            all_train_lps.extend(new_train_lps)

        for id_batch in range(nb_batch):
            batch_lps = all_train_lps[
                nb_lp_per_batch * id_batch : nb_lp_per_batch * (id_batch + 1)
            ]
            optimizer.zero_grad()
            losses = []
            all_train_distances_primal = []
            all_train_distances_dual = []

            for lp in batch_lps:
                x, y, mean_loss, distances_primal, distances_dual = net(
                    lp,
                    nb_max_iter=nb_max_iter_per_lp_train,
                    cheat=cheat_train,
                    beta=0,
                    std_y=0,
                )
                all_train_distances_primal.append(distances_primal.detach().numpy())
                if use_dual_solution:
                    all_train_distances_dual.append(distances_dual.detach().numpy())
                losses.append(mean_loss)

            if id_batch == 0:
                ax_train_batch_visu.clear()
                ax_train_batch_visu.plot(np.column_stack(all_train_distances_primal))
                ax_train_batch_visu.set_ylim([0, 2.5])
                fig_train_batch_visu.savefig(
                    os.path.join(
                        outputs_folder,
                        f"train_primal/batch{id_batch}_epoch{n_epoch}_convergence_curves.png",
                    )
                )
                if use_dual_solution:
                    ax_train_batch_visu.clear()
                    ax_train_batch_visu.plot(np.column_stack(all_train_distances_dual))
                    ax_train_batch_visu.set_ylim([0, 2.5])
                    fig_train_batch_visu.savefig(
                        os.path.join(
                            outputs_folder,
                            f"train_dual/batch{id_batch}_epoch{n_epoch}_convergence_curves.png",
                        )
                    )

            train_loss = torch.mean(torch.stack(losses))
            train_loss.backward()

            if clip_grad_norm_threshold:
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), clip_grad_norm_threshold, inf
                )

            optimizer.step()

            print(
                f"epoch {n_epoch} batch {id_batch} train loss ={train_loss.detach():f} "
            )

        with torch.no_grad():
            test_losses = []
            mean_abs_diffs = []
            all_test_distances_primal = []
            all_test_distances_dual = []
            for lp in test_lps:
                x, y, mean_loss, distances_primal, distances_dual = net(
                    lp, nb_max_iter=20, cheat=False, beta=0, std_y=0
                )
                all_test_distances_primal.append(distances_primal)
                if use_dual_solution:
                    all_test_distances_dual.append(distances_dual)
                mean_abs_diff = np.mean(np.abs(np.array(x) - lp.solution))
                test_losses.append(mean_loss)
                mean_abs_diffs.append(mean_abs_diff)
            test_loss = np.mean(test_losses)
            test_mean_abs_diff = np.mean(mean_abs_diffs)

            ax_test_batch_visu.clear()
            ax_test_batch_visu.plot(np.column_stack(all_test_distances_primal))
            ax_test_batch_visu.set_ylim([0, 2.5])
            fig_test_batch_visu.savefig(
                os.path.join(
                    outputs_folder, f"test_primal/epoch{n_epoch}_convergence_curves.png"
                )
            )
            if use_dual_solution:
                ax_test_batch_visu.clear()
                ax_test_batch_visu.plot(np.column_stack(all_test_distances_dual))
                ax_test_batch_visu.set_ylim([0, 2.5])
                fig_test_batch_visu.savefig(
                    os.path.join(
                        outputs_folder,
                        f"test_dual/epoch{n_epoch}_convergence_curves.png",
                    )
                )

        test_losses_curve.append(test_loss)
        print(
            f"epoch {n_epoch} batch {id_batch} test_loss = {test_loss} test_mean_abs_diff ={test_mean_abs_diff:f} "
        )


if __name__ == "__main__":
    np.random.seed(0)
    outputs_folder = os.path.join(
        os.path.dirname(__file__),
        "deep_primal_dual_training",
        datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S"),
    )
    os.makedirs(outputs_folder)
    torch.random.manual_seed(0)
    use_dual_solution = False
    has_bounds = not (use_dual_solution)
    dcp = DeepPrimalDual(has_bounds=has_bounds, coef_increase_primal=3)
    train(dcp, use_dual_solution=use_dual_solution, outputs_folder=outputs_folder)
