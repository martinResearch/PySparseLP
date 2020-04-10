"""Sparse inverse covariance estimation."""

import matplotlib.pyplot as plt

import numpy as np

from pysparselp.SparseLP import SparseLP

from scipy import linalg

from sklearn.datasets import make_sparse_spd_matrix


class SparseInvCov(SparseLP):
    """Class to model the sparse inverse covariance problem."""

    def add_abs_penalization(self, ids, coef_penalization):

        aux = self.add_variables_array(
            ids.shape, upper_bounds=None, lower_bounds=0, costs=coef_penalization
        )

        if np.isscalar(coef_penalization):
            assert coef_penalization > 0
        # allows a penalization that is different for each edge (could be dependent on an edge detector)
        else:
            assert coef_penalization.shape == aux.shape
            assert np.min(coef_penalization) >= 0
        aux_ravel = aux.ravel()
        row_ravel = ids.ravel()
        cols = np.column_stack((row_ravel, aux_ravel))
        vals = np.tile(np.array([1, -1]), [ids.size, 1])
        self.add_linear_constraint_rows(cols, vals, lower_bounds=None, upper_bounds=0)
        vals = np.tile(np.array([-1, -1]), [ids.size, 1])
        self.add_linear_constraint_rows(cols, vals, lower_bounds=None, upper_bounds=0)


def run(display=True):

    ##############################################################################
    # Generate the data
    n_samples = 40
    n_features = 20

    prng = np.random.RandomState(1)
    prec = make_sparse_spd_matrix(
        n_features, alpha=0.98, smallest_coef=0.4, largest_coef=0.7, random_state=prng
    )
    cov = linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    x = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)

    ##############################################################################
    # Estimate the covariance
    emp_cov = np.dot(x.T, x) / n_samples

    lp = SparseInvCov()
    ids = lp.add_variables_array(
        shape=emp_cov.shape, lower_bounds=None, upper_bounds=None
    )
    lamb = 0.15
    from scipy import sparse

    c = sparse.kron(sparse.csr_matrix(emp_cov), sparse.eye(n_features))
    lp.add_constraints_sparse(
        c,
        np.eye(emp_cov.shape[0]).flatten() - lamb,
        np.eye(emp_cov.shape[0]).flatten() + lamb,
    )
    lp.add_abs_penalization(ids, 1)
    lp.convert_to_one_sided_inequality_system()
    x = lp.solve(method="mehrotra", nb_iter=6000, max_time=np.inf)[0]
    print(lp.pobj_curve)
    # x=LP.solve(method='chambolle_pock_ppd')[0]
    lp_prec_ = x[ids]
    lp_prec_ = 0.5 * (lp_prec_ + lp_prec_.T)
    plt.figure()
    v_max = 0.9 * prec.max()
    lp_prec_ = lp_prec_ * (np.abs(lp_prec_) > 1e-8)
    this_prec = lp_prec_
    lp_cov_ = np.linalg.inv(lp_prec_)

    ##############################################################################
    # Plot the results
    if display:

        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.02, right=0.98)

    # plot the covariances
    covs = [("Empirical", emp_cov), ("LP", lp_cov_), ("True", cov)]
    v_max = cov.max()
    if display:
        for i, (name, this_cov) in enumerate(covs):
            plt.subplot(2, 3, i + 1)
            plt.imshow(
                this_cov,
                interpolation="nearest",
                vmin=-v_max,
                vmax=v_max,
                cmap=plt.cm.RdBu_r,
            )
            plt.xticks(())
            plt.yticks(())
            plt.title("%s covariance" % name)

    # plot the precisions
    precs = [("Empirical", linalg.inv(emp_cov)), ("LP", lp_prec_), ("True", prec)]
    v_max = 0.9 * prec.max()
    if display:
        for i, (name, this_prec) in enumerate(precs):
            ax = plt.subplot(2, 3, i + 4)
            plt.imshow(
                np.ma.masked_equal(this_prec, 0),
                interpolation="nearest",
                vmin=-v_max,
                vmax=v_max,
                cmap=plt.cm.RdBu_r,
            )
            plt.xticks(())
            plt.yticks(())
            plt.title("%s precision" % name)
            ax.set_facecolor(".7")

        plt.tight_layout()

        plt.show()

    sum_abs_diff = np.sum(np.abs(lp_prec_ - prec))
    nb_zeros_lp = np.sum(lp_prec_ == 0)
    return sum_abs_diff, nb_zeros_lp


if __name__ == "__main__":
    run()
