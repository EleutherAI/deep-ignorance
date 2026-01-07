# Adapted from https://github.com/moskomule/anatome/blob/master/anatome/distance.py

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor


def _svd(input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # torch.svd style
    try:
        U, S, Vh = torch.linalg.svd(input, full_matrices=False)
    except Exception as e:
        print(e)
        torch.backends.cuda.preferred_linalg_library("magma")
        U, S, Vh = torch.linalg.svd(input, full_matrices=False)
        breakpoint()

    V = Vh.transpose(-2, -1)
    return U, S, V


def _zero_mean(input: Tensor, dim: int) -> Tensor:
    return input - input.mean(dim=dim, keepdim=True)


def _check_shape_equal(x: Tensor, y: Tensor, dim: int):
    if x.size(dim) != y.size(dim):
        raise ValueError(
            f"x.size({dim}) == y.size({dim}) is expected, but got {x.size(dim)=}, {y.size(dim)=} instead."
        )


def cca_by_svd(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """CCA using only SVD.
    For more details, check Press 2011 "Canonical Correlation Clarified by Singular Value Decomposition"

    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    # torch.svd(x)[1] is vector
    u_1, s_1, v_1 = _svd(x)
    u_2, s_2, v_2 = _svd(y)
    uu = u_1.t() @ u_2
    u, diag, v = _svd(uu)
    # a @ (1 / s_1).diag() @ u, without creating s_1.diag()
    a = v_1 @ (1 / s_1[:, None] * u)
    b = v_2 @ (1 / s_2[:, None] * v)
    return a, b, diag


def cca_by_qr(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """CCA using QR and SVD.
    For more details, check Press 2011 "Canonical Correlation Clarified by Singular Value Decomposition"

    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    q_1, r_1 = torch.linalg.qr(x)
    q_2, r_2 = torch.linalg.qr(y)
    qq = q_1.t() @ q_2
    u, diag, v = _svd(qq)
    # a = r_1.inverse() @ u, but it is faster and more numerically stable
    a = torch.linalg.solve(r_1, u)
    b = torch.linalg.solve(r_2, v)
    return a, b, diag


def cca(x: Tensor, y: Tensor, backend: str) -> tuple[Tensor, Tensor, Tensor]:
    """Compute CCA, Canonical Correlation Analysis

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        backend: svd or qr

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    _check_shape_equal(x, y, 0)

    if x.size(0) < x.size(1):
        raise ValueError(f"x.size(0) >= x.size(1) is expected, but got {x.size()=}.")

    if y.size(0) < y.size(1):
        raise ValueError(f"y.size(0) >= y.size(1) is expected, but got {y.size()=}.")

    if backend not in ("svd", "qr"):
        raise ValueError(f"backend is svd or qr, but got {backend}")

    x = _zero_mean(x, dim=0)
    y = _zero_mean(y, dim=0)
    return cca_by_svd(x, y) if backend == "svd" else cca_by_qr(x, y)


def _svd_reduction(input: Tensor, accept_rate: float) -> Tensor:
    left, diag, right = _svd(input)
    full = diag.abs().sum()
    ratio = diag.abs().cumsum(dim=0) / full
    num = torch.where(
        ratio < accept_rate,
        input.new_ones(1, dtype=torch.long),
        input.new_zeros(1, dtype=torch.long),
    ).sum()
    return input @ right[:, :num]


def svcca_distance(x: Tensor, y: Tensor, accept_rate: float, backend: str) -> Tensor:
    """Singular Vector CCA proposed in Raghu et al. 2017.

    Args:
        x: input tensor of Shape DxH, where D>H
        y: input tensor of Shape DxW, where D>H
        accept_rate: 0.99
        backend: svd or qr

    Returns:

    """

    x = _svd_reduction(x, accept_rate)
    y = _svd_reduction(y, accept_rate)
    div = min(x.size(1), y.size(1))
    a, b, diag = cca(x, y, backend)
    return 1 - diag.sum() / div


def svcca_transform(
    x: Tensor,
    y: Tensor,
    accept_rate: float,
    backend: str
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Singular Vector CCA with transformation matrices.

    Similar to svcca_distance, but returns the transformation matrices
    and reduced representations for mapping between activation spaces.

    Args:
        x: input tensor of Shape DxH, where D>H
        y: input tensor of Shape DxW, where D>W
        accept_rate: 0.99 (threshold for SVD reduction)
        backend: svd or qr

    Returns:
        tuple of (x_reduced, y_reduced, a, b, diag) where:
        - x_reduced: SVD-reduced x (keeps components up to accept_rate variance)
        - y_reduced: SVD-reduced y (keeps components up to accept_rate variance)
        - a: CCA transformation matrix for x_reduced
        - b: CCA transformation matrix for y_reduced
        - diag: canonical correlations (diagonal of CCA)
    """
    x_reduced = _svd_reduction(x, accept_rate)
    y_reduced = _svd_reduction(y, accept_rate)
    a, b, diag = cca(x_reduced, y_reduced, backend)
    return x_reduced, y_reduced, a, b, diag


def pwcca_distance(x: Tensor, y: Tensor, backend: str) -> Tensor:
    """Projection Weighted CCA proposed in Marcos et al. 2018.

    Args:
        x: input tensor of Shape DxH, where D>H
        y: input tensor of Shape DxW, where D>H
        backend: svd or qr

    Returns:

    """

    a, b, diag = cca(x, y, backend)
    a, _ = torch.linalg.qr(a)  # reorthonormalize
    alpha = (x @ a).abs_().sum(dim=0)
    alpha /= alpha.sum()
    return 1 - alpha @ diag


def _debiased_dot_product_similarity(
    z: Tensor,
    sum_row_x: Tensor,
    sum_row_y: Tensor,
    sq_norm_x: Tensor,
    sq_norm_y: Tensor,
    size: int,
) -> Tensor:
    return (
        z
        - size / (size - 2) * (sum_row_x @ sum_row_y)
        + sq_norm_x * sq_norm_y / ((size - 1) * (size - 2))
    )


def linear_cka_distance(x: Tensor, y: Tensor, reduce_bias: bool) -> Tensor:
    """Linear CKA used in Kornblith et al. 19

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        reduce_bias: debias CKA estimator, which might be helpful when D is limited

    Returns:

    """

    _check_shape_equal(x, y, 0)

    x = _zero_mean(x, dim=0)
    y = _zero_mean(y, dim=0)
    dot_prod = (y.t() @ x).norm("fro").pow(2)
    norm_x = (x.t() @ x).norm("fro")
    norm_y = (y.t() @ y).norm("fro")

    if reduce_bias:
        size = x.size(0)
        # (x @ x.t()).diag()
        sum_row_x = torch.einsum("ij,ij->i", x, x)
        sum_row_y = torch.einsum("ij,ij->i", y, y)
        sq_norm_x = sum_row_x.sum()
        sq_norm_y = sum_row_y.sum()
        dot_prod = _debiased_dot_product_similarity(
            dot_prod, sum_row_x, sum_row_y, sq_norm_x, sq_norm_y, size
        )
        norm_x = _debiased_dot_product_similarity(
            norm_x.pow(2), sum_row_x, sum_row_x, sq_norm_x, sq_norm_x, size
        ).sqrt()
        norm_y = _debiased_dot_product_similarity(
            norm_y.pow(2), sum_row_y, sum_row_y, sq_norm_y, sq_norm_y, size
        ).sqrt()
    return 1 - dot_prod / (norm_x * norm_y)


def orthogonal_procrustes_distance(
    x: Tensor,
    y: Tensor,
) -> Tensor:
    """Orthogonal Procrustes distance used in Ding+21

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW

    Returns:

    """
    _check_shape_equal(x, y, 0)

    frobenius_norm = partial(torch.linalg.norm, ord="fro")
    nuclear_norm = partial(torch.linalg.norm, ord="nuc")

    x = _zero_mean(x, dim=0)
    x /= frobenius_norm(x)
    y = _zero_mean(y, dim=0)
    y /= frobenius_norm(y)
    # frobenius_norm(x) = 1, frobenius_norm(y) = 1
    # 0.5*d_proc(x, y)
    return 1 - nuclear_norm(x.t() @ y)
