"""
Flow transforms and utilities.
"""
import torch
from typing import Callable
from glasflow.nflows import distributions, flows, transforms, utils
import glasflow.nflows.nn.nets as nflows_nets
import torch.nn.functional as F


def create_flow(
        input_dim: int,
        context_dim: int,
        num_flow_steps: int,
        base_transform: Callable, middle_transform: Callable,
        base_transform_kwargs: dict, middle_transform_kwargs: dict,
        final_transform: Callable, final_transform_kwargs: dict,
        emb_net=None,
):
    """
    Adapted from https://github.com/dingo-gw/dingo/tree/main/dingo.

    Build NSF model. This models the posterior distribution p(y|x).
    The model consists of
        * a base distribution (StandardNormal, dim(y))
        * a sequence of transforms, each conditioned on x
    :param input_dim: int,
        Dimensionality of y.
    :param context_dim: int,
        Dimensionality of the (embedded) context.
    :param num_flow_steps: int,
        Number of sequential transforms.
    :param base_transform: Callable,
        Main transform of a flow step. Context-sensitive.
    :param middle_transform: Callable,
        Intermediate transform of a flow step. Need not be context-sensitive.
    :param base_transform_kwargs: dict,
        Hyperparameters for the base transform, repeated in every step.
    :param middle_transform_kwargs: dict,
        Hyperparameters for the middle transform, repeated in every step.
    :param final_transform: Callable,
        Last transform of the samples. Need not be context-sensitive.
    :param final_transform_kwargs: dict,
        Hyperparameters for the final transform.
    :param emb_net: torch.nn.Module, None,
        Embedding net for the flow.

    :return: Flow
        The posterior model.
    """

    # We will always start from a N(0, 1)
    distribution = distributions.StandardNormal(shape=(input_dim,))
    transform = create_transform(
        num_flow_steps, input_dim, context_dim,
        base_transform, middle_transform,
        base_transform_kwargs, middle_transform_kwargs,
        final_transform, final_transform_kwargs
    )
    return flows.Flow(transform, distribution, embedding_net=emb_net)


def create_transform(
        num_flow_steps: int, param_dim: int, context_dim: int,
        base_transform: Callable, middle_transform: Callable,
        base_transform_kwargs: dict, middle_transform_kwargs: dict,
        final_transform: Callable, final_transform_kwargs: dict
):
    """
    Adapted from https://github.com/dingo-gw/dingo/tree/main/dingo.
    
    Build a sequence of custom transforms, which maps parameters y into the
    base distribution u (noise). Transforms are conditioned on context data x.

    Note that the forward map is f^{-1}(y, x).

    Each step in the sequence consists of
        * An custom intermediate transform of y, usually context-insensitive.
        * A complex (NSF+affine in the example) transform of y, conditioned on x.
    There is one final customizable transform at the end.

    :param num_flow_steps: int,
        Number of transforms in sequence.
    :param param_dim: int,
        Dimensionality of parameter space (y).
    :param context_dim: int,
        Dimensionality of context (x).
    :param base_transform: Callable,
        Main transform of a flow step. Context-sensitive.
    :param middle_transform: Callable,
        Intermediate transform of a flow step. Need not be context-sensitive.
    :param base_transform_kwargs: dict,
        Hyperparameters for the base transform, repeated in every step.
    :param middle_transform_kwargs: dict,
        Hyperparameters for the middle transform, repeated in every step.
    :param final_transform: Callable,
        Last transform of the samples. Need not be context-sensitive.
    :param final_transform_kwargs: dict,
        Hyperparameters for the final transform.

    :return: Transform
        the NSF transform sequence
    """

    transform = transforms.CompositeTransform(
        [transforms.CompositeTransform([
            middle_transform(param_dim, **middle_transform_kwargs),
            base_transform(
                i, param_dim, context_dim=context_dim,
                **base_transform_kwargs), ]
        )
            for i in range(num_flow_steps)]
        + [final_transform(param_dim, **final_transform_kwargs)]
    )

    return transform


def random_perm_and_lulinear(param_dim: int):
    """
    Create the composite linear transform PLU.

    :param param_dim: int
        dimension of the parameter space
    :return: nde.Transform
        the linear transform PLU
    """

    return transforms.CompositeTransform([
        transforms.RandomPermutation(features=param_dim),
        transforms.LULinear(param_dim, identity_init=True),
    ])


def mask_affine_autoreg(i: int,
                        param_dim: int,
                        context_dim: int,
                        hidden_dim: int,
                        num_transform_blocks: int,
                        use_residual_blocks=True,
                        random_mask=False,
                        activation: str | Callable = F.relu,
                        dropout_probability=0.0,
                        use_batch_norm: bool = False):
    """Reverse permutation + masked affine autoregressive transform.
    See https://arxiv.org/abs/1705.07057 for the later."""
    if type(activation) is str:
        activation = get_activation_function_from_string(activation)
    return transforms.CompositeTransform([
        transforms.ReversePermutation(features=param_dim),
        transforms.MaskedAffineAutoregressiveTransform(features=param_dim,
                                                       context_features=context_dim,
                                                       hidden_features=hidden_dim,
                                                       num_blocks=num_transform_blocks,
                                                       use_residual_blocks=use_residual_blocks,
                                                       random_mask=random_mask,
                                                       activation=activation,
                                                       dropout_probability=dropout_probability,
                                                       use_batch_norm=use_batch_norm)])


# Usable only if inputs to spline in interval [0, 1]
def mask_piece_l_autoreg(i: int,
                         param_dim: int,
                         context_dim: int,
                         hidden_dim: int,
                         num_transform_blocks: int,
                         use_residual_blocks=True,
                         random_mask=False,
                         activation: str | Callable = F.relu,
                         dropout_probability=0.0,
                         num_bins: int = 8,
                         use_batch_norm: bool = False
                         ):
    if type(activation) is str:
        activation = get_activation_function_from_string(activation)
    return transforms.CompositeTransform([
        transforms.ReversePermutation(features=param_dim),
        transforms.MaskedPiecewiseLinearAutoregressiveTransform(features=param_dim,
                                                                context_features=context_dim,
                                                                hidden_features=hidden_dim,
                                                                num_bins=num_bins,
                                                                num_blocks=num_transform_blocks,
                                                                use_residual_blocks=use_residual_blocks,
                                                                random_mask=random_mask,
                                                                activation=activation,
                                                                dropout_probability=dropout_probability,
                                                                use_batch_norm=use_batch_norm)])


# Usable only if inputs to spline in interval [0, 1]
def mask_piece_c_autoreg(i: int,
                         param_dim: int,
                         context_dim: int,
                         hidden_dim: int,
                         num_transform_blocks: int,
                         use_residual_blocks=True,
                         random_mask=False,
                         activation: str | Callable = F.relu,
                         dropout_probability=0.0,
                         num_bins: int = 8,
                         use_batch_norm: bool = False
                         ):
    if type(activation) is str:
        activation = get_activation_function_from_string(activation)
    return transforms.CompositeTransform([
        transforms.ReversePermutation(features=param_dim),
        transforms.MaskedPiecewiseCubicAutoregressiveTransform(features=param_dim,
                                                               context_features=context_dim,
                                                               hidden_features=hidden_dim,
                                                               num_bins=num_bins,
                                                               num_blocks=num_transform_blocks,
                                                               use_residual_blocks=use_residual_blocks,
                                                               random_mask=random_mask,
                                                               activation=activation,
                                                               dropout_probability=dropout_probability,
                                                               use_batch_norm=use_batch_norm)])


def mask_piece_q_autoreg(i: int,
                         param_dim: int,
                         context_dim: int,
                         hidden_dim: int,
                         num_transform_blocks: int,
                         use_residual_blocks=True,
                         random_mask=False,
                         activation: str | Callable = F.relu,
                         dropout_probability=0.0,
                         num_bins: int = 8,
                         tail_bound: float = 1.0,
                         use_batch_norm: bool = False
                         ):
    if type(activation) is str:
        activation = get_activation_function_from_string(activation)
    return transforms.CompositeTransform([
        transforms.ReversePermutation(features=param_dim),
        transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(features=param_dim,
                                                                   context_features=context_dim,
                                                                   hidden_features=hidden_dim,
                                                                   num_bins=num_bins,
                                                                   tails='linear',
                                                                   tail_bound=tail_bound,
                                                                   num_blocks=num_transform_blocks,
                                                                   use_residual_blocks=use_residual_blocks,
                                                                   random_mask=random_mask,
                                                                   activation=activation,
                                                                   dropout_probability=dropout_probability,
                                                                   use_batch_norm=use_batch_norm)])


def mask_piece_rq_autoreg(i: int,
                          param_dim: int,
                          context_dim: int,
                          hidden_dim: int,
                          num_transform_blocks: int,
                          use_residual_blocks=True,
                          random_mask=False,
                          activation: str | Callable = F.relu,
                          dropout_probability=0.0,
                          num_bins: int = 8,
                          tail_bound: float = 1.0,
                          use_batch_norm: bool = False
                          ):
    if type(activation) is str:
        activation = get_activation_function_from_string(activation)
    return transforms.CompositeTransform([
        transforms.ReversePermutation(features=param_dim),
        transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=param_dim,
                                                                           context_features=context_dim,
                                                                           hidden_features=hidden_dim,
                                                                           num_bins=num_bins,
                                                                           tails='linear',
                                                                           tail_bound=tail_bound,
                                                                           num_blocks=num_transform_blocks,
                                                                           use_residual_blocks=use_residual_blocks,
                                                                           random_mask=random_mask,
                                                                           activation=activation,
                                                                           dropout_probability=dropout_probability,
                                                                           use_batch_norm=use_batch_norm)])


def dingo_rq_coupling(i: int,
                      param_dim: int,
                      context_dim: int,
                      hidden_dim: int,
                      num_transform_blocks: int,
                      use_batch_norm: bool = False,
                      activation: str | Callable = F.relu,
                      dropout_probability=0.0,
                      num_bins: int = 8,
                      tail_bound: float = 1.0,
                      apply_unconditional_transform: bool = False,
                      ):
    """Dingo's implementation of a rq-coupling transform.
    https://github.com/dingo-gw/dingo/blob/main/dingo/core/nn/nsf.py#L99"""
    if type(activation) is str:
        activation = get_activation_function_from_string(activation)
    if param_dim == 1:
        mask = torch.tensor([1], dtype=torch.uint8)
    else:
        mask = utils.create_alternating_binary_mask(
            param_dim, even=(i % 2 == 0)
        )
    return transforms.PiecewiseRationalQuadraticCouplingTransform(
        mask=mask,
        transform_net_create_fn=(
            lambda in_features, out_features: nflows_nets.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_dim,
                context_features=context_dim,
                num_blocks=num_transform_blocks,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            )
        ),
        num_bins=num_bins,
        tails="linear",
        tail_bound=tail_bound,
        apply_unconditional_transform=apply_unconditional_transform,
    )


def dingo_rq_autoreg(i: int,
                     param_dim: int,
                     context_dim: int,
                     hidden_dim: int,
                     num_transform_blocks: int,
                     use_batch_norm: bool = False,
                     activation_fn=F.relu,
                     dropout_probability=0.0,
                     num_bins: int = 8,
                     tail_bound: float = 1.0,
                     apply_unconditional_transform: bool = False,
                     ):
    """Dingo's implementation of a rq-autoregressive transform.
        https://github.com/dingo-gw/dingo/blob/main/dingo/core/nn/nsf.py#L126"""
    return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
        features=param_dim,
        hidden_features=hidden_dim,
        context_features=context_dim,
        num_bins=num_bins,
        tails="linear",
        tail_bound=tail_bound,
        num_blocks=num_transform_blocks,
        use_residual_blocks=True,
        random_mask=False,
        activation=activation_fn,
        dropout_probability=dropout_probability,
        use_batch_norm=use_batch_norm,
    )


def d_rq_coupling_and_affine(i: int,
                             param_dim: int,
                             context_dim: int,
                             hidden_dim: int,
                             num_transform_blocks: int,
                             use_batch_norm: bool = False,
                             activation: str | Callable = F.relu,
                             dropout_probability=0.0,
                             num_bins: int = 8,
                             tail_bound: float = 1.0,
                             apply_unconditional_transform: bool = False,
                             use_residual_blocks=True,
                             random_mask=False,
                             ):
    """Composition of rq-coupling and autoregressive transforms 1 to 1."""
    return transforms.CompositeTransform([
        dingo_rq_coupling(i,
                          param_dim,
                          context_dim,
                          hidden_dim,
                          num_transform_blocks,
                          use_batch_norm,
                          activation,
                          dropout_probability,
                          num_bins,
                          tail_bound,
                          apply_unconditional_transform),
        mask_affine_autoreg(i,
                            param_dim,
                            context_dim,
                            hidden_dim,
                            num_transform_blocks,
                            use_residual_blocks,
                            random_mask,
                            activation,
                            dropout_probability,
                            use_batch_norm)

    ])

def d_rq_coupling_half_affine(i: int,
                            param_dim: int,
                            context_dim: int,
                            hidden_dim: int,
                            num_transform_blocks: int,
                            use_batch_norm: bool = False,
                            activation: str | Callable = F.relu,
                            dropout_probability=0.0,
                            num_bins: int = 8,
                            tail_bound: float = 1.0,
                            apply_unconditional_transform: bool = False,
                            use_residual_blocks=True,
                            random_mask=False,
                            ):
    """Composition of rq-coupling and autoregressive transforms 2 to 1."""
    if i % 2 == 0:
        return d_rq_coupling_and_affine(i,
                                        param_dim,
                                        context_dim,
                                        hidden_dim,
                                        num_transform_blocks,
                                        use_batch_norm,
                                        activation,
                                        dropout_probability,
                                        num_bins,
                                        tail_bound,
                                        apply_unconditional_transform,
                                        use_residual_blocks,
                                        random_mask)
    else:
        return dingo_rq_coupling(i,
                          param_dim,
                          context_dim,
                          hidden_dim,
                          num_transform_blocks,
                          use_batch_norm,
                          activation,
                          dropout_probability,
                          num_bins,
                          tail_bound,
                          apply_unconditional_transform)



transform_ref = {f.__name__: f for f in [random_perm_and_lulinear,
                                         mask_affine_autoreg,
                                         mask_piece_q_autoreg,
                                         mask_piece_rq_autoreg,
                                         dingo_rq_autoreg,
                                         dingo_rq_coupling,
                                         d_rq_coupling_and_affine,
                                         d_rq_coupling_half_affine]}


def get_activation_function_from_string(activation_name: str):
    """
    Returns an activation function, based on the name provided.

    :param activation_name: str
        name of the activation function, one of {'elu', 'relu', 'leaky_rely'}
    :return: function
        corresponding activation function
    """
    if activation_name.lower() == "elu":
        return F.elu
    elif activation_name.lower() == "relu":
        return F.relu
    elif activation_name.lower() == "leaky_relu":
        return F.leaky_relu
    else:
        raise ValueError("Invalid activation function.")
