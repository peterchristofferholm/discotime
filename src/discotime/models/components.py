from typing import Optional, Type

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class Block(nn.Module):
    """Neural network building block for the `Net` class.

    Args:
        n_hidden_units: number of units in each hidden layer.
        add_skip_connection: Defaults to True.
        activation_function: Defaults to
            nn.SiLU.
        batch_normalization: Should batch
            normalization be performed? Defaults to True.
        dropout_rate: dropout_rate is being
            passed along to `nn.Dropout()`. If None, then dropout is not
            being used. Defaults to None.
    """

    def __init__(
        self,
        *,
        n_hidden_units: int,
        add_skip_connection: bool = True,
        activation_function: Type[nn.Module] = nn.SiLU,
        batch_normalization: bool = True,
        dropout_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        self._should_skip = add_skip_connection
        self.activation_function = activation_function
        self.net = nn.Sequential(
            nn.LazyBatchNorm1d() if batch_normalization else nn.Identity(),
            nn.LazyLinear(out_features=n_hidden_units),
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
            self.activation_function(),
            nn.LazyLinear(out_features=n_hidden_units),
            nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_function()(
            x + self.net(x) if self._should_skip else self.net(x)
        )


class Net(nn.Module):
    """
    Feed-forward neural network.

    Has support for dropout, skip connections, and activation function can
    be easily switched out for a different one.

    Args:
        n_out_features : length of output tensor (1D).
        n_blocks (int): Number of `Block()` units included.
        n_hidden_units (int): Number of neurons in each hidden unit.
        add_skip_connection: Defaults to True.
        activation_function: Defaults to
            nn.SiLU.
        batch_normalization: Should batch
            normalization be performed? Defaults to True.
        dropout_rate: dropout_rate is being
            passed along to `nn.Dropout()`. If None, then dropout is not
            being used. Defaults to None.

    :meta private:
    """

    def __init__(
        self,
        n_out_features: int,
        n_blocks: int,
        n_hidden_units: int,
        add_skip_connection: bool = True,
        activation_function: Type[nn.Module] = nn.SiLU,
        batch_normalization: bool = True,
        dropout_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(out_features=n_hidden_units),
            *(
                Block(
                    n_hidden_units=n_hidden_units,
                    activation_function=activation_function,
                    batch_normalization=batch_normalization,
                    dropout_rate=dropout_rate,
                    add_skip_connection=add_skip_connection,
                )
                for _ in range(n_blocks)
            ),
            nn.LazyLinear(out_features=n_out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def negative_log_likelihood(
    logits: torch.Tensor, time: torch.Tensor, event: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood for logistic hazard model with competing risks.

    The hazards are expected to be given as logits scale, i.e. they should not
    have been passed through ``torch.log_softmax()`` or similar.

    An implementation of equation (8.6) from Tutz and Schmid [1], inspired by
    the one in ``pycox`` following Kvamme et. al. [2]

    Args:
        logits (:obj:`torch.Tensor`): input logits
        time (:obj:`torch.Tensor`): discretized event times
        event (:obj:`torch.Tensor`): events (0=censored, 1/2/...=events)

    [1]: Tutz, Gerhard, and Matthias Schmid. Modeling discrete time-to-event
    data. New York: Springer, 2016.

    [2]: Kvamme, Håvard, Ørnulf Borgan, and Ida Scheel. "Time-to-event
    prediction with neural networks and Cox regression." arXiv preprint
    arXiv:1907.00825 (2019).
    """

    if logits.ndim != 3:
        raise ValueError(
            "A tensor with exactly three dimensions is expected, "
            f"instead {logits.ndim} dimension was supplied."
        )

    # construct labels that `F.cross_entropy()` can use
    time, event = time.view(-1, 1), event.view(-1, 1)
    target = torch.zeros(logits.shape[:2]).to(time)
    target = target.scatter(dim=1, index=time.long(), src=event)

    return torch.mean(
        (
            F.cross_entropy(
                input=rearrange(logits, "b t r -> b r t"),
                target=target.long(),
                reduction="none",
            )
            .cumsum(dim=1)
            .gather(dim=1, index=time.long())
        )
    )
