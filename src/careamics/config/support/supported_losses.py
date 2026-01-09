"""Losses supported by CAREamics."""

from careamics.utils import BaseEnum


# TODO register loss with custom_loss decorator?
class SupportedLoss(str, BaseEnum):
    """Supported losses.

    Attributes
    ----------
    MSE : str
        Mean Squared Error loss.
    MAE : str
        Mean Absolute Error loss.
    N2V : str
        Noise2Void loss.
    """

    MSE = "mse"
    MAE = "mae"
    N2V = "n2v"
    N2V_POISSON = "n2v_poisson"
    N2V_ANSCOMBE = "n2v_anscombe"
    N2V_SIGNAL_ONLY = "n2v_signal_only"
    PN2V = "pn2v"
    HDN = "hdn"
    MUSPLIT = "musplit"
    MICROSPLIT = "microsplit"
    DENOISPLIT = "denoisplit"
    DENOISPLIT_MUSPLIT = (
        "denoisplit_musplit"  # TODO refac losses, leave only microsplit
    )
    # CE = "ce"
    # DICE = "dice"
