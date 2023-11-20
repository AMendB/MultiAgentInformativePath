"""
The :mod:`GaussianProcess.GPimplementations` module implements Gaussian Process
based in different libraries.
"""

from .GPModels import GaussianProcessScikit
from .GPModels import GaussianProcessGPyTorch
from .GPModels import ExactGPModel
# from .GPModels import GaussianProcessBoTorch

__all__ = ["GaussianProcessScikit", "GaussianProcessGPyTorch", "ExactGPModel"]