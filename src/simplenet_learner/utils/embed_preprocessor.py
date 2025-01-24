import torch
import torch.nn.functional as F


class EmbeddingPreprocessor(torch.nn.Module):
    """
    A PyTorch module for preprocessing input features using a list of MeanMapper modules.
    Args:
        input_dims (list of int): List of input dimensions for each feature.
        output_dim (int): The output dimension for the MeanMapper modules.
    Attributes:
        input_dims (list of int): List of input dimensions for each feature.
        output_dim (int): The output dimension for the MeanMapper modules.
        preprocessing_modules (torch.nn.ModuleList): List of MeanMapper modules for preprocessing each feature.
    Methods:
        forward(features):
            Applies the preprocessing modules to the input features and returns the processed features.
            Args:
                features (list of torch.Tensor): List of input feature tensors.
            Returns:
                torch.Tensor: Processed feature tensors stacked along the second dimension.
    """

    def __init__(self, input_dims: list[int], output_dim: int):
        super(EmbeddingPreprocessor, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features: list[torch.Tensor]):
        """
        Applies a sequence of preprocessing modules to a list of input features and stacks the results.
        Args:
            features (list[torch.Tensor]): A list of input tensors to be processed by the corresponding preprocessing modules.
        Returns:
            torch.Tensor: A tensor containing the processed features stacked along the second dimension.
        """

        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    """
    A PyTorch module that performs mean pooling on input features to a specified dimension.
    Args:
        preprocessing_dim (int): The dimension to which the input features will be pooled.
    Methods:
        forward(features):
            Applies adaptive average pooling to the input features and reshapes them.
            Args:
                features (torch.Tensor): The input features tensor of shape (batch_size, feature_dim).
            Returns:
                torch.Tensor: The pooled features tensor of shape (batch_size, preprocessing_dim).
    """

    def __init__(self, preprocessing_dim: int):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor):
        """
        Forward pass for the embed preprocessor.
        This method reshapes the input tensor and applies adaptive average pooling.
        Args:
            features (torch.Tensor): Input tensor of shape (batch_size, feature_dim).
        Returns:
            torch.Tensor: Processed tensor after adaptive average pooling, with shape
                          (batch_size, preprocessing_dim).
        """

        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class EmbeddingAggregator(torch.nn.Module):
    """
    A PyTorch module that reshapes and average pools input features to a target dimension.
    Args:
        target_dim (int): The target dimension for the output features.
    Methods:
        forward(features):
            Reshapes and average pools the input features to the target dimension.
            Args:
                features (torch.Tensor): Input tensor of shape (batchsize, number_of_layers, input_dim).
            Returns:
                torch.Tensor: Output tensor of shape (batchsize, target_dim).
    """

    def __init__(self, target_dim: int):
        super(EmbeddingAggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor):
        """
        Forward pass for the embed preprocessor.
        This method reshapes the input features and applies adaptive average pooling
        to transform the input dimensions to the target dimensions.
        Args:
            features (torch.Tensor): Input tensor of shape (batchsize, number_of_layers, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batchsize, target_dim).
        """

        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)
