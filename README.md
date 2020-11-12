### The Convolution Exponential and Generalized Sylvester Flows


![Banner](https://github.com/ehoogeboom/convolution_exponential_and_sylvester/blob/main/visualizations/images/banner.png?raw=true)

#### Example code.

Below an example of the implementation of the convolution exponential.
Note the very cool property that inv_conv_exp is computed by simply negating
the kernel. For more detailed code see [here](https://github.com/ehoogeboom/convolution_exponential_and_sylvester/blob/main/models/transformations/convexp/functional.py#L42).

```python
import torch.nn.functional as F


def conv_exp(input, kernel, terms=10):
    B, C, H, W = input.size()

    result = input
    product = input

    for i in range(1, terms + 1):
        product = F.conv2d(product, kernel, padding=1, stride=(1, 1)) / i
        result = result + product

    return result


def inv_conv_exp(input, kernel, terms=10):
    return conv_exp(input, -kernel, terms)
```



#### Abstract
This paper introduces a new method to build linear flows,
by taking the exponential of a linear transformation.
This linear transformation does not need to be invertible itself,
and the exponential has the following desirable properties: it is
guaranteed to be invertible, its inverse is straightforward to
compute and the log Jacobian determinant is equal to the trace of
the linear transformation. An important insight is that the
exponential can be computed implicitly, which allows the use of
convolutional layers. Using this insight, we develop new invertible
transformations named convolution exponentials and graph convolution
exponentials, which retain the equivariance of their underlying
transformations. In addition, we generalize Sylvester Flows and
propose Convolutional Sylvester Flows which are based on the
generalization and the convolution exponential as basis change.
Empirically, we show that the convolution exponential outperforms
other linear transformations in generative flows on CIFAR10 and the
graph convolution exponential improves the performance of graph
normalizing flows. In addition, we show that Convolutional Sylvester
Flows improve performance over residual flows as a generative flow
model measured in log-likelihood.

#### Experiments
The commands below reproduce the results for the mixing experiments:
```
python main_experiment.py --n_levels 2 --densenet_depth 8 --n_subflows 10 --densenet_growth 64 --n_intermediate_channels 64 --use_splitprior --dequantize_distribution flow --use_gated_conv --epochs 1000 --exp_name 'conv1x1' --mixing '1x1'
```

```
python main_experiment.py --n_levels 2 --densenet_depth 8 --n_subflows 10 --densenet_growth 63 --n_intermediate_channels 64 --use_splitprior --dequantize_distribution flow --use_gated_conv --epochs 1000 --exp_name 'convexp' --mixing 'convexp'
```

```
python main_experiment.py --n_levels 2 --densenet_depth 8 --n_subflows 10 --densenet_growth 63 --n_intermediate_channels 64 --use_splitprior --dequantize_distribution flow --use_gated_conv --epochs 1000 --exp_name 'emerging' --mixing 'emerging'
```

```
python main_experiment.py --n_levels 2 --densenet_depth 8 --n_subflows 10 --densenet_growth 63 --n_intermediate_channels 64 --use_splitprior --dequantize_distribution flow --use_gated_conv --epochs 1000 --exp_name 'woodbury' --mixing 'woodbury'
```

The Commands below reproduce the results for the residual transformations experiment:

Conv. Sylvester Flow (var. deq.):
```
python experiment_sylvester_network.py --n_levels 3 --n_subflows 16 --n_internal_channels 528 --use_splitprior --dequantize_distribution flow --epochs 1000 --exp_name 'sylvester' --model_type sylvester
```

Conv. Sylvester Flow (unif. deq.):
```
python experiment_sylvester_network.py --n_levels 3 --n_subflows 16 --n_internal_channels 528 --use_splitprior --dequantize_distribution uniform --epochs 1000 --exp_name 'sylvester' --model_type sylvester
```

Residual Block Flow (var. deq.):
```
python experiment_sylvester_network.py --n_levels 3 --n_subflows 16 --n_intermediate_channels 528 --use_splitprior --dequantize_distribution flow --epochs 1000 --exp_name 'residualflow' --model_type residual_flow
```

Residual Block Flow (unif. deq.):
```
python experiment_sylvester_network.py --n_levels 3 --n_subflows 16 --n_intermediate_channels 528 --use_splitprior --dequantize_distribution uniform --epochs 1000 --exp_name 'residualflow' --model_type residual_flow
```

Baseline Coupling Flow (var. deq.):
```
python experiment_sylvester_network.py --n_levels 3 --n_subflows 16 --n_intermediate_channels 528 --use_splitprior --dequantize_distribution flow --epochs 1000 --exp_name 'coupling' --model_type coupling_flow
```

Baseline Coupling Flow (unif. deq.):
```
python experiment_sylvester_network.py --n_levels 3 --n_subflows 16 --n_intermediate_channels 528 --use_splitprior --dequantize_distribution uniform --epochs 1000 --exp_name 'coupling' --model_type coupling_flow
```


Residual Block Flow (var. deq.) (equal memory):
```
python experiment_sylvester_network.py --n_levels 3 --n_subflows 8 --n_intermediate_channels 528 --use_splitprior --dequantize_distribution flow --epochs 1000 --exp_name 'residualflow' --model_type residual_flow
```

Residual Block Flow (unif. deq.) (equal memory):
```
python experiment_sylvester_network.py --n_levels 3 --n_subflows 8 --n_intermediate_channels 528 --use_splitprior --dequantize_distribution uniform --epochs 1000 --exp_name 'residualflow' --model_type residual_flow
```


#### Acknowledgements
The Robert Bosch GmbH is acknowledged for financial support.

