# The Convolution Exponential and Generalized Sylvester Flows


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



