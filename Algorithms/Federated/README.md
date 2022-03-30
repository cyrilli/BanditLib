## Implementation of our papers in federated bandits

To run the algorithms in simulation environment, execute
```
python Simulation.py --T 50000 --n 1000  # simulate homogeneous clients with linear reward model
python Simulation.py --T 50000 --n 1000 --globaldim 16  # simulate heterogeneous clients with linear reward model
python Simulation.py --T 50000 --n 1000 --reward_model sigmoid # simulate homogeneous clients with logistic reward model
```
- T: length of time horizon
- n: number of clients
- contextdim: dimension of the context vector
- globaldim: dimension of global components that are shared by all the clients
- reward_model: set reward model to be linear or logistic

If you find this code useful in your research, please consider citing:

    @article{li2021asynchronous,
      title={Asynchronous Upper Confidence Bound Algorithms for Federated Linear Bandits},
      author={Li, Chuanhao and Wang, Hongning},
      journal={arXiv preprint arXiv:2110.01463},
      year={2021}
    }

    @article{li2022communication,
      title={Communication Efficient Federated Learning for Generalized Linear Bandits},
      author={Li, Chuanhao and Wang, Hongning},
      journal={arXiv preprint arXiv:2202.01087},
      year={2022}
    }
