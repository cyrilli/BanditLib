## Standard multi-armed bandit and linear bandit algorithms

To run the algorithms in multi-armed bandit simulation environment, execute:
```
python Simulation.py --contextdim 25 --actionset basis_vector
```
To run the algorithms in linear bandit simulation environment, execute:
```
python Simulation.py --contextdim 25 --actionset random
```
- actionset: Setting to “basis_vector” or “random”. “basis_vector” constrains the articles feature vectors to be basis vector like e_0 = [1, 0, 0, …, 0], e_1 = [0, 1, 0, …, 0]. Therefore, feature vectors of the articles will be orthogonal, which means observation about reward of one article brings no information about reward of another article. “random” means using randomly sampled vectors from l2 unit ball.
- context_dimension: dimension of article feature vector and user linear parameter.

Other parameters that are set in script:
- testing_iterations: total number of time steps to run
- NoiseScale: standard deviation of the Gaussian noise in the reward
- n_articles: total number of articles in the articles set
- n_users: total number of users
- poolArticleSize: If it is set to None, in each time step, the action set contains all articles. Otherwise, randomly sample a size “poolArticleSize” subset of articles in each time step as the action set. This is the setting that many linear bandit adopts.
