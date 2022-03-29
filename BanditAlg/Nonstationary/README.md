## Implementation of our papers in non-stationary bandits

To run the algorithms in simulation environment, execute
```
python Simulation.py --T 2500 --SMIN 400 --SMAX 2500 --n 30 --m 5 --sigma 0.09
```
- T: number of iterations to run
- SMIN: minimum length of each stationary period
- SMAX: maximum length of each stationary period
- n: number of users
- m: number of unique parameters shared by users
- sigma: standard deviation of Gaussian noise in observed reward

If you find this code useful in your research, please consider citing:

    @inproceedings{wu2018learning,
      title={Learning contextual bandits in a non-stationary environment},
      author={Wu, Qingyun and Iyer, Naveen and Wang, Hongning},
      booktitle={The 41st International ACM SIGIR Conference on Research \& Development in Information Retrieval},
      pages={495--504},
      year={2018}
    }

    @inproceedings{wu2019dynamic,
      title={Dynamic ensemble of contextual bandits to satisfy users' changing interests},
      author={Wu, Qingyun and Wang, Huazheng and Li, Yanen and Wang, Hongning},
      booktitle={The World Wide Web Conference},
      pages={2080--2090},
      year={2019}
    }

    @inproceedings{li2021unifying,
      title={Unifying clustered and non-stationary bandits},
      author={Li, Chuanhao and Wu, Qingyun and Wang, Hongning},
      booktitle={International Conference on Artificial Intelligence and Statistics},
      pages={1063--1071},
      year={2021},
      organization={PMLR}
    }
 
     @inproceedings{li2021and,
      title={When and Whom to Collaborate with in a Changing Environment: A Collaborative Dynamic Bandit Solution},
      author={Li, Chuanhao and Wu, Qingyun and Wang, Hongning},
      booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
      pages={1410--1419},
      year={2021}
    }
