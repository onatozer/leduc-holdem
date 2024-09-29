# Leduc'holdem
This repo contains the CFR code that we wrote last week, but this time integrated with the Leduc holdem instead of Kuhn poker. Right now, there are three functions left that need to be implemented: calculate_strategy() and get_average_strategy() in infoset.py, and _state_abstraction in cfr.py  


## Usage
To train your model with x amount of calls to walk_trees, run 
```bash
python3 train_model.py --epochs <num_iterations>
```

To see how well your model performs against a random agent, run
```bash
python3 test_model.py 
```
In general, if your model achieved equilibrium, then it should get an average reward above .7.

A model trained without any abstractions needs a little over 1000 epochs to achieve equilibrium, lets see if you can beat that! 
