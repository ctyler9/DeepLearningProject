CS 7643 Project
Team Members: Joshua Gundugollu, Nazanin Tabatabaei, Clay Tyler, Naila Fatima, Ruize Yang
Group 48

Project Topic:Use a population-based/generational approach to train RL agents 
Github link: https://github.com/ctyler9/DeepLearningProject

We have implemented a novel generational approach to train RL agents as well as a baseline approach
which makes use of a shared buffer. The 2 approaches have been implemented for the Cartpole-v0 and 
Space Invaders-v0 gym environments of Open AI.

In order to run the approaches for the Cartpole-v0 environment:
Open the cartpole_complete_code.ipynb notebook.
	1. For the novel generational approach: Run Part 1 of the notebook (note that you need to run
all the import/setup cells before Part 1). Certain hyperparameters such as pop_size (population size/number of agents), num_gens (number of generations) and num_eps (number of episodes) can be changed.
	2. For the baseline approach: Run Part 2 of the notebook (note that you need to run all the import/setup cells before Part 1). Certain hyperparameters such as pop_size (population size/number of agents), num_gens (number of generations) and num_eps (number of episodes) can be changed.
The best_score variable tells us the highest score attained by following a certain approach.

In order to run the approaches for the Space Invaders-v0 environment:
Open the space_invaders_complete_code.ipynb notebook.
	1. For the novel generational approach: Run Part 2 of the notebook (note that you need to run
all the import/setup cells before Part 1). Certain hyperparameters such as pop_size (population size/number of agents), num_gens (number of generations) and num_eps (number of episodes) can be changed.
	2. For the baseline approach: Run Part 3 of the notebook (note that you need to run all the import/setup cells before Part 1). Certain hyperparameters such as pop_size (population size/number of agents), num_gens (number of generations) and num_eps (number of episodes) can be changed.
The best_score variable tells us the highest score attained by following a certain approach.
