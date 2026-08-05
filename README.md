# Paper

This repository contains the supplementary material of the paper:

	Matteo Zavatteri, Davide Bresolin, Nicolò Navarin. 
	Certified Neural Networks: from Verification to Synthesis. Journal of Artificial Intelligence Research, 2026.

# Dataset

This code works on the original dataset of the paper:
	
	Jonas Tjomsland, Sinan Kalkan and Hatice Gunes
	Mind Your Manners! A Dataset and A Continual Learning Approach for Assessing Social Appropriateness of Robot Actions
	Frontiers in Robotics and AI, Special Issue on Lifelong Learning and Long-term Human-Robot Interaction, 9:1-18, 2022

[Link to the paper](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.669420/full)

We do not provide the dataset here. However, to get all set up to reproduce the experiments:

- Download [https://github.com/jonastjoms/MANNERS-DB/blob/master/src/data/not_normallized.csv](https://github.com/jonastjoms/MANNERS-DB/blob/master/src/data/not_normallized.csv)
- Run `$ python3 preprocessing.py`. This will create the file `dataset.csv`.

To run all the experiments of the paper run:

	$ python3 Run.py

# Usage

To run a single experiment run 

	$ python3 RobotCertifier.py -h 

to see the help screen (see also inside Run.py to have example of possible commands).

# Data
The original data of the experimental evaluation discussed in the paper in contained in the folder `experiments`.
