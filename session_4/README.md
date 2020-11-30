# Practical session 4


## Function approximation

In this project, you are asked to solve the classic Mountain Car (https://gym.openai.com/envs/MountainCar-v0/). Unlike previous environment, states are continuous so that you need to approximate the Q values Q(s, a). For more details about action and observation space, please refer to the OpenAI documentation here: https://github.com/openai/gym/wiki/MountainCar-v0

![](mountain_car.gif)

You will implement:
* Q-Learning (with Function Approximation)
* Double Q-Learning (optional) (https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)


Usage command line:

```bash
python main.py -h
usage: main.py [-h] [--agent AGENT_CLASS] [--nepisodes n] [--max_iter n]
               [--batch nagent] [--verbose]

Practical Session on Function Approximation

optional arguments:
  -h, --help           show this help message and exit
  --agent AGENT_CLASS  Class to use for the agent. Must be in the agents
                       module. Possible choice: (QL, DQL)
  --nepisodes n        number of episodes to simulate
  --max_iter n         max number of iterations per episode
  --batch nagent       batch run several agents at the same time
  --verbose            Display cumulative results at each step
```


## How do I proceed to be evaluated ?

Send `agents.py` to heri(at)lri(dot)fr before December, 2nd 2020 at 23:59.
