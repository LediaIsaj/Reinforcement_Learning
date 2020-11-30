import argparse
import agents
import environment
import runner
import sys
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Practical Session on Function Approximation')
parser.add_argument('--agent', metavar='AGENT_CLASS', choices=['QL', 'DQL'], default='QL', type=str, help='Class to use for the agent. Must be in the \'agents\' module. Possible choice: (QL, DQL)')
parser.add_argument('--nepisodes', type=int, metavar='n', default='100', help='number of episodes to simulate')
parser.add_argument('--max_iter', type=int, metavar='n', default='10000', help='max number of iterations per episode')
parser.add_argument('--batch', type=int, metavar='nagent', default=None, help='batch run several agents at the same time')
parser.add_argument('--verbose', action='store_true', help='Display cumulative results at each step')

def main():
    args = parser.parse_args()
    print("Run Function Approximation Experiment")
    if args.agent == "QL":
        agent_class = agents.Q_Learning_Function_Approximation
    else:
        agent_class = agents.Double_Q_Learning
    env_class = environment.FA_Environment

    if args.batch is not None:
        print("Running a batched simulation with {} agents in parallel...".format(args.batch))
        my_runner = runner.BatchFARunner(env_class, agent_class, args.batch, args.verbose)
        final_reward, list_cumul = my_runner.loop(args.nepisodes, args.max_iter)
        print("Obtained a final average reward of {}".format(final_reward))
        print("Obtained a final reward of {}".format(final_reward))
        plt.plot(list_cumul)
        plt.xlabel("Iter")
        plt.ylabel("Cum. Reward")
        plt.title("Agent: {}".format(args.agent))
        plt.show()
    else:
        print("Running a single instance simulation...")
        my_runner = runner.FARunner(env_class(), agent_class(), args.verbose)
        final_reward = my_runner.loop(args.nepisodes, args.max_iter)


if __name__ == "__main__":
    main()
