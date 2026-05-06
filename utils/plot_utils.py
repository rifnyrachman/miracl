import matplotlib.pyplot as plt
from ray import tune


class PlotUtils:

    def __init__(self, results):
        self.results = results

    def plot_reward(self):
        for i, result in enumerate(self.results):
            label = f"lr={result.config['lr']:.3f}, inner_adaptation_steps={result.config['inner_adaptation_steps']}"
            plt.plot(result.metrics_dataframe["timesteps_total"], result.metrics_dataframe["episode_reward_mean"], label=label)
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Reward')
        plt.legend()
        plt.show()

    def plot_eval(self):
        print(self.results.get_dataframe().columns.tolist())
        for i, result in enumerate(self.results):
            print(result.metrics)
            #print(result.config)
            print(result.metrics_dataframe.columns.tolist())

            label = f"lr={result.config['lr']:.3f}, inner_adaptation_steps={result.config['inner_adaptation_steps']}"
            plt.plot(result.metrics_dataframe["episodes_total"], result.metrics_dataframe["episode_reward_mean"], label=label)
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Reward')
        plt.legend()
        plt.show()

