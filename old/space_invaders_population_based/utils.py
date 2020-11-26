import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.optim as optim

def get_optimizer(model):
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    optimizer_class = optim.SGD
    lr = np.random.choice(np.logspace(-5, 0, base=10))
    momentum = np.random.choice(np.linspace(0.1, .9999))
    return optimizer_class(model.parameters(), lr=lr, momentum=momentum)


def exploit_and_explore(top_checkpoint_path, bot_checkpoint_path, hyper_params,
                        perturb_factors=(1.2, 0.8)):
    """Copy parameters from the better model and the hyperparameters
       and running averages from the corresponding optimizer."""
    # Copy model parameters
    checkpoint = torch.load(top_checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optim_state_dict']
    batch_size = checkpoint['batch_size']
    for hyperparam_name in hyper_params['optimizer']:
        perturb = np.random.choice(perturb_factors)
        for param_group in optimizer_state_dict['param_groups']:
            param_group[hyperparam_name] *= perturb
    if hyper_params['batch_size']:
        perturb = np.random.choice(perturb_factors)
        batch_size = int(np.ceil(perturb * batch_size))
    checkpoint = dict(model_state_dict=state_dict,
                      optim_state_dict=optimizer_state_dict,
                      batch_size=batch_size)
    torch.save(checkpoint, bot_checkpoint_path)


def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
