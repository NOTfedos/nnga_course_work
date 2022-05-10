import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp


def plot_env_history(env):
    color_list = mcp.gen_color(cmap="winter", n=env.entity_count)

    x = list(range(1, env.evo_epochs+1))

    for i, entity in enumerate(env.entities):
        plt.plot(x, entity.entity_history, label=i+1)  # color=color_list[entity.color], label=entity.color)

    plt.legend()
    plt.show()


def plot_ents_results(env, x, y):

    ents = env.entities[:4]

    fig, axes = plt.subplots(2, 2)

    ax11 = axes[0][0]
    ax11.plot(x, y, label="True")
    y_p = ents[0].predict(x.unsqueeze(1))
    ax11.plot(x, y_p.squeeze(1), label="Model")
    ax11.legend()

    ax12 = axes[0][1]
    ax12.plot(x, y, label="True")
    y_p = ents[1].predict(x.unsqueeze(1))
    ax12.plot(x, y_p.squeeze(1), label="Model")
    ax12.legend()

    ax21 = axes[1][0]
    ax21.plot(x, y, label="True")
    y_p = ents[2].predict(x.unsqueeze(1))
    ax21.plot(x, y_p.squeeze(1), label="Model")
    ax21.legend()

    ax22 = axes[1][1]
    ax22.plot(x, y, label="True")
    y_p = ents[3].predict(x.unsqueeze(1))
    ax22.plot(x, y_p.squeeze(1), label="Model")
    ax22.legend()

    plt.show()
