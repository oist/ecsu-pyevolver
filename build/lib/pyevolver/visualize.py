# TODO: move this to respective objects
import pandas as pd
import plotnine as p9
import matplotlib.pyplot as plt


def plot_brain_state(dict_df, filename):
    data_dim = dict_df['brain_state'].shape
    # convert to data frame
    df = pd.DataFrame({'t': list(range(data_dim[0]))})
    for n in range(data_dim[1]):
        var = 'n' + str(n + 1)
        df[var] = dict_df['brain_state'][:, n]
    # convert wide to long format
    df_long = pd.melt(df, id_vars=['t'], var_name='neuron', value_name='state')
    p = p9.ggplot(df_long, p9.aes('t', 'state', color='neuron')) + p9.geom_line()
    p.save(filename)


def plot_brain_output(dict_df, filename):
    data_dim = dict_df['brain_output'].shape
    # convert to data frame
    df = pd.DataFrame({'t': dict_df['t'][:, 0]})
    for n in range(data_dim[1]):
        var = 'n' + str(n + 1)
        df[var] = dict_df['brain_output'][:, n]
    # convert wide to long format
    df_long = pd.melt(df, id_vars=['t'], var_name='neuron', value_name='output')
    p = p9.ggplot(df_long, p9.aes('t', 'output', color='neuron')) + p9.geom_line()
    p.save(filename)


def plot_performances(evo):
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Agent Performances")

    plt.plot(evo.best_performances, label='Best')
    plt.plot(evo.avg_performances, label='Avg')
    plt.plot(evo.worst_performances, label='Worst')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
