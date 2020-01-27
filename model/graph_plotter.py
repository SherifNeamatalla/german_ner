import matplotlib.pyplot as plt


def plot_graphs(validation_metrics, training_metrics, path):
    print("Plotting graphs")
    processed_validation_metrics = extract_one_metric_information(validation_metrics)
    processed_training_metrics = extract_one_metric_information(training_metrics)

    for x in processed_validation_metrics:
        plot_graph(x, processed_validation_metrics[x], processed_training_metrics[x], len(validation_metrics), path)
        save_array(x, processed_validation_metrics[x], processed_training_metrics[x], len(validation_metrics), path)


def plot_graph(metric_name, validation_metrics, training_metrics, nepochs, path):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel("epochs")
    ax.set_ylabel(str(metric_name).lower())
    ax.plot(range(1, nepochs + 1), validation_metrics, 'b-', label='validation')
    ax.plot(range(1, nepochs + 1), training_metrics, 'r-', label='training')
    ax.legend(loc="upper left")
    fig.savefig(path + "/graphs/" + str(metric_name).lower() + '_matplot')
    plt.close(fig)


def save_array(metric_name, validation_metrics, training_metrics, nepochs, path):
    new_file = open(path + "/graphs/" + str(metric_name).lower() + 'data', 'w+')
    validation_data = extract_data(validation_metrics)
    training_data = extract_data(training_metrics)
    total_data = validation_data + training_data
    new_file.write(total_data)
    new_file.close()


def extract_data(metrics):
    result = ''
    for value in metrics:
        result += str(value) + " "
    result += '\n'
    return result


def extract_one_metric_information(metrics_list):
    result = dict()
    for epoch_entry in metrics_list:
        for key in epoch_entry:
            if key in result:
                old_list = result[key]
                old_list.append(epoch_entry[key])
                result.update({key: old_list})
            else:
                result.update({key: [epoch_entry[key]]})
    return result
