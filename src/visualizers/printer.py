import time
from prettytable import PrettyTable


def print_table(data):
    """
    Prints the key and value of the dict (data) in a pretty table
    :param data: dict
    """
    table = PrettyTable()
    table.field_names = ['Parameter', 'Value']
    for key in data:
        table.add_row([key, data[key]])

    print(table)


def print_epoch_info(start_time, epoch_start_time, time_per_epoch, epoch, num_epochs, train_loss, validation_loss,
                     train_acc, validation_acc):
    total_epoch_time = time.time() - epoch_start_time
    time_per_epoch.append(total_epoch_time)
    total_time = time.time() - start_time
    avg_time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
    remaining_time = (num_epochs - epoch) * avg_time_per_epoch
    print('\n=== Epoch %d/%d ===' % (epoch + 1, num_epochs))
    print('Train Loss: %.3f' % train_loss)
    print('Validation Loss: %.3f' % validation_loss)
    print('Train accuracy: %f' % train_acc)
    print('Validation accuracy: %f' % validation_acc)
    print('Epoch duration: %s' % time.strftime('%H:%M:%S', time.gmtime(total_epoch_time)))
    print('Elapsed / Remaining time: %s/%s' % (
        time.strftime('%H:%M:%S', time.gmtime(total_time)), time.strftime('%H:%M:%S', time.gmtime(remaining_time))))
