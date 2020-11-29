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
