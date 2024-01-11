import matplotlib.pyplot as plt

def line_chart(x_list:list, y_list:list, char_name, x_lable, y_label):
    plt.figure()

    plt.plot(x_list, y_list,  linestyle='-', label={char_name})

    plt.xlabel(x_lable)
    plt.ylabel(y_label)

    plt.title('line chart')

    plt.legend()

    plt.show()