import matplotlib.pyplot as plt


def load_clean_data():
    '''
    Loading data from file and filling in the missing values
    :return: data[]
    '''

    # Read data from file into 2D array
    data = []
    with open("project1-hits.txt", mode="r") as fin:
        for line in fin:
            parts = line.strip().split(",")
            data.append(parts)

    # Convert to int and fill in missing data
    sum = 0
    count = 0
    nan_found = False
    for i in range(0, len(data), 1):
        hour = int(data[i][0])
        visits = data[i][1]
        if visits == 'nan':
            nan_found = True
            visits = -1
        else:
            visits = int(visits)
            sum += visits
            count += 1
        data[i] = [hour, visits]

        # For every 24 hours i.e. day, replace missing data (-1) with
        # mean visitor count for that day.
        if nan_found and data[i][0] % 24 == 0:
            mean = sum * 1.0 / count
            for j in range((i - 24 + 1), i + 1, 1):
                if data[j][1] == -1:
                    data[j][1] = mean
            sum = 0
            count = 0
            nan_found = False

    return data


def scatterplot_data(data):
    '''
    Scatter Plot using the data
    :param data: visitor count data
    '''

    x_points = [row[0] for row in data]
    y_points = [row[1] for row in data]
    plt.scatter(x_points, y_points, 5)
    plt.title("Scatter-plot of Visitor counts")
    plt.xlabel("Hour")
    plt.ylabel("Number of Visitors")
    plt.show()

    x_points = [i for i in range(1, 32, 1)]
    y_points = []
    i = 0
    while i < len(data):
        sum = 0
        count = 0
        for j in range(i, i+24, 1):
            sum += data[j][1]
            count += 1
        y_points.append(sum*1.0/count)
        i += 24

    plt.scatter(x_points, y_points, 5)
    plt.title("Scatter-plot of mean visitor counts per day")
    plt.xlabel("Days")
    plt.ylabel("Mean Number of Visitors")
    plt.show()


def linear_regression(data):
    '''
    Linear Regression over Scatter plot
    using sum squared error equations for linear regression.
    :param data: visitor count data
    '''

    sum_X = sum_Y = sum_XY = sum_X2 = sum_Y2 = 0.0
    N = len(data)
    for row in data:
        sum_X += row[0]
        sum_Y += row[1]
        sum_XY += (row[0] * row[1])
        sum_X2 += (row[0] ** 2)
        sum_Y2 += (row[1] ** 2)

    slope = (N * sum_XY - sum_X * sum_Y) / (N * sum_X2 - (sum_X ** 2))
    intercept = (sum_Y - slope * sum_X) / N

    print(f"Equation: y = {intercept:.2f} + {slope:.2f} * X")

    # Predict values for linear regression line.
    lr_x_points = [data[0][0], data[1][0], data[2][0], data[-3][0], data[-2][0], data[-1][0]]
    lr_y_points = [
        (intercept + slope * data[0][0]),
        (intercept + slope * data[1][0]),
        (intercept + slope * data[2][0]),
        (intercept + slope * data[-3][0]),
        (intercept + slope * data[-2][0]),
        (intercept + slope * data[-1][0])
    ]

    x_points = [row[0] for row in data]
    y_points = [row[1] for row in data]

    plt.plot(lr_x_points, lr_y_points, c="r")
    plt.scatter(x_points, y_points, 5)

    plt.title("Scatter-plot with Linear Regression line")
    plt.xlabel("Hour")
    plt.ylabel("Number of Visitors")

    plt.show()


def print_data(data):
    for i in range(0, len(data), 1):
        print(f"{data[i][0]}, {data[i][1]}")


def start():
    data = load_clean_data()
    scatterplot_data(data)
    linear_regression(data)
    # print_data(data)


if __name__ == '__main__':
    start()
