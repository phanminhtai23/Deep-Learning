from ucimlrepo import fetch_ucirepo
from matplotlib import pyplot as plt

# Load dữ liệu iris từ UCI
iris = fetch_ucirepo(id=53)

Data = iris.data.original

# Biến đổi nhãn Iris-setosa -> 0, Iris-versicolor -> 1, Iris-virginica -> 2
Data = Data.replace(
    ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2])

# Xáo trộn dữ liệu và chia thành 2 tập train và test
Data = Data.sample(frac=1).reset_index(drop=True)

# Vẽ biểu đồ boxplot của từng đặc trưng theo nhãn (cột cuối cùng)
for i in range(Data.shape[1] - 1):
    plt.figure()
    Data.boxplot(column=Data.columns[i], by=Data.columns[-1])
    plt.xlabel('Class')
    plt.ylabel(Data.columns[i])
    plt.title(f'Boxplot of {Data.columns[i]} by Class')
    plt.suptitle('')
    plt.show()

