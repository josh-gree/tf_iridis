from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from numpy.random import permutation


class batch_gen:

    def __init__(self, data, target, batch_size):

        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.randomise = True

        self.mlb = MultiLabelBinarizer(classes=[i for i in range(10)])

    def __iter__(self):

        if self.randomise:
            perm = permutation(len(self))
            data = self.data[perm]
            target = self.target[perm]
            target = [(x,) for x in target]
            target = self.mlb.fit_transform(target)
        else:
            data = self.data
            target = self.target
            target = [(x,) for x in target]
            target = self.mlb.fit_transform(target)

        num_batches = (len(self) / self.batch_size)

        for i in range(num_batches):
            yield (
                data[i * self.batch_size:(i + 1) * self.batch_size, :],
                target[i * self.batch_size:(i + 1) * self.batch_size, :]
            )

    def __len__(self):

        return len(self.data)


def mnist_load():
    mnist = fetch_mldata('MNIST original', data_home="./data")

    data, target = mnist.data, mnist.target
    data_train, data_test, target_train, target_test = train_test_split(
        data, target, test_size=0.33, random_state=42)
    train_gen = batch_gen(data_train, target_train, 10)
    test_gen = batch_gen(data_test, target_test, 1000)

    return train_gen, test_gen


if __name__ == "__main__":
    mnist_load()
