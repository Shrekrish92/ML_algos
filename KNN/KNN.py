import numpy as np

class KNN:
    def __init__(self) -> None:
        self.x=None
        self.y=None
    def train(self, x, y):
        self.x=x
        self.y=y
    def distance(self, a, b):
        return np.linalg.norm(a - b)
    def predict(self, x, k):
        distance_label=[(self.distance(x,train_points), train_label)for train_points, train_label in zip(self.x, self.y)]
        neighbors=sorted(distance_label)[:k]
        print(neighbors)
        return sum(label for _, label in neighbors)/k

knn=KNN()

train_x=np.array([[1,2],[3,4],[9,5]])
train_y=np.array([0,1,0])

knn.train(train_x,train_y)
test_x=np.array([2.3, 7.3])

k=2

prediction=knn.predict(test_x,k)

print("predicted table: ", prediction)
    
