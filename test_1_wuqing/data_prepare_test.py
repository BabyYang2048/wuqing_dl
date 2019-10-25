import random
import pickle

data = []

with open("../data/result_test.txt", encoding='utf-8') as f:
    for line in f:
        list = line.split("||")
        sen1 = list[0]
        sen2 = list[1]
        label = list[2]
        data.append((sen1, sen2, int(label)))


random.shuffle(data)

pickle.dump(data,open("../data/test_data.pkl","wb"))