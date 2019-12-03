
from tensor_model import load_model, predict
from sys import argv

model = load_model('./simple_model.h5')

print(model.summary())

with open(argv[1]) as f:
    l = predict(model, f)
    print(l)
