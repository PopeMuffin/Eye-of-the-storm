from src.models.model import MeiboModel
from src.models.model1 import Model1
from src.utils.data import load_data

if __name__ == "__main__":
    train, validation, test = load_data()

    model: MeiboModel = Model1()

    model.compile()
    model.train(train, validation)
