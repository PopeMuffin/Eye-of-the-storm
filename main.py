from src.models.model import LEARN_RATE, MeiboModel
from src.models.model1 import Model1
from src.utils.data import load_data
from src.utils.history import log_evaluation, plot_confusion_matrix, show_history

BATCH_SIZE = 64

if __name__ == "__main__":
    train, validation, test = load_data()

    model: MeiboModel = Model1()

    model.compile()
    history = model.train(train, validation)

    results = model.evaluate(test)
    y_true, y_pred, _softmax_values = model.predict_dataset(test)

    show_history(history)
    plot_confusion_matrix(y_true, y_pred)
    log_evaluation(
        history, y_true, y_pred, results, batch_size=BATCH_SIZE, learn_rate=LEARN_RATE
    )
