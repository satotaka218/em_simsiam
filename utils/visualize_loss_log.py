import matplotlib.pyplot as plt
import numpy as np
save_path = '../result_figure/'


class plot_result :
    def __init__(self, loss_train_log: np.ndarray, loss_eval_log: np.ndarray, acc_train_log: np.ndarray, acc_eval_log: np.ndarray) :
        # 有効数字の桁を揃える
        self._loss_train_log = np.round(loss_train_log, 2).tolist()
        self._loss_eval_log = np.round(loss_eval_log, 2).tolist()
        self._acc_train_log = np.round(acc_train_log, 2).tolist()
        self._acc_eval_log = np.round(acc_eval_log, 2).tolist()

    def plot(self, figure_name: str) :
        fig = plt.figure()
        plt.subplots_adjust(wspace=0.5, hspace=0.7)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot([i for i in range(len(self._loss_train_log))], self._loss_train_log, label = "train loss log")
        ax1.plot([i for i in range(len(self._loss_eval_log))], self._loss_eval_log, label = "eval loss log")
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("loss")
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot([i for i in range(len(self._acc_train_log))], self._acc_train_log, label = "train acc log")
        ax2.plot([i for i in range(len(self._acc_eval_log))], self._acc_eval_log, label = "eval acc log")
        ax2.set_xlabel("epochs")
        ax2.set_ylabel("acc")
        ax2.legend()

        fig.savefig('./result_figure/' + figure_name +'.png')

        plt.show()
