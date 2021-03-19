import requests
import keras
import telegram
import matplotlib.pyplot as plt

from io import BytesIO


class ParentCallback(keras.callbacks.Callback):

    def __init__(self,
                 modelName = 'model',
                 loss_metrics = ['loss'],
                 acc_metrics = [],
                 log_metrics  = [],
                 patience = 1,
                 getSummary = False,
                 app=None):
        self.modelName = modelName
        self.loss_metrics = loss_metrics
        self.acc_metrics = acc_metrics
        self.log_metrics = log_metrics
        self.patience = patience
        self.getSummary = getSummary
        self.app = app
        self.logs_arr = []

    def on_train_begin(self, logs=None):
        text = f"Hi! your *{self.modelName}* training has started."
        self.send_message(text)

    def on_train_end(self, logs=None):
        text = f"Your *{self.modelName}* has finished training."
        self.send_message(text)

        if self.getSummary:
            summary = self.make_summary(self.logs_arr)
            self.send_message(summary)

        if self.app in ['slack']:
            return

        if len(self.loss_metrics)>0:
            for metric in self.loss_metrics:
                plt.plot([epoch[metric] for epoch in self.logs_arr],
                         label=f'{metric}')
                plt.legend()

        out = BytesIO()
        plt.savefig(fname=out,format='png')
        out.seek(0)
        self.send_message(out, type='image')
        plt.clf()

        if len(self.acc_metrics)>0:
            for metric in self.acc_metrics:
                plt.plot([epoch[metric] for epoch in self.logs_arr],
                         label=f'{metric}')
                plt.legend()

        out = BytesIO()
        plt.savefig(fname=out,format='png')
        out.seek(0)
        self.send_message(out, type='image')

    def on_epoch_end(self, epoch, logs=None):
        self.logs_arr.append(logs)
        if (not self.getSummary) and ((epoch+1)%self.patience==0):
            text = f'`{"Epoch":13}: {epoch+1}`\n'
            for key, value in logs.items():
                if key in self.log_metrics:
                    text += f'`{key:13}: {value:.4f}`\n'
            self.send_message(text, type='text')

    def make_summary(self, logs_arr):
        summary = ''
        for epoch, log in enumerate(logs_arr):
            summary += f'\n`{"Epoch":13}: {epoch+1}`\n'
            for key, value in log.items():
                summary += f'`{key:13}*: {value:.4f}`\n'
        return summary


class TelegramCallback(ParentCallback):

    def __init__(self,
                 bot_token = None,
                 chat_id = None,
                 modelName = 'model',
                 loss_metrics = ['loss'],
                 acc_metrics = [],
                 log_metrics = [],
                 patience = 1,
                 getSummary=False):
        ParentCallback.__init__(self,
                                modelName,
                                loss_metrics,
                                acc_metrics,
                                log_metrics,
                                patience,
                                getSummary,
                                app='telegram')
        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id

    def send_message(self, message, type='text'):
        if type == 'text':
            response = self.bot.send_message(self.chat_id, message, parse_mode='Markdown')
        elif type == 'image':
            response = self.bot.send_photo(self.chat_id, photo=message)
