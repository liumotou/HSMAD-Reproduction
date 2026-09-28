import os
import time
import yaml
import matplotlib as mpl
import datetime

import matplotlib.pyplot as plt

class Logger():


    def __init__(self, opt, name):
        self.opt = opt
        self.name = name
        self.saved = False

        if not os.path.exists(self.name):
            os.makedirs(self.name)
        self.log_name = os.path.join(self.name, 'loss_log.txt')
        self.metrics_name = os.path.join(self.name, 'metrics.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Loss (%s) ================\n' % now)
        with open(self.metrics_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Metric (%s) ================\n' % now)

        with open(os.path.join(self.name, 'config.yaml'), 'w') as f:
            yaml.dump(vars(opt), f, default_flow_style=False)


    def reset(self):
        self.saved = False

    def print_current_losses(self, epoch, iters, losses, t_comp):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, iters, t_comp)
        for k, v in losses.items():
            message += '%s: %.5f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_metrics(self, epoch, iters, metrics, t_val):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, iters, t_val)
        for k, v in metrics.items():
            message += '%s: %.5f ' % (k, v)

        print(message)
        with open(self.metrics_name, "a") as log_file:
            log_file.write('%s\n' % message)