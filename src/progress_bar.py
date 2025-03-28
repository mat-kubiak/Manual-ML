import time
from tqdm import tqdm

def to_color_code(color_name):
    color_map = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m'
    }

    # Default to reset
    return color_map.get(color_name.lower(), '\033[0m')

class DownloadProgressBar:
    def __init__(self, total_iters, color='green'):
        self.n_fmt_length = len(str(total_iters))

        self.ACCENT = to_color_code(color)
        self.RESET = to_color_code('reset')

        self.start_time = None

        bar_format=f"{self.ACCENT}{{bar:30}}{self.RESET} | {{n_fmt:>{self.n_fmt_length}}}/{{total_fmt}} ({{percentage:.1f}}%) ETA {{remaining}}"
        self.bar = tqdm(None, bar_format=bar_format, ascii='\u2500\u2501', total=total_iters)

    def update(self, progress):
        if self.start_time == None:
            self.start_time = time.perf_counter()
        self.bar.n = progress
        self.bar.refresh()

    def close(self):
        end = time.perf_counter()
        duration = end - self.start_time

        bar_completed = f'\u2501 COMPLETED ' + ''.join(['\u2501' for n in range(18)])

        bar_format=f"{self.ACCENT}{bar_completed}{self.RESET} | {{n_fmt:>{self.n_fmt_length}}}/{{total_fmt}} | {duration:.2f} s"
        self.bar.bar_format = bar_format
        self.bar.refresh()
        self.bar.close()

        return duration

def _format_num(value):
    if value < 0.01 or value > 100.0:
        return f'{value:.4e}'
    else:
        return f'{value:.4f}'

class ProgressBar:
    def __init__(self, total_iters, iter_name='batch', color='green'):
        self.n_fmt_length = len(str(total_iters))

        self.ACCENT = to_color_code(color)
        self.RESET = to_color_code('reset')

        self.iter_name = iter_name

        self.start_time = None

        bar_format=f"{self.ACCENT}{{bar:30}}{self.RESET} | {self.iter_name} {{n_fmt:>{self.n_fmt_length}}}/{{total_fmt}} ({{percentage:.1f}}%) ETA {{remaining}} | {{desc}}"
        self.bar = tqdm(None, bar_format=bar_format, ascii='\u2500\u2501', total=total_iters)

    def update(self, loss, metrics):
        if self.start_time == None:
            self.start_time = time.perf_counter()

        metrics_str = ''
        for m in metrics:
            metrics_str += f' | {m.get_name()}: {_format_num(m.get())}'

        self.bar.set_description_str(f"{self.ACCENT}loss: {_format_num(loss)}{self.RESET}{metrics_str} ")
        self.bar.update()

    def close(self):
        end = time.perf_counter()
        duration = end - self.start_time

        bar_completed = f'\u2501 COMPLETED ' + ''.join(['\u2501' for n in range(18)])

        bar_format=f"{self.ACCENT}{bar_completed}{self.RESET} | {self.iter_name} {{n_fmt:>{self.n_fmt_length}}}/{{total_fmt}} | {duration:.2f} s | {{desc}}"
        self.bar.bar_format = bar_format
        self.bar.refresh()
        self.bar.close()

        return duration
