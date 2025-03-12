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

class ProgressBar:
    def __init__(self, total_iters, iter_name='batch', color='green'):
        self.n_fmt_length = len(str(total_iters))

        self.ACCENT = to_color_code(color)
        self.RESET = to_color_code('reset')

        self.iter_name = iter_name

        self.start_time = None

        bar_format=f"{self.ACCENT}{{bar:30}}{self.RESET} | {self.iter_name} {{n_fmt:>{self.n_fmt_length}}}/{{total_fmt}} ({{percentage:.1f}}%) ETA {{remaining}} | {{desc}}"
        self.bar = tqdm(None, bar_format=bar_format, ascii='\u2500\u2501', total=total_iters)

    def update(self, loss):
        if self.start_time == None:
            self.start_time = time.perf_counter()
        
        if loss < 0.01 or loss > 100.0:
            loss_str = f'{loss:.4e}'
        else:
            loss_str = f'{loss:.4f}'

        self.bar.set_description_str(f"{self.ACCENT}loss: {loss_str}{self.RESET} ")
        self.bar.update()

    def close(self):
        end = time.perf_counter()
        duration = end - self.start_time

        bar_format=f"{self.ACCENT}{{bar:30}}{self.RESET} | {self.iter_name} {{n_fmt:>{self.n_fmt_length}}}/{{total_fmt}} {duration:.2f} s | {{desc}}"
        self.bar.bar_format = bar_format
        self.bar.refresh()
        self.bar.close()
