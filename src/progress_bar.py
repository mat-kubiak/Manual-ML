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
    def __init__(self, iterator, iter_name='batch', color='green', total_iters=0):
        n_fmt_length = len(str(total_iters))

        self.ACCENT = to_color_code(color)
        self.RESET = to_color_code('reset')

        bar_format=f"{self.ACCENT}{{bar:30}}{self.RESET} | {iter_name} {{n_fmt:>{n_fmt_length}}}/{{total_fmt}} ({{percentage:.1f}}%) ETA {{remaining}} | {{desc}}"
        self.bar = tqdm(iterator, bar_format=bar_format, ascii='\u2500\u2501')

    def update_loss(self, loss):
        if loss < 0.01 or loss > 100.0:
            loss_str = f'{loss:.4e}'
        else:
            loss_str = f'{loss:.4f}'
        self.bar.set_description_str(f"{self.ACCENT}loss: {loss_str}{self.RESET} ")
