import atexit
import copy
import json
import os
import os.path as osp
from datetime import datetime
from typing import Any, Dict, List, NamedTuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string: str, color: str, bold: bool = False, highlight: bool = False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    if color is None:
        return string
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")

    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


class Logger:
    def __init__(self, base_path: str, parameter: NamedTuple, output_dir = 'log_file'):
        # read config
        self.base_path = base_path
        self.parameter = parameter
        self.__dict__.update(vars(self.parameter))

        self.record_params = [
            "seed",
            "exec_time",
            # "env_type",
            "rnn_fix_length",
            "ep_dim",
            "description",
        ]

        self.exec_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.output_dir = os.path.join(os.getcwd(), output_dir, self.parameter.env_type, self.exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = open(os.path.join(self.output_dir, "log.txt"), "w")
        self.output_file = open(osp.join(self.output_dir, "progress.txt"), "w")
        atexit.register(self.output_file.close)
        self.save_config()
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.log_last_row = None
        self.tb_x_label = None
        self.tb_x = None
        self.step = 0
        self.current_data: Dict[str, List] = dict()
        self.logged_data = set()
        self.model_output_dir = osp.join(self.output_dir, "model")
        os.makedirs(self.model_output_dir, exist_ok=True)
        self.tb = SummaryWriter(self.output_dir)
        self.tb_header_dict: Dict = dict()

    def log(self, *args, color: bool = None, bold: bool = True):
        s = ""
        for item in args[:-1]:
            s += str(item) + " "
        s += str(args[-1])
        print(colorize(s, color, bold=bold))
        if self.log_file is not None:
            print(s, file=self.log_file)
            self.log_file.flush()

    def save_config(self):
        with open(os.path.join(self.output_dir, "parameter.json"), "w") as f:
            ser = json.dumps(vars(self.parameter))
            f.write(ser)

    @property
    def exp_name(self):
        name = ""
        for item in self.record_params:
            value = getattr(self, item)
            name += f"_{item}:{value}"
        return name

    def backup_code(self):
        things = []
        for item in os.listdir(self.base_path):
            p = os.path.join(self.base_path, item)
            if (
                not item.startswith(".")
                and not item.startswith("__")
                and not item == "log_file"
            ):
                things.append(p)
        code_path = os.path.join(self.output_dir, "codes")
        os.makedirs(code_path, exist_ok=True)
        for item in things:
            os.system(f"cp -r {item} {code_path}")

    def log_dict(self, color: bool = None, bold: bool = False, **kwargs):
        for k, v in kwargs.items():
            self.log("{}: {}".format(k, v), color=color, bold=bold)

    def log_dict_single(self, data: Dict, color: bool = None, bold: bool = False):
        for k, v in data.items():
            self.log("{}: {}".format(k, v), color=color, bold=bold)

    def __call__(self, *args, **kwargs):
        self.log(*args, **kwargs)

    def log_tabular(
        self,
        key: str,
        val: Any = None,
        tb_prefix: str = None,
        with_min_and_max: bool = False,
        average_only: bool = False,
        no_tb: bool = False,
    ):
        def _log_tabular(key: str, val: Any, tb_prefix: str = None, no_tb: str = False):
            if self.tb_x_label is not None and key == self.tb_x_label:
                self.tb_x = val
            if self.first_row:
                self.log_headers.append(key)
                self.log_headers = sorted(self.log_headers)
            else:
                assert key in self.log_headers, (
                    "Trying to introduce a new key %s that you didn't include in the first iteration"
                    % key
                )
            assert key not in self.log_current_row, (
                "You already set %s this iteration. Maybe you forgot to call dump_tabular()"
                % key
            )
            self.log_current_row[key] = val
            if tb_prefix is None:
                tb_prefix = "tb"
            if not no_tb:
                if self.tb_x_label is None:
                    self.tb.add_scalar(f"{tb_prefix}/{key}", val, self.step)
                else:
                    self.tb.add_scalar(f"{tb_prefix}/{key}", val, self.tb_x)

        if val is not None:
            _log_tabular(key, val, tb_prefix, no_tb=no_tb)
        else:
            if key in self.current_data:
                self.logged_data.add(key)
                _log_tabular(
                    key if average_only else "Average" + key,
                    np.mean(self.current_data[key]),
                    tb_prefix,
                    no_tb=no_tb,
                )
                if not average_only:
                    _log_tabular(
                        "Std" + key,
                        np.std(self.current_data[key]),
                        tb_prefix,
                        no_tb=no_tb,
                    )
                    if with_min_and_max:
                        _log_tabular(
                            "Min" + key,
                            np.min(self.current_data[key]),
                            tb_prefix,
                            no_tb=no_tb,
                        )
                        _log_tabular(
                            "Max" + key,
                            np.max(self.current_data[key]),
                            tb_prefix,
                            no_tb=no_tb,
                        )

    def append_key(self, d: Dict[str, Any], tail: str):
        res = {}
        for k, v in d.items():
            res[k + tail] = v
        return res

    def add_tabular_data(self, tb_prefix: str = None, **kwargs):
        for k, v in kwargs.items():
            if tb_prefix is not None and k not in self.tb_header_dict:
                self.tb_header_dict[k] = tb_prefix
            if k not in self.current_data:
                self.current_data[k] = []
            if not isinstance(v, list):
                self.current_data[k].append(v)
            else:
                self.current_data[k] += v

    def update_tb_header_dict(self, tb_header_dict: Dict):
        self.tb_header_dict.update(tb_header_dict)

    def dump_tabular(self, write : bool = True):
        for k in self.current_data:
            if k not in self.logged_data:
                if k in self.tb_header_dict:
                    self.log_tabular(
                        k, tb_prefix=self.tb_header_dict[k], average_only=True
                    )
                else:
                    self.log_tabular(k, average_only=True)
        self.logged_data.clear()
        self.current_data.clear()

        # dump data
        if self.first_row:
            self.log_last_row = self.log_current_row
        for k, v in self.log_last_row.items():
            if k not in self.log_current_row:
                self.log_current_row[k] = v
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        if len(key_lens) > 0:
            max_key_len = max(15, max(key_lens))
        else:
            max_key_len = 15
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        head_indication = f" iter {self.step} "
        bar_num = n_slashes - len(head_indication)
        if write:
            self.log("-" * (bar_num // 2) + head_indication + "-" * (bar_num // 2))
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g" % val if hasattr(val, "__float__") else val
                self.log(fmt % (key, valstr))
                vals.append(val)
            self.log("-" * n_slashes + "\n")
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers) + "\n")
                self.output_file.write("\t".join(map(str, vals)) + "\n")
                self.output_file.flush()
        self.log_last_row = copy.deepcopy(self.log_current_row)
        self.log_current_row.clear()
        self.first_row = False
        self.step += 1

class TBLogger:
    def __init__(self, log_path, env_name, changing_para, time_dir, seed, info_str = "", warning_level = 3, print_to_terminal = True):
        unique_path = self.make_simple_log_path(info_str, seed)
        log_path = os.path.join(log_path, env_name, changing_para, time_dir, unique_path)
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.tb_writer = SummaryWriter(log_path)
        self.log_file_path = os.path.join(log_path, "logs.txt")
        self.print_to_terminal = print_to_terminal
        self.warning_level = warning_level
        self.log_str("logging to {}".format(self.log_path))

    def make_simple_log_path(self, info_str, seed):
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d-%H-%M")
        pid_str = os.getpid()
        if info_str != "":
            return "{}-{}-{}_{}".format(time_str, seed, pid_str, info_str)
        else:
            return "{}-{}-{}".format(time_str, seed, pid_str)

    @property
    def log_dir(self):
        return self.log_path

    def log_str(self, content, level = 4):
        if level < self.warning_level:
            return
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.print_to_terminal:
            print("\033[32m{}\033[0m:\t{}".format(time_str, content))
        with open(self.log_file_path, 'a+') as f:
            f.write("{}:\t{}\n".format(time_str, content))

    def log_var(self, name, val, timestamp):
        self.tb_writer.add_scalar(name, val, timestamp)
    
    def log_fig(self, name, fig, timestep):
        self.tb_writer.add_figure(name, fig, timestep)

    def log_str_object(self, name: str, log_dict: dict = None, log_str: str = None):
        if log_dict != None:
            log_str = json.dumps(log_dict, indent = 4)
        elif log_str != None:
            pass
        else:
            assert 0
        if name[-4:] != ".txt":
            name += ".txt"
        target_path = os.path.join(self.log_path, name)
        with open(target_path, 'w+') as f:
            f.write(log_str)
        self.log_str("saved {} to {}".format(name, target_path))
    
    def save_config(self, config : Dict, config_name : str = 'config.json'):
        config_dict = config.copy()
        for key in config_dict.keys():
            config_dict[key] = str(config_dict[key])
        config_json = json.dumps(config_dict, sort_keys = False, indent = 4, separators = (',', ': '))
        with open(os.path.join(self.log_dir, config_name), 'w') as f:
            f.write(config_json)
        f.close()
