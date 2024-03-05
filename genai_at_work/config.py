import os
from dotenv import dotenv_values
config = {
    **dotenv_values("env.shared"),  # load shared development variables
    **dotenv_values("env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}

import torch
from rich.console import Console 
from transformers.utils import is_flash_attn_2_available
from dotenv import dotenv_values
use_flash_attention_2 = is_flash_attn_2_available()

system_cpus_count = os.cpu_count()
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    #torch.manual_seed(10)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(int(system_cpus_count/2))
except Exception as e:
    print(e)
