# 配置环境

## verl-agent 环境配置

```
conda create -n verl-agent python=3.12 -y
conda activate verl-agent

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation

pip3 install -e .

pip3 install vllm==0.8.5
```

注意在安装 vllm 时会报错：

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
opentelemetry-exporter-prometheus 0.59b0 requires opentelemetry-sdk~=1.38.0, but you have opentelemetry-sdk 1.26.0 which is incompatible.
```

如果运行 `pip install -U "opentelemetry-sdk~=1.38.0"` 会报更多错误，暂时忽略不管。

## ALFWorld Benchmark 环境配置

首先运行下面的代码以配置环境：

```
conda create -n alfworld python=3.12 -y
conda activate alfworld

pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
pip install vllm==0.8.5
```

然后运行下面的代码以下载 PDDL 和游戏文件以及预训练的 MaskRCNN 检测器（将存储在 ~/.cache/alfworld/ 目录下）：


```
alfworld-download -f
```

注意，将上面的包都安装在同一个环境中，然后运行 `bash examples/grpo_trainer/run_alfworld.sh` 会遇到报错：

```
Traceback (most recent call last):
  File "/home/zyc/songzijun/verl-agent-master/verl/trainer/main_ppo.py", line 29, in main
    run_ppo(config)
  File "/home/zyc/songzijun/verl-agent-master/verl/trainer/main_ppo.py", line 41, in run_ppo
    ray.get(runner.run.remote(config))
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/ray/_private/worker.py", line 2882, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/ray/_private/worker.py", line 968, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ModuleNotFoundError): ray::TaskRunner.run() (pid=3563827, ip=172.18.132.17, actor_id=73f57c3eb9a956525801c2a901000000, repr=<main_ppo.TaskRunner object at 0x71c8bd7bec00>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zyc/songzijun/verl-agent-master/verl/trainer/main_ppo.py", line 82, in run
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
  File "/home/zyc/songzijun/verl-agent-master/verl/workers/fsdp_workers.py", line 63, in <module>
    from peft import LoraConfig, TaskType, get_peft_model
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/peft/__init__.py", line 17, in <module>
    from .auto import (
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/peft/auto.py", line 32, in <module>
    from .peft_model import (
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/peft/peft_model.py", line 42, in <module>
    from peft.tuners.lora.variants import get_alora_offsets_for_forward, get_alora_offsets_for_generate
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/peft/tuners/__init__.py", line 15, in <module>
    from .adalora import AdaLoraConfig, AdaLoraModel
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/peft/tuners/adalora/__init__.py", line 18, in <module>
    from .config import AdaLoraConfig
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/peft/tuners/adalora/config.py", line 19, in <module>
    from peft.tuners.lora import LoraConfig
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/peft/tuners/lora/__init__.py", line 23, in <module>
    from .model import LoraModel
  File "/home/zyc/miniconda3/envs/verl-agent2/lib/python3.12/site-packages/peft/tuners/lora/model.py", line 26, in <module>
    from transformers.modeling_layers import GradientCheckpointingLayer
ModuleNotFoundError: No module named 'transformers.modeling_layers'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

运行 `pip install transformers==4.53.1` 可以正常运行。

## WebShop Benchmark 环境配置
首先运行下面的代码以配置环境：

```
conda create -n webshop python==3.10 -y
conda activate webshop
```

```
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

```
cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
```

# Run Examples
## RL Training
### GRPO
```
conda activate verl-agent3
cd songzijun/verl-agent
nohup bash examples/grpo_trainer/run_alfworld.sh > alfworld-Qwen3-8B-SFT.log 2>&1 &

nohup bash examples/grpo_trainer/run_alfworld_debug.sh > alfworld_debug.log 2>&1 &


conda activate verl-agent3
cd songzijun/verl-agent-master
nohup bash examples/grpo_trainer/run_alfworld_qwen3_4b.sh > alfworld-Qwen3-4B.log 2>&1 &
nohup bash examples/grpo_trainer/run_alfworld_qwen3_4b_sft.sh > alfworld-Qwen3-4B-SFT.log 2>&1 &

```