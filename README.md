# Decoupled Knowledge Distillation with Cross-Entropy

## MDistiller

MDistiller is a PyTorch library that provides classical knowledge distillation algorithms on mainstream CV benchmarks.

## Installation

Environments:

- Python 3.6
- PyTorch 1.9.0
- torchvision 0.10.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
```

Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

## Execution
 ```bash
  # For DKD
  python3 tools/train.py --cfg configs/cifar100/dkd/{yaml_configuration_name}.yaml

  # For all other KD methods
  python3 tools/train.py --cfg configs/cifar100/{yaml_configuration_name}.yaml
  
  # For our loss function, e.g. loss_1
  python3 tools/train.py --cfg configs/cifar100/loss_1.yaml
  python3 tools/train.py --cfg configs/cifar100/loss_2.yaml
  ```

## Results

Execution results can be found in ./output
Project Poster: [LINK](https://drive.google.com/file/d/1SYEYSd0BlLx-URfty_nH8zEoztEKVMe1/view>)

# License

MDistiller is released under the MIT license. See [LICENSE](LICENSE) for details.

# Acknowledgement

Thanks to megvii-research for MDistiller
