# PaddlePaddle for Iluvatar GPU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify paddlepaddle_iluvatar.

## Compilation and Installation

```bash
# Please contact Iluvatar customer support (services@iluvatar.com) to obtain the SDK image

# Clone PaddleCustomDevice source code
git clone https://github.com/PaddlePaddle/Paddle_Iluvatar.git

bash build_paddle.sh

# Install
pip install Paddle/build/python/dist/paddlepaddle*
```
## For incremental compilation（faster rebuilds after code changes）
```bash
# For incremental compilation (faster rebuilds after code changes, also installs whl)
cd Paddle/build
ninja -j$(nproc)
```

## Verification

```bash
# Run tests
cd tests
bash run_test.sh
```
