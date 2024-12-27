# Model-Centric Track

This is the official GitHub of the **Model-Centric Track** of the Wake Vision Challenge (TBD: link to the website challenge).

It asks participants to **push the boundaries of tiny computer vision** by **innovating model architectures** to achieve **high test accuracy** while **minimizing resource usage**, leveraging the newly released [Wake Vision](https://wakevision.ai/), a person detection dataset.

Proposed model architectures will be evaluated over a **private test set**.

## To Get Started

Install [docker engine](https://docs.docker.com/engine/install/).

### If you don't have a GPU 

Run the following command inside the directory in which you cloned this repository.

```
sudo docker run -it --rm -v $PWD:/tmp -w /tmp wake_vision_challenge/tensorflow python model_centric_track.py
```

It trains the [ColabNAS](https://github.com/harvard-edge/Wake_Vision/blob/main/experiments/comprehensive_model_architecture_experiments/wake_vision_quality/k_8_c_5.py) model, a state-of-the-art person detection model, on the Wake Vision dataset to get you started. 

Then you can modify the "model_centric_track.py" script as you like, and propose your own model architecture.

The first execution will require a lot of hours since it has to download the whole dataset on your machine (365 GB). 

### If you have a GPU

Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (if you have problems, [check your GPU drivers](https://ubuntu.com/server/docs/nvidia-drivers-installation)).

Run the following command inside the directory in which you cloned this repository.

```
sudo docker run -it --rm -v $PWD:/tmp -w /tmp wake_vision_challenge/tensorflow-gpu python model_centric_track.py
```

It trains the [ColabNAS](https://github.com/harvard-edge/Wake_Vision/blob/main/experiments/comprehensive_model_architecture_experiments/wake_vision_quality/k_8_c_5.py) model, a state-of-the-art person detection model, on the Wake Vision dataset to get you started. 

Then you can modify the "model_centric_track.py" script as you like, and propose your own model architecture.

The first execution will require a lot of hours since it has to download the whole dataset on your machine (365 GB). 
