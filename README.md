# Data-Centric Track

This is the official GitHub of the **Data-Centric Track** of the Wake Vision Challenge (TBD: link to the challenge).

It asks participants to **push the boundaries of tiny computer vision** by enhancing the data quality of the newly released [Wake Vision](https://wakevision.ai/), a person detection dataset.

Participants will be able to **modify** as they want the **provided data**. 

The quality increments will be assessed over a private dataset, after training the [mcunet-vww2 model](https://github.com/mit-han-lab/mcunet), a state-of-the-art model in person detection, on the resulting dataset.

## To Get Started

Install [docker engine](https://docs.docker.com/engine/install/).

### If you don't have a GPU 

Run the following command inside the directory in which you cloned this repository.

```
sudo docker run -it --rm -v $PWD:/tmp -w /tmp wake_vision_challenge/tensorflow python data_centric_track.py
```

It trains the [mcunet-vww2 model](https://github.com/mit-han-lab/mcunet) on the original dataset to get you started. 

Then you can modify the dataset to enhance the model's test accuracy, such as by correcting incorrect labels or augmenting the data. Just don't change the model architecture.

The first execution will require a lot of hours since it has to download the whole dataset on your machine (365 GB).

### If you have a GPU

Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (if you have problems, [check your GPU drivers](https://ubuntu.com/server/docs/nvidia-drivers-installation)).

Run the following command inside the directory in which you cloned this repository.

```
sudo docker run -it --rm -v $PWD:/tmp -w /tmp wake_vision_challenge/tensorflow-gpu python data_centric_track.py
```

It trains the [mcunet-vww2 model](https://github.com/mit-han-lab/mcunet) on the original dataset to get you started. 

Then you can modify the dataset to enhance the model's test accuracy, such as by correcting incorrect labels or augmenting the data. Just don't change the model architecture.

The first execution will require a lot of hours since it has to download the whole dataset on your machine (365 GB).
