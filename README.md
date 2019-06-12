<p align="center">
  <img src="docs/repo_logo.png">
</p>

<p align="center">
  Learn to segment images using only bounding box annotations!
</p>

## Features


The following package provides a framework to easily apply weakly annotated learning for segmentation.


#### Current limitations:
- Only works on greyscale images
- Fixed image sizes only

#### Dependencies

- OpenCV
- Keras with Tensorflow Backend

## Setup

#### Installation
To install the package above, pleae run:
```shell
pip install -r requiremnts
```

Add learn2seg to PYTHONPATH
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/learn2seg
```

## Usage

#### Data Structure
- train/val/test
- image/label
