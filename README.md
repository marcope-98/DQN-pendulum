# Deep Q Network on a single/double pendulum

Deep Q Network implementation on a single and double pendulum swing-up manoeuvre. 

The code in this repository was submitted in partial fulfillment of the requirements for the course '145873 - Optimisation-based Robot Control'. 

## Dependencies

```bash
# Installation of required packages
$ sudo apt install python3-numpy python3-scipy python3-matplotlib curl
# Set up openrobots package repository
$ sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
$ sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
$ curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
# update apt repositories
sudo apt update
```

On Ubuntu 18.04 install these packages:
```console
$ sudo apt install robotpkg-py36-pinocchio robotpkg-py36-example-robot-data robotpkg-urdfdom robotpkg-py36-qt4-gepetto-viewer-corba robotpkg-osg-dae robotpkg-py36-quadprog robotpkg-py36-tsid
```

On Ubuntu 20.04 install these packages:
```console
$ sudo apt install robotpkg-py38-pinocchio robotpkg-py38-example-robot-data robotpkg-urdfdom robotpkg-py38-qt5-gepetto-viewer-corba robotpkg-py38-quadprog robotpkg-py38-tsid
```

On Ubuntu 22.04 install these packages:
```console
$ sudo apt install robotpkg-py310-pinocchio robotpkg-py310-example-robot-data robotpkg-urdfdom robotpkg-py310-qt5-gepetto-viewer-corba robotpkg-py310-quadprog robotpkg-py310-tsid
```
Configure the environment variables by adding the following lines to your `~/.bashrc` file:
```bash
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export ROS_PACKAGE_PATH=/opt/openrobots/share
# Depending on your Ubuntu distribution uncomment one of these lines:
# export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.10/site-packages # Ubuntu 18.04
# export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.10/site-packages # Ubuntu 20.04
# export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.10/site-packages # Ubuntu 22.04
```

The repository depends on `pytorch`, follow the instructions provided on [this website](https://pytorch.org/get-started/locally/) to install it on your system.

## Repository Structure
~~~~~
  ./
  ├── env/
  │   ├── __init__.py
  │   ├── cdPendulum.py
  │   ├── conf.py
  │   ├── display.py
  │   ├── Network.py
  │   └── pendulum.py
  │
  ├── resources/
  │   └── ...
  │
  ├── results/
  │   ├── csv/
  │   │   ├── cost_to_go.csv
  │   │   ├── decay.csv
  │   │   └── loss.csv
  │   └── model/
  │       └── ...
  │
  ├── DQNAgent.py
  ├── README.md
  ├── report.pdf
  └── test.py
~~~~~

- The `./env` folder contains the implementation of classes for the Neural Network, the pendulum environment, the configuration file with the hyperparameters for the network and a `diplay` class for visualization.
- The `./resources` folder contains papers on Deep Q Network.
- The `./results` folder (as well as the `csv` and `model` subfolders) needs to be created before running the script as they are initially empty.
- The `DQNAgent.py` performs the training step of the Deep Q Network.
- The `test.py` performs the testing step of the trained Deep Q Network and displays relevant plots and metrics.
- The `report.pdf` is the report of the Final Assignment with a detail explanation of the project and theory on Deep Q Networks.

## Execution
Before executing the script create the `./results` folder and its subfolders
```console
$ mkdir -p ./results/csv ./results/model
```

Open two terminals:
- On the first terminal run the gepetto gui interface that is used for visualization of the double pendulum
```console
$ gepetto-gui
```
- On the second terminal execute the `DQNAgent.py` or the `test.py` script
```console
$ python3 ./DQNAgent.py
<!-- or -->
$ python3 ./test.py
```

**Important**: the file `test.py` requires you to choose a model, edit the filepath in the script to one of the models in the `./results/model` subfolder.

**Important**: to switch between the single and double pendulum configuration edit the file `./env/conf.py` by settings the if condition to `True` for the single pendulum or to `False` for the double pendulum configuration. 

## Results

![](https://github.com/marcope-98/DQN-pendulum/blob/master/media/result.gif)