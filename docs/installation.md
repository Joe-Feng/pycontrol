## source code

```shell
sudo apt install git
git clone https://github.com/aikudexiaohai/pycontrol.git
```




## Dependencies

* **python packages**

  ```shell
  cd pycontrol && sudo pip install -r requirements.txt
  ```

  or add a mirror source

  ```shell
  cd pycontrol && sudo pip install -r requirements.txt -i https://pypi.douban.com/simple
  ```

  add pycontrol to environment

  ```shell
  sudo vi ~/.bashrc
  export PYTHONPATH=the parent dir of pycontrol:$PYTHONPATH
  source ~/.bashrc
  ```

  

* **Pangolin**     [https://github.com/stevenlovegrove/Pangolin](https://github.com/stevenlovegrove/Pangolin)

  ```shell
  sudo apt install cmake libgl1-mesa-dev libglew-dev libeigen3-dev
  cd pycontrol/3rdparty
  git clone https://github.com/stevenlovegrove/Pangolin.git
  cd Pangolin
  git submodule init && git submodule update
  mkdir build && cd build
  cmake ..
  make
  ```
