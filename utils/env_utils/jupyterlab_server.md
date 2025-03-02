

# 安装 jupyterlab 库

```bash
$ pip install jupyterlab
```

# 搭建远程 jupyterlab 服务

由于 GPU 算力通常在远程服务器，而开发是本地笔记本，因此，
需要搭建远程 jupyterlab 服务，然后在本地访问，具体操作如下所示。

1. 首先，生成配置文件

> `jupyter_lab_config.py`

```bash
$ jupyter lab --generate-config
```

2. 然后，对密码进行加密

```python
from jupyter_server.auth
import passwd

passwd()
```

3. 之后，修改生成的配置文件 `jupyter_lab_config.py`

```python
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_remote_access = True
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.open_browser = False  
c.ServerApp.password = '加密后的密码'
c.ServerApp.port = 9999
```

4. 最后，在服务器启动 jupyterlab 服务

```bash
$ jupyter lab --allow-root
$ nphup jupyter lab --allow-root > jupyterlab.log 2>&1 &
```
