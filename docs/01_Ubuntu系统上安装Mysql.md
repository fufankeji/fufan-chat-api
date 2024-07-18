## <center> Ubuntu 安装 Mysql 数据库

&emsp;&emsp;首先更新apt-get工具，执行命令如下：
```basah
apt-get upgrade
```

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604104800254.png" width=100%></div>

&emsp;&emsp;安装Mysql，执行如下命令：
```bash
apt-get install mysql-server
```

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604104721648.png" width=100%></div>

&emsp;&emsp;开启Mysql 服务，执行命令如下：
```bash
service mysql start
```

&emsp;&emsp;并确认是否成功开启mysql,执行命令如下：
```bash
service mysql status
```

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604104940068.png" width=100%></div>

&emsp;&emsp;确认是否启动成功，在LISTEN状态下，启动成功：
```bash
netstat -tap | grep mysql
```

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604105250306.png" width=100%></div>

&emsp;&emsp;在最新使用的 MySQL 版本中，默认使用 auth_socket 插件来进行身份验证，这意味着 root 用户通过操作系统的用户身份进行认证，而不是使用密码。这种配置在许多 Linux 系统上是默认的，特别是在安装 MySQL 时不要求设置密码的情况下。如果我们希望使用传统的密码验证方式来登录 MySQL，需要更改 root 用户的认证方式。以下是如何将 root 用户从 auth_socket 插件更改为使用密码认证的步骤：

1. **登录 MySQL**:
   首先，使用以下命令登录到 MySQL：
   ```bash
   sudo mysql
   ```

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604105909768.png" width=100%></div>

2. **更改认证插件和设置密码**:
   在 MySQL 命令行中，使用以下命令来更改 `root` 用户的认证插件并设置一个新密码：
   ```sql
   ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '新密码';
   FLUSH PRIVILEGES;
   ```
   把 `'新密码'` 替换为你想要设置的密码。

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604110024977.png" width=100%></div>

3. **退出并测试登录**:
   更改完成后，退出 MySQL：
   ```sql
   exit;
   ```

&emsp;&emsp;然后尝试使用新密码重新登录：
   ```bash
   sudo mysql -u root -p
   ```
&emsp;&emsp;系统将提示你输入密码，此时应输入你刚才设置的密码。

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604110332263.png" width=100%></div>

&emsp;&emsp;MySQL 配置文件中的 bind-address 参数限制了可以接受连接的 IP 地址。需要确认它是否设置为允许从你的客户端 IP 访问。查看 /etc/mysql/mysql.conf.d/mysqld.cnf 文件中的 bind-address：

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604111001209.png" width=100%></div>

&emsp;&emsp;如果设置为 127.0.0.1（只允许本地连接），需要改为 0.0.0.0（允许任何 IP 连接）或具体的外部 IP 地址，然后重启 MySQL 服务：

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604111201173.png" width=100%></div>

&emsp;&emsp;执行重启命令：
```bash
sudo systemctl restart mysql
```

&emsp;&emsp;再次登录Mysql，确保 MySQL 用户的主机设置允许从你的客户端 IP 地址连接。可以在 MySQL 中运行以下 SQL 命令来检查：

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604111336897.png" width=100%></div>

&emsp;&emsp;接下来，我们需要使用 root 用户或具有相应权限的用户登录到 MySQL。可以通过以下命令登录：
```bash
mysql -u root -p

```

&emsp;&emsp;登录后，需要选择 mysql 数据库，因为用户信息存储在这个数据库中：
```bash
USE mysql;

```

&emsp;&emsp;执行以下 SQL 命令来查看所有用户及其主机：
```bash
SELECT user, host FROM user;

```

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604113004560.png" width=100%></div>

&emsp;&emsp;决定修改现有用户（如 root 用户），可以更改用户的 host 值，以允许从任意 IP 地址连接。执行如下命令：
```bash
UPDATE user SET host = '%' WHERE user = 'root' AND host = 'localhost';
FLUSH PRIVILEGES

```

&emsp;&emsp;这里将 root 用户的 host 从 localhost 改为 %，表示从任何 IP 地址都允许连接。

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604113057311.png" width=100%></div>

&emsp;&emsp;Windows下载Navicat，地址：https://www.navicat.com/en/download/direct-download?product=navicat170_premium_en_x64.exe&location=1

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604110608705.png" width=100%></div>

&emsp;&emsp;选择Connection：

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604110722646.png" width=100%></div>

&emsp;&emsp;新建一个Mysql连接：

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604110741430.png" width=100%></div>

&emsp;&emsp;输入远程服务器的IP，Mysql的用户名和密码，执行连通性测试。

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604113157216.png" width=100%></div>

&emsp;&emsp;如何能够正常连接，会提示`Connection Successful`字样。

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604113216256.png" width=100%></div>

&emsp;&emsp;接下来就可以正常在Navinate工具中进行Mysql操作了。

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240604113230687.png" width=100%></div>
