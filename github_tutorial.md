# Github教程
## 初始设置
- 设置姓名和邮箱地址, 提高可读性
```
git config --global user.name "Firstname Lasname"
git config --global user.email "your_email@example.com"
git config --global color.ui auto
```
会在"~/.gitconfig" 中以如下形式输出设置文件
```
[user]
  name = Firstname Lastname
  email = your_email@example.com
[color]
  ui = auto
```

- 设置SSH Key
```
ssh-keygen -t rsa -C "your_email@example.com"
回车确认保存路径
输入密码
再次输入密码
```
用`cat ~/.ssh/id_rsa.pub`查看公开秘钥内容, 粘贴到GITHUB网页上"Add SSH Key"中.

之后可以用私人秘钥与GITHUB进行认证和通信
```
ssh -T git@github.com
```

## 基本操作

- clone已有仓库
```
git clone git@github:com*************
```

- 初始化仓库
```
mkdir git-tutorial
cd git-tutorial
git init
```
生成.git目录

- 建立README.md文件作为管理对象
```
touch README.md
```
可编辑README.md (任何文件都可创建)

 - 提交
```
git add hello_world.py                              # 向暂存区在心中添加文件
git commit -m "add hello workld script by python"   # 保存仓库历史记录
```
```
git commit -am "add hello workld script by python"    # 合并简写
```
```
git log                # 查看提交日志
git log -p README.md   # 文件前后差别
git status             # 查看仓库状态  
git diff               # 查看工作树, 暂存区, 最新提交之间的差别
```

- 更新github仓库
```
git push
```

## 分支操作
- 显示分支一览表
```
git branch
```

- 创建切换分支
```
git checkout -b feature-A
```
```
git branch feature-A    # 创建feature-A
git checkout feature-A  # 切换到feature-A
```

- 合并分支
```
git checkout master
git merge --no-ff feature-A
```

- 以图表形式查看分支
```
git log --graph
```

## 更改提交的操作

- 回溯历史版本
```
git reflog  # 查看当前仓库执行过的操作的日志, 获得任务哈希值
git reset --hard 哈希值  # 回溯到指定哈希值对应的时间点
```

## 推送至远程仓库

本地仓库与远程仓库名一致

- 添加远程仓库
```
git remote add origin git@github.com:************
```
Git自动将远程仓库的名称设置为origin(标识符)

- 将本地仓库中的内容推送至远程仓库
```
git push -u origin master # 推送至远程仓库的master分支
```
```
git checkout -b feature-D    # 本地仓库创建分支feature-D
git push -u origin feature-D # 推送给远程仓库并保持分支名称不变
```

- 从远程仓库获取
```
git clone **********
git branch -a # 查看当前分支
git checkout -b feature-D origin/feature-D # 将feature-D分支获取至本地仓库, -b后面是本地仓库新建分支名称
```
```
git diff                        # 查看更改
git commit -am "add feature-D"  # 提交更改
git push                        # 推送feature-D分支
```

- 获取最新的远程仓库分支
```
git pull origin feature-D
```

- Tasklist语法
```
# 本月要做的任务
- [ ] 任务A
- [x] 任务B
- [ ] 任务C
```

- [ ] 任务A
- [x] 任务B
- [ ] 任务C

