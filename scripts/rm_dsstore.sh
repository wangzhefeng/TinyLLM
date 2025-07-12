# !/bin/bash
# https://github.com/nodejh/nodejh.github.io/issues/18

# 删除项目中的所有.DS_Store。这会跳过不在项目中的 .DS_Store
find . -name .DS_Store -print0 | xargs -0 git rm -f --ignore-unmatch
# 将 .DS_Store 加入到 .gitignore
echo .DS_Store >> ~/.gitignore
