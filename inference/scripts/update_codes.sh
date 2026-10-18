#!/bin/bash

# 运行方式：
# 仅 pull: bash scripts/update_codes.sh pull
# 要 push: bash scripts/update_codes.sh push

echo "--------------------------"
echo "update TinyLLMFinetuning codes..."
echo "--------------------------"
git checkout main
echo "Successfully checked out main."

if [ $1 == "push" ]; then
    git add .
    git commit -m "update tf codes"
fi

git pull
echo "Successfully pulled the latest changes."

if [ $1 == "push" ]; then
    git push
    echo "Successfully checked out master and updated the code."
fi

# push tokenizers
# echo "--------------------------"
# echo "update tokenizers codes..."
# echo "--------------------------"
# cd layers/tokenizers

# if [ $1 == "push" ]; then
#     git add .
#     git commit -m "update"
# fi

# git pull
# echo "Successfully pulled the latest changes."

# if [ $1 == "push" ]; then
#     git push
#     echo "Successfully checked out master and updated the code."
# fi

# push utils
echo "--------------------------"
echo "update utils codes..."
echo "--------------------------"
cd utils


if [ $1 == "push" ]; then
    git add .
    git commit -m "update"
fi

git pull
echo "Successfully pulled the latest changes."

if [ $1 == "push" ]; then
    git push
    echo "Successfully checked out master and updated the code."
fi
