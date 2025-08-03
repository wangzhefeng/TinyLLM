# !/bin/bash

echo "--------------------------"
echo "update TinyLLM codes..."
echo "--------------------------"
git checkout main
echo "Successfully checked out main."

git add .
git commit -m "update tf codes"

git pull
echo "Successfully pulled the latest changes."

git push
echo "Successfully checked out master and updated the code."

# push utils
echo "--------------------------"
echo "update utils codes..."
echo "--------------------------"
cd utils
pwd

git add .
git commit -m "update"

git pull
echo "Successfully pulled the latest changes."

git push
echo "Successfully checked out master and updated the code."


# push tokenizers
echo "--------------------------"
echo "update tokenizer codes..."
echo "--------------------------"
cd ../layers/tokenizers
pwd

git add .
git commit -m "update"

git pull
echo "Successfully pulled the latest changes."

git push
echo "Successfully checked out master and updated the code."
