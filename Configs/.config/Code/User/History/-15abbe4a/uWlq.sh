#!/bin/sh -x

# list of folders to copy
files = (".zshrc" ".config")
rm -r Configs/.config
# copy all config files from list
for file in "${files[@]}"
do
    cp -r ~/$file Configs/$file
done

