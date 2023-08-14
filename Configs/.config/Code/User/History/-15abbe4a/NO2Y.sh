#!/bin/bash -x

# list of folders to copy
folders=( "Code" "hypr" "neofetch" "helix" "nvim" "rofi" "dunst" "gtk-3.0" "kitty" "nwg-look" "qt5ct" "swaylock" "swww" "waybar" "wlogout" "xsettingsd" "dolphinrc" "kdeglobals")
rm -r Configs/.config

# copy all config files from list
for folder in "${folders[@]}"
do
    cp -r ~/.config/$folder Configs/$folder
done

cp ~/.zshrc Configs/.zshrc