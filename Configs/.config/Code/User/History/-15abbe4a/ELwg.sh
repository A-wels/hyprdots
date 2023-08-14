#!/bin/sh -x

# list of folders to copy
folders = (".zshrc" ".config" "Code" "hypr" "neofetch" "rofi" "dunst" "gtk-3.0" "kitty" "nwg-look" "qt5ct" "swaylock" "swww" "waybar" "wloggout" "xsettingsd" "dolphinrc" "kdeglobals")
rm -r Configs/.config
# copy all config files from list
for folders in "${files[@]}"
do
    cp -r ~/$file Configs/$file
done



