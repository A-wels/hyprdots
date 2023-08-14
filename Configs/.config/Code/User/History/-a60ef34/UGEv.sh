packagesNeeded='neovim zsh helix lolcat  python3-dev python3-pip python3-setuptools firefox git'
packagesManjaro='git base-devel yay neovim zsh cowsay helix lolcat firefox python-pynvim flatpak gtk3 pango gdk-pixbuf2 cairo glib2 gcc-libs glibc polybar nextcloud-client thunderbird texstudio texlive syncthing jdk11-openjdk fuse2'
packagesYay='cowsay visual-studio-code-bin zoom zsh-syntax-highlighting-git oh-my-zsh-git zsh-autosuggestions-git'
packagesFlatpak='md.obsidian.Obsidian'

if [ -x "$(command -v apk)" ];       then sudo apk add --no-cache $packagesNeeded
fi
if [ -x "$(command -v apt-get)" ]; then
 sudo add-apt-repository ppa:maveonair/helix-editor
 sudo apt update
 sudo apt-get install $packagesNeeded

fi
if [ -x "$(command -v dnf)" ];     then sudo dnf install $packagesNeeded
fi
if [ -x "$(command -v zypper)" ];  then sudo zypper install $packagesNeeded
fi
if [ -x "$(command  -v pacman)" ];  then
echo "Running pacman -Syy"
sudo pacman -Syy
for package in $packagesManjaro; do
    sudo pacman -S $package --noconfirm;
done
# install yay
cd ~ &&  git clone https://aur.archlinux.org/yay.git &&  cd yay &&  makepkg -si --noconfirm && rm -rf ~/yay

for package in $packagesYay; do
    yay -S $package --answerdiff=None;
done
fi

# install flatpak packages
for package in $packagesFlatpak; do
    flatpak install $package -y;
done

# move files
#
# zsh
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ~/powerlevel10k
echo 'source ~/powerlevel10k/powerlevel10k.zsh-theme' >>~/.zshrc

cp .zshrc ~/.zshrc

#nvim
mkdir ~/.config
mkdir ~/.config/nvim
cp init.vim ~/.config/nvim/init.vim
cp -r spell ~/.config/nvim

#bash
#cp .bashrc ~/bashrc

# scripts folder
mkdir ~/Scripts
cp -r Scripts ~/Scripts

# gitconfig
# cp .gitconfig ~/.gitconfig
# run installs
#
# nvim
pip install pynvim
curl -fLo ~/.local/share/nvim/site/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
nvim --headless +PlugInstall +qall
nvim --headless +UpdateRemotePlugins +qall


echo "Installing rust"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

chsh -s $(which zsh)
sudo chsh -s $(which zsh)


# setup syncthing
cp files/syncthing.service ~/.config/systemd/user/syncthing.service
systemctl --user enable syncthing.service
systemctl --user start syncthing.service