#pokemon-colorscripts --no-title -r 1,2,3,4
pokemon-colorscripts -r 1,2,3,4
# load oh-my-zsh
ZSH=/usr/share/oh-my-zsh/

# load plugins
plugins=(git)
source /usr/share/zsh/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
source /usr/share/zsh/plugins/zsh-autosuggestions/zsh-autosuggestions.plugin.zsh
source $ZSH/oh-my-zsh.sh

# set color for autosuggest
ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE='fg=5'


# Enable Powerlevel10k instant prompt. Should stay close to the top of ~/.zshrc.
# Initialization code that may require console input (password prompts, [y/n]
# confirmations, etc.) must go above this block; everything else may go below.

# auto complete
#autoload -Uz compinit; compinit; _comp_options+=(globdots);

if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# Use powerline
USE_POWERLINE="true"
# Source manjaro-zsh-configuration
if [[ -e /usr/share/zsh/manjaro-zsh-config ]]; then
  source /usr/share/zsh/manjaro-zsh-config
fi
# Use manjaro zsh prompt
if [[ -e /usr/share/zsh/manjaro-zsh-prompt ]]; then
  source /usr/share/zsh/manjaro-zsh-prompt
fi

source ~/powerlevel10k/powerlevel10k.zsh-theme

# To customize prompt, run `p10k configure` or edit /usr/share/zsh/p10k.zsh.
[[ ! -f /usr/share/zsh/p10k.zsh ]] || source /usr/share/zsh/p10k.zsh

# react native
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/emulator
export PATH=$PATH:$ANDROID_HOME/tools
export PATH=$PATH:$ANDROID_HOME/tools/bin
export PATH=$PATH:$ANDROID_HOME/platform-tools
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin
export PATH=$PATH:/var/lib/snapd/snap/bin/android-studio
export PATH=$PATH:/home/alex/Android/jdk-14.0.2/bin
alias picon="ssh pi@192.168.178.101"
alias blog="cd ~/Nextcloud/Dokumente/blog"
alias root="ssh root@a-wels.de"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/alex/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/alex/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/alex/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/alex/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
alias Scripts="cd ~/Scripts"
alias Webcam="cd ~/Scripts && sh webcam.sh 4"
alias ussh="ssh u103774064@access860583244.webspace-data.io"
alias rs="ssh alex@202.61.245.114"

# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh.
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh
export XDG_RUNTIME_DIR=/run/user/$(id -u)
HISTFILE=~/.zsh_history
HISTSIZE=10000
SAVEHIST=10000
setopt appendhistory
