call plug#begin('~/.local/share/nvim/plugged')
if has('nvim')
  Plug 'Shougo/deoplete.nvim', { 'do': ':UpdateRemotePlugins' }
else
  Plug 'Shougo/deoplete.nvim'
  Plug 'roxma/nvim-yarp'
  Plug 'roxma/vim-hug-neovim-rpc'
endif
Plug 'neomake/neomake'
Plug 'sirVer/ultisnips'
Plug 'honza/vim-snippets'
"VimTex
Plug 'lervag/vimtex'

" colorschemes"
Plug 'altercation/vim-colors-solarized'
Plug 'rafi/awesome-vim-colorschemes'
"this has to be under"
Plug 'dracula/vim'

"air and lightline"
Plug 'itchyny/lightline.vim'
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
Plug 'tpope/vim-fugitive'

"Auto Pairs
Plug 'jiangmiao/auto-pairs'
Plug 'machakann/vim-sandwich'

" A Vim Plugin for Lively Previewing LaTeX PDF Output
Plug 'xuhdev/vim-latex-live-preview', { 'for': 'tex' }

call plug#end()

set tabstop=4
set shiftwidth=4

"Spellchecking
set spell spelllang=en_us,de
" Show nine spell checking candidates at most
set spellsuggest=best,9

syntax enable
set t_Co=256
set background=dark
colorscheme onedark

" Activate mouse
" set mouse=a
"More undos"
set undofile
set undodir=$HOME/.vim/undo
set undolevels=1000
set undoreload=10000

"Spell check error color"
hi clear SpellBad
hi SpellBad cterm=underline
	:highlight clear SpellBad
	:highlight SpellBad  ctermfg=196 cterm=underline
hi SpellBad gui=undercurl

hi clear SpellCap
hi SpellCap cterm=underline
	:highlight clear SpellCap
	:highlight SpellCap  ctermfg=196 cterm=underline
hi SpellCap gui=undercurl

hi clear SpellRare



"Set line numbers
set number
let g:UltiSnipsExpandTrigger="<tab>"
let g:UltiSnipsJumpForwardTrigger="<tab>"
let g:UltiSnipsJumpBackwardTrigger="<s-tab>"
let g:UltiSnipsListSnippets="<c-tab>"
let g:deoplete#enable_at_startup = 1
call deoplete#custom#var('omni', 'input_patterns', {
        \ 'tex': g:vimtex#re#deoplete
        \})

"air-line"
"shows the current branch"
let g:lightline = { 'colorscheme': 'icebergDark' }
let g:airline#extensions#branch#enabled = 1
let g:airline#extensions#branch#empty_message = ''
let g:airline#extensions#ale#enabled = 1
"Font for symbols"
let g:airline_powerline_fonts = 1
if !exists('g:airline_symbols')
  let g:airline_symbols = {}
endif

let g:airline_symbols.branch = ''
let g:airline_symbols.notexists = 'Ɇ'
let g:airline_symbols.linenr = '☰'
let g:airline_left_sep = '»'
let g:airline_symbols.dirty='!'
let g:airline_symbols.maxlinenr = ' ㏑'
let g:airline_symbols.paste = 'ρ'
"lightline colorscheme"
let g:lightline = {
      \ 'colorscheme': 'solarized',
      \ }
