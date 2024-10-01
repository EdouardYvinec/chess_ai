# How to use

You can use several players:
```bash
play --random
```
or Stockfish with a specific elo (e.g. 1150)
```bash
play --stockfish 1150
```
or use a supervised model (trained from stockfish)
```bash
play --supervised
```

## todo
- [ ] debug training protocol
- [ ] save and load game

## on mac
for mac installs, you may need to run the following
```bash
brew install stockfish
brew install cairo
```
add below line to bash or zsh
```bash
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export DYLD_LIBRARY_PATH="/usr/local/lib:/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
```
restart terminal

## on windows
```bash
pip install pipwin

pipwin install cairocffi
```