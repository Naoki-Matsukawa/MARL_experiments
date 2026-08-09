#!/usr/bin/env bash
# check_packages_user.sh  (root 不要)

PACKAGES=(
  git cmake build-essential libgl1-mesa-dev libsdl2-dev
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip
)

printf "\n%-25s | %-20s\n" "PACKAGE" "INSTALLED VERSION"
printf "%0.s-" {1..50}; echo

for pkg in "${PACKAGES[@]}"; do
  ver=$(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null || true)
  [[ -z $ver ]] && ver="-"
  printf "%-25s | %-20s\n" "$pkg" "$ver"
done
