#!/bin/bash
set -x
# git add mat/scripts/ 
git add mat/algorithms/mat/
git add mat/scripts/train/
git add mat/scripts/*.sh

git commit -m "fix"

git push
