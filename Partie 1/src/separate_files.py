#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from shutil import copyfile

DB_src = '../database/src/'
DB_train = '../database/train/'
DB_test = '../database/test/'

if not os.path.exists(DB_train):
  os.mkdir(DB_train)
if not os.path.exists(DB_test):
  os.mkdir(DB_test)

for root, _, files in os.walk(DB_src, topdown=False):
  cls = root.split('/')[-1]
  dir_train = os.path.join(DB_train, cls)
  if not os.path.exists(dir_train):
      os.mkdir(dir_train)

for root, _, files in os.walk(DB_src, topdown=True):
  cls = root.split('/')[-1]
  i = 0
  for name in files:
    imgSrc = os.path.join(root, name)
    if not name.endswith('.jpg'):
      continue
    imgName = name.split('.')[-2]
    if i % 2 == 0 or i % 5 == 0:
      imgDst = os.path.join(DB_train, cls, name)
    else:
      imgDst = os.path.join(DB_test, name)
    i = i + 1
    if not os.path.exists(imgDst):
      copyfile(imgSrc, imgDst)



