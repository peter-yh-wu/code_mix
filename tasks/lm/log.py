#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

import configs
import logging
import os

_log_file = '{}/{}.{}.log'.format(configs.log_dir, configs.args.mode, configs.timestamp)


def init_logger():
    if not os.path.exists(configs.log_dir):
        try:
            os.mkdir(configs.log_dir)
        except IOError:
            print("Can not create log file directory.")
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(_log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

