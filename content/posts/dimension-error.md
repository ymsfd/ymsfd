+++ 
draft = false
date = 2021-01-31T17:34:28+01:00
title = "Dimension varying when loading pretrained model"
description = ""
slug = ""
authors = []
tags = []
categories = []
externalLink = ""
series = []
+++

* seq2seq model, load data with torchtext Field, TabularData.. when loading the saved model got the error of varying dimension inside of layer. The total vocabulary size is changing
everytime by loading. 
* stuck reason: thought it is the model problem or data been randomly shufftled when loading
* real problem: data is saved in .csv file, and inside of data there are also commas, therefore, the data is incorrectly read.


