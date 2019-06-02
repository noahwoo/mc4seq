#!/bin/sh
../../crf_learn -c 4.0 -t template -a svmmc train.data model
#../../.libs/crf_learn -c 4.0 -t template -a svmmc train.data model
# ../../crf_test -m model test.data

# ../../crf_learn -a MIRA template train.data model
# ../../crf_test -m model test.data

#../../crf_learn -a CRF-L1 template train.data model
#../../crf_test -m model test.data

# rm -f model
