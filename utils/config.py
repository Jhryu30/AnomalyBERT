import os

PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATASET_DIR = '/mnt/d/ubuntu/datasets/anomaly/processed/'
LOG_DIR = 'logs/'
DATA_PROPERTY_DIR = 'data/'


TRAIN_DATASET = {'SMAP' : DATASET_DIR+'SMAP_train.npy',
                 'MSL' : DATASET_DIR+'MSL_train.npy',
                 'SMD' : DATASET_DIR+'SMD_train.npy',
                 'SWaT' : DATASET_DIR+'SWaT_train.npy'
                }

TEST_DATASET = {'SMAP' : DATASET_DIR+'SMAP_test.npy',
                'MSL' : DATASET_DIR+'MSL_test.npy',
                'SMD' : DATASET_DIR+'SMD_test.npy',
                'SWaT' : DATASET_DIR+'SWaT_test.npy'
               }

TEST_LABEL = {'SMAP' : DATASET_DIR+'SMAP_test_label.npy',
              'MSL' : DATASET_DIR+'MSL_test_label.npy',
              'SMD' : DATASET_DIR+'SMD_test_label.npy',
              'SWaT' : DATASET_DIR+'SWaT_test_label.npy'
             }

DATA_DIVISION = {'SMAP' : {'channel' : DATA_PROPERTY_DIR+'SMAP_test_channel.json',
                           'class' : DATA_PROPERTY_DIR+'SMAP_test_class.json'},
                 'MSL' : {'channel' : DATA_PROPERTY_DIR+'MSL_test_channel.json',
                          'class' : DATA_PROPERTY_DIR+'MSL_test_class.json'},
                 'SMD' : {'channel' : DATA_PROPERTY_DIR+'SMD_test_channel.json'}
                }


NUMERICAL_COLUMNS = {'SMAP' : (0,),
                     'MSL' : (0,),
                     'SMD' : tuple(list(range(7)) + list(range(8, 38)))
                    }

CATEGORICAL_COLUMNS = {'SMAP' : range(1, 25),
                       'MSL' : range(1, 55),
                       'SMD' : (7,),
                       'SWaT' : tuple([2,3,4] + list(range(9, 16)) + list(range(19, 25))\
                                      + list(range(29, 34)) + [42,43,48,49,50])
                      }


# SMD series
train_smd = {'SMD{}'.format(i) : DATASET_DIR+'SMD{}_train.npy'.format(i) for i in range(28)}
test_smd = {'SMD{}'.format(i) : DATASET_DIR+'SMD{}_test.npy'.format(i) for i in range(28)}
label_smd = {'SMD{}'.format(i) : DATASET_DIR+'SMD{}_test_label.npy'.format(i) for i in range(28)}
numerical_smd = {'SMD{}'.format(i) : NUMERICAL_COLUMNS['SMD'] for i in range(28)}
categorical_smd = {'SMD{}'.format(i) : (7,) for i in range(28)}

TRAIN_DATASET.update(train_smd)
TEST_DATASET.update(test_smd)
TEST_LABEL.update(label_smd)
NUMERICAL_COLUMNS.update(numerical_smd)
CATEGORICAL_COLUMNS.update(categorical_smd)

# SWaT update
NUMERICAL_COLUMNS['SWaT'] = tuple([i for i in range(0, 51) if i not in CATEGORICAL_COLUMNS['SWaT']])