import os

PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATASET_DIR = '/mnt/d/ubuntu/datasets/anomaly/processed/'
LOG_DIR = 'logs/'
DATA_PROPERTY_DIR = 'data/'


TRAIN_DATASET = {'SMAP' : DATASET_DIR+'SMAP_train.npy',
                 'MSL' : DATASET_DIR+'MSL_train.npy',
                 'SMD' : DATASET_DIR+'SMD_train.npy'
                }

TEST_DATASET = {'SMAP' : DATASET_DIR+'SMAP_test.npy',
                'MSL' : DATASET_DIR+'MSL_test.npy',
                'SMD' : DATASET_DIR+'SMD_test.npy'
               }

TEST_LABEL = {'SMAP' : DATASET_DIR+'SMAP_test_label.npy',
              'MSL' : DATASET_DIR+'MSL_test_label.npy',
              'SMD' : DATASET_DIR+'SMD_test_label.npy'
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
                       'SMD' : (7,)
                      }