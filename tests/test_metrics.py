import unittest

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import learn2seg.metrics as metrics


class MetricTest(unittest.TestCase):

    def test_iou(self):
        # Setup TF configs
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        set_session(sess)

        # test the iou output
        true = [[0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 0]
                ]

        true = np.asarray(true, dtype=np.float32)
        # change to [batch, im_height, im_width, channel]
        true = np.expand_dims(true, axis=0)
        true = np.expand_dims(true, axis=3)

        pred = [[0, 0, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 1, 0],
                [0, 0, 0, 0]
                ]

        pred = np.asarray(pred, dtype=np.float32)
        pred = np.expand_dims(pred, axis=0)
        pred = np.expand_dims(pred, axis=3)

        pred = tf.convert_to_tensor(pred)
        true = tf.convert_to_tensor(true)

        out_iou = sess.run(metrics.bin_iou(pred, true))

        expected_iou = 0.625
        self.assertEqual(expected_iou, out_iou)

    def test_np_iou(self):

        # test the iou output
        true = [[0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 0]
                ]

        true = np.asarray(true, dtype=np.float32)

        pred = [[0, 0, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 1, 0],
                [0, 0, 0, 0]
                ]

        pred = np.asarray(pred, dtype=np.float32)

        out_iou = metrics.np_bin_iou(pred, true)

        expected_iou = 0.625
        self.assertEqual(expected_iou, out_iou)


if __name__ == "main":
    unittest.main()
