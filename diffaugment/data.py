"""Modified from github/data-efficient-gans"""

import glob
import os
import tensorflow as tf
import numpy as np
import PIL

def error(msg):
    print('Error: ' + msg)
    exit(1)


class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, resolution=None, print_progress=True,
                 progress_interval=10, channels_first=True):
        self.tfrecord_dir = tfrecord_dir
        self.tfr_prefix = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        if resolution:
            self.tfr_prefix += '-{}'.format(resolution)
        self.expected_images = expected_images
        self.cur_images = 0
        self.shape = None
        self.resolution_log2 = None
        self.tfr_writers = []
        self.print_progress = print_progress
        self.progress_interval = progress_interval
        self.channels_first = channels_first

        if channels_first:
            self.hax = 1
            self.wax = 2
            self.cax = 0
        else:
            self.hax = 0
            self.wax = 1
            self.cax = 2


        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self):  # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def set_shape(self, shape):
        self.shape = shape
        self.resolution_log2 = int(np.log2(self.shape[self.hax]))
        # assert self.shape[0] in [1, 3]
        assert self.shape[self.hax] == self.shape[self.wax]
        assert self.shape[self.hax] == 2**self.resolution_log2
        tfr_opt = tf.compat.v1.python_io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.NONE)
        tfr_file = self.tfr_prefix + '.tfrecords'
        self.tfr_writers.append(tf.compat.v1.python_io.TFRecordWriter(tfr_file, tfr_opt))

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.set_shape(img.shape)
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                if self.channels_first:
                    img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
                else:
                    img = (img[0::2, 0::2, :] + img[0::2, 1::2, :] + img[1::2, 0::2, :] + img[1::2, 1::2, :]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.int32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_from_images(data_dir, resolution=None, tfrecord_dir=None, shuffle=True, channels_first=True):
    if tfrecord_dir is None:
        tfrecord_dir = data_dir
    print('Loading images from "%s"' % data_dir)
    image_filenames = sorted(glob.glob(os.path.join(data_dir, '*')))
    image_filenames = [fname for fname in image_filenames if fname.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'bmp']]
    if len(image_filenames) == 0:
        error('No input images found')

    img = np.asarray(PIL.Image.open(image_filenames[0]))
    if resolution is None:
        resolution = img.shape[0]
        if img.shape[1] != resolution:
            error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    channels = img.shape[2] if img.ndim == 3 else 1
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir, len(image_filenames),
                          resolution, channels_first=channels_first) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = PIL.Image.open(image_filenames[order[idx]])
            if resolution is not None:
                img = img.resize((resolution, resolution), PIL.Image.LANCZOS)
            img = np.asarray(img)
            if channels == 1 or len(img.shape) == 2:
                ax = 0 if channels_first else -1
                img = np.stack([img] * channels, axis=ax)  # HW => CHW || HWC
            else:
                if channels_first:
                    img = img.transpose([2, 0, 1])  # HWC => CHW

            tfr.add_image(img)
    return tfrecord_dir
