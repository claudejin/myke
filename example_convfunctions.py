import myke.datasets
from PIL import Image

url = 'https://github.com/WegraLee/deep-learning-from-scratch-3/' \
      'raw/images/zebra.jpg'
img_path = myke.utils.get_file(url)
img = Image.open(img_path)
img.show()

import numpy as np
from myke.models import VGG16

x = VGG16.preprocess(img)
x = x[np.newaxis]
print(type(x), x.shape)

model = VGG16(pretrained=True)
with myke.test_mode():
    y = model(x)
predicted_id = np.argmax(y.data)

model.plot(x, to_file='vgg.pdf')
labels = myke.datasets.ImageNet.labels()
print(labels[predicted_id])
