# I borrowed some of these tasks from: https://sparrow.dev/torchvision-transforms/

# documentation for the io library found here: https://docs.python.org/3/library/io.html#io.BytesIO 
import io
import requests
import torchvision.transforms as T

# PIL, or Python Image Library, is a library of functions for working with images
from PIL import Image

import matplotlib.pylab as plt
import numpy as np

# We request the image from a give link, I found a new sparrow photo after the one offered in the blog had expired
# Docuemntation for the Python Image Library: https://pillow.readthedocs.io/en/stable/reference/Image.html 
resp = requests.get('https://www.allaboutbirds.org/guide/assets/photo/305880301-1280px.jpg')
img = Image.open(io.BytesIO(resp.content))

# Here is some code to look at the pixel distribution:
plt.hist(np.array(img).ravel(), bins=50, density=True);
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels");

# to see the plot of the distribution run:
plt.show()
# This is the distribution of each "channel" in the image
# Remember that each channel is a represenation of rgb
# np.array(img) turns our image object in to a 3-d array with shape: (960, 1280, 3)
#np.array(img).ravel() turns this into a 1-D array, and strips away all of the association.
# For more resources on working with image channels: https://bioimagebook.github.io/chapters/1-concepts/4-colors/python.html 


# This is normalizing the image so that we can work on it with ML techniques
# Can read more here: https://sparrow.dev/pytorch-normalize/ 
preprocess = T.Compose([
   T.Resize(256),
   T.CenterCrop(224),
   T.ToTensor(),
   T.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
   )
])

x = preprocess(img)
x.shape
#torch.Size([3, 224, 224])
# We basically reduced the image to be 224 X 224