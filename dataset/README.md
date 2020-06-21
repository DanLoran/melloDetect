## Data manipulation

We have several scripts that can help to manipulate image so that is more convenient
to run training and evaluation operations on the dataset.

### Shrink
The script `shrink.py` can be used to resize the images in the target directory
`Data/Images` to the size 256x256. The images to be resized can have extensions
`.jpeg`, `.jpg` or `.png`. The images can have any dimension. All the output Images
will be created in the same directory of the other images, and will not create a copy
if an image with the same name was found. The output image will have name
`<original_name>_small.jpeg`.

### Featurize
The script `featurize.py` can be used to produce feature vectors given the name
of a pretrained model and a set of images. The images can have the extension
accepted by the shrinking script. The vectors will be saved in the same directory
of the images and will have the the following name: `<original_name>_<pretrained_model>.pt`.
The format **.pt** can be read by pytorch. The vector size depends on the model
that has been selected.
