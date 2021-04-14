# PyTorch Image Manipulation Manual

Here we outline a few simple packages and their capabilities to work with images.

## Using `imageio`

We import the `imageio` package and use the `imageio.imread` function to read an image.

This imports an image as a numpy array in the $H\times W\times C$ arrangement.

Here $H$ is height $W$ is width and $C$ is channels.

```python
import imageio

img_arr = imageio.imread('https://i.redd.it/1wel0f8sbdl41.jpg')
img_arr.shape
## Out:
# (<H>, <W>, <C>)
```

With PyTorch we're interested in converting this to the $C\times H \times W$ format since convolutions in PyTorch expect this arrangement.

We can do so using the `.permute()` attribute after converting the numpy array into a tensor.

We can do the following:

```python
import torch

img_t = torch.from_numpy(img_arr)
out = img_t.permute(2, 0, 1) # C x H x W
```

## Creating Batches of Images

In PyTorch it is more efficient to pre-allocate memory where possible. To do so we shall first create a tensor of zeros of the appropriate shape and size as demonstrated.

We will simply assume that a list of image file names is provided to us and we are to randomly select $n$ images to create a batch.

We will also assume that these images are of the same resolution and have the same number of channels for simplicity.

Prior to using this for RGB images it is a good idea to think about whether or not these images have an alpha channel that needs to be ignored.

```python
import random
import imageio

def create_batch(image_filenames, n):
  sample = random.sample(image_filenames, n)
  imgs = [imageio.imread(fname) for fname in sample]
  outs = [torch.from_numpy(img).permute(2, 0 , 1) for img in imgs]
  C, H, W = outs[0].shape
  batch = torch.zeroes(n, C, H, W)
  for i in range(n):
    batch[i] = outs[i]
```

> A more "PyTorch"-y way of doing this is described in the [](pytorch.dataloader-and-dataset-manual.md).

## Using OpenSlide to load whole slide images

References: <https://openslide.org/api/python/>
and <https://openslide.org/>
and <https://openslide.org/docs/properties/>

## Whole-Slide Images (WSI)

Whole-slide images (WSI) are high resolution images used in digital histopathology.

> **Histopathology** (compound of three [Greek](https://en.wikipedia.org/wiki/Greek_language) words: ἱστός **histos** "tissue", πάθος **pathos** "suffering", and -λογία **[-logia](https://en.wikipedia.org/wiki/-logy)** "study of") refers to the [microscopic](https://en.wikipedia.org/wiki/Light_microscope) examination of [tissue](<https://en.wikipedia.org/wiki/Tissue_(biology)>) in order to study the manifestations of [disease](https://en.wikipedia.org/wiki/Disease). Specifically, in clinical medicine, histopathology refers to the examination of a [biopsy](https://en.wikipedia.org/wiki/Biopsy) or surgical [specimen](https://en.wikipedia.org/wiki/Laboratory_specimen) by a [pathologist](https://en.wikipedia.org/wiki/Pathology), after the specimen has been processed and histological sections have been placed onto glass slides

In terms of relative scale, looking for important artifacts within a WSI is like looking for something the size of a football in a football field. Since these images are such high resolution they also pose interesting computational challenges. The images uncompressed often exceed the amount of RAM available in commercial computers. This makes processing the image challenging since traditional image processing tools do not cater to such large images with immense detail.
Whole-slide images occur in various formats meant to store large images that are retrieved from different microscope and imaging software vendors. Common formats are listed below:

- Aperio SVS
- Hamamatsu VMS
- Hamamatsu VMU
- Leica SCN
- 3DHISTECH MRXS ("MIRAX")
- Trestle TIFF
- Generic tiled TIFF.

## Utility of OpenSlide

OpenSlide is a C program that efficiently manages the reading and utilization of WSI's for a variety of formats. Since there is no agreed upon standard for WSI's each vendor implements its own libraries and viewers. Furthermore, to protect intellectual property, these vendors obscure important details from these implementations. Thus the need for libraries like OpenSlide which bridges the divide by providing a free and Open Source solution to managing and reading WSI's.

## Basic usage of OpenSlide in Python

Imaging instrument manufacturers store WSI's as multi-resolution images. These images are at fixed zoom levels. Open slide is used to read a small amount of a huge image available at a desired resolution that is closest to an available zoom level.

### Reading the Whole-Slide Image as a python Object

WSI files are usually files that range from a few 100MB to a few GB's and have file extensions as mentioned above. OpenSlide uses a filename as a constructor and determines the vendor of the format using a combination of the filename and metadata present in the WSI file. To use OpenSlide we construct an OpenSlide object as demonstrated below.

```python
import openslide

wsi = openslide.OpenSlide('<filename>')
wsi.dimensions
## Out
# (width, height) of the entire slide.
```

These properties greatly aid processing the WSI and we will focus on them a little in the next section.

### Down-sampling Factors

We generally wish to see the resolutions of the down-sampling factors that are present in the WSI. This is done using the `level_dimensions` and `level_downsamples` attribute.

```python
import openslide

wsi.level_dimensions
##Out
# ((53130, 153470),
# (13283, 38368),
# (3321, 9592),
# (831, 2400),
# (209, 601))
slide.level_downsamples
##Out
# (1.0,
# 3.999898652415998,
# 15.998992404088622,
# 63.94042569193742,
# 254.7841317103074)
```

As we can see from this example, the dimensions and down-sampling factors are present in the order of highest resolution to lowest (with 1x being the highest and approx. 255x being the lowest).

> These magnification factors are also referred to as levels. For a slide with 4 magnification levels we say that there are 0-3 levels with each level increasing in down-sampling factor. In this example level 0 corresponds to a 1x down-sampling, level 2 corresponds to ~4x downsampling and so on.

### Other properties attribute

To know more about the image that encapsulated by this file, a `properties` attribute is provided. We can access it using the following attribute but instead of directly using this attribute, this snippet will pretty print the Map object returned.

```python
print('\n'.join(["{}:{}".format(k, wsi.properties[k])
                 for k in wsi.properties]))
```

Which outputs something like:

```text
leica.aperture:0.4
leica.creation-date:2018-11-03T10:29:21.633Z
leica.device-model:Leica SCN400;Leica SCN
leica.device-version:1.5.1.10804 2012/05/10 13:29:07;1.5.1.10864
leica.illumination-source:brightfield
leica.objective:20
openslide.bounds-height:44032
openslide.bounds-width:40608
openslide.bounds-x:4338
openslide.bounds-y:54127
openslide.level-count:5
openslide.level[0].downsample:1
openslide.level[0].height:153470
openslide.level[0].width:53130
openslide.level[1].downsample:3.9998986524159981
openslide.level[1].height:38368
openslide.level[1].width:13283
openslide.level[2].downsample:15.998992404088622
openslide.level[2].height:9592
openslide.level[2].width:3321
openslide.level[3].downsample:63.940425691937421
openslide.level[3].height:2400
openslide.level[3].width:831
openslide.level[4].downsample:254.78413171030741
openslide.level[4].height:601
openslide.level[4].width:209
openslide.mpp-x:0.5
openslide.mpp-y:0.5
openslide.objective-power:20
openslide.quickhash-1:1318cf83f75c09612990b092b90a423ae8cf9785fa6c936abdf14cc3cda436ce
openslide.region[0].height:44032
openslide.region[0].width:40608
openslide.region[0].x:4338
openslide.region[0].y:54127
openslide.vendor:leica
tiff.ResolutionUnit:centimeter
tiff.XResolution:20000
tiff.YResolution:20000
```

### Useful `.properties` of the WSI

There are three kinds of properties seen above.

- Manufacturer / Vendor specific properties starts with `leica.`
- Format based properties starts with `tiff.`
- Properties generated by OpenSlide. `openslide.`

We will focus on properties generated by OpenSlide, for more read the documentation [here](https://openslide.org/docs/properties/).

- `openslide.bounds-width` and `openslide.bounds.height`: The width and height of the rectangle that bounds the non-empty region of the slide.
- `openslide.bounds-x` and `openslide.bounds-y`: The X and Y coordinate of the top left corner of the non-empty region of the slide.
- `openslide.mpp-x` and `openslide.mpp-y`: Microns per-pixel in the X and Y dimension of the slide at level 0.

> **These properties are not always present** but when they are, they are highly useful.

## Reading a region from the slide

To read a region from the slide we use the `.read_region` method on the WSI object created above.

> This function **lazily** reads a region of the WSI at the specified down-sampling level. The fact that this function does not load the entire WSI into RAM is critical to OpenSlide usability. It is important to note that it is easy to misuse this function since no memory checks are made to validate the arguments.
> Read region takes three arguments as described below.

```python
import matplotlib.pyplot as plt

location = (0, 0) # the top left pixel
      # in the level 0 reference frame

level = 3         # downsample factor chosen as a
      # zero-based index into the
      # wsi.level_dimensions attribute.
size = (256, 256) # Total size of the region to be read in
      # at the downsampling level
        # specified above.
img = wsi.read_region(location, level, size)
# returns a PIL Image
```

> Note that we have specified the size of the image to be $256\times256$ to be read in at a down-sampling level of $3$. This is equivalent to reading a $256\times64, 256\times64$ of the original WSI starting at the top right corner. This is because level $3$ corresponds to $~64$ in the `.level_downsamples` attribute as observed in the Down-sampling Factors section.

## Reading the non-empty region of the slide

We must deal with the fact that the non-empty region is a rectangular region within the WSI surrounded by a black border that does not contain any information. Since we know how to read a non empty region and understand some of the attributes of we will proceed to design a function that reads a usable region of the slide. For this function we will need access to the following attributes of the slide, furnished by `openslide`.

> **Thus function will work for the simple case in which OpenSlide is able to determine the following attributes** > `openslide.bounds-height`, `openslide.bounds-width` > `openslide.bounds-x` and `openslide.bounds-y`.

We will use the `OpenSlide` object constructed above as a parameter to this function.

```python
def read_ne_region(wsi, top_left_x, top_left_y,
                        width, height, level,verbose=False):

    ne_start_x, ne_start_y = wsi.properties['openslide.bounds-x'], wsi.properties['openslide.bounds-y']
    max_width, max_height = wsi.properties['openslide.bounds-height'], wsi.properties['openslide.bounds-width']

    if level > len(wsi.level_downsamples) - 1:
      raise ValueError("Level should be in the range {} to {}".format(0, len(wsi.level_downsamples) - 1))
    downsampling_factor = wsi.level_downsamples[level]

    if verbose == True:
      print("Downsampling factor chosen {}".format(downsampling_factor))

    if width > (max_width // downsampling_factor):
      raise ValueError("Width should be less that {}".format(max_width // downsampling_factor))
    if height > (max_height // downsampling_factor):
      raise ValueError("Height should be less that {}".format(max_height // downsampling_factor))

    ne_img = wsi.read_region((top_left_x, top_left_y),
                              level, (width, height))
    return ne_img
```

This function ensures that we read a non-empty region of the slide or raise errors where inputs either request for regions outside the non-empty regions or outside the slide.
