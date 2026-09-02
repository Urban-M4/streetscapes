# Import local images into streetscapes

If you have images you collected yourself, or have images from a different online
source that is not (yet) supported by this package, you can still register them in
the database and run segmentation models on the images.

This short tutorial will guide you through this process. 
Let's start by creating a new project:

```sh
streetscapes config set active_project local-test
```

Next, download two building images from Wikimedia as a test:

```sh
mkdir test-images
wget -O test-images/img1.jpg -q "https://upload.wikimedia.org/wikipedia/commons/8/8f/Arnstorf_-_Schloss_Mariakirchen_westlich,_Aussenansicht.JPG?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original"
wget -O test-images/img2.jpg -q "https://upload.wikimedia.org/wikipedia/commons/2/20/Bad_S%C3%A4ckingen_%E2%80%94_Rathausplatz_3_und_M%C3%BCnsterplatz_34.JPG?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original"
```

These images can be added to the database with:

```sh
streetscapes images add test-images/
```

Note that all images in the working directory will be copied (to the image cache dir)
and registered in the project database.

To segment the images do (for example with Maskformer):

```sh
streetscapes segment-images maskformer
```

Now the images can be viewed in the explorer with

```sh
streetscapes-explorer
```
