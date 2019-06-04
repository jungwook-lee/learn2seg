from learn2seg.data.TeethBoxes import TeethBoxesDataset


def main():
    # Calculate the statistics of the following data
    teeth_data = TeethBoxesDataset()

    # Check consistency of image sizes
    im = teeth_data.get_image(0)
    im_size = im.shape
    print("Image size is :", im.shape)
    out_of_size = 0
    for i in range(teeth_data.size):
        im = teeth_data.get_image(i)
        # assert (im_size == im.shape)
        if im_size != im.shape:
            print(teeth_data.img_files[i])
            print('Image : ' + str(i))
            print(im.shape)
            out_of_size += 1
    print('Images out of size are: ' + str(out_of_size))

    # First return the number of images
    print("Available images: " + str(teeth_data.size))

    # Calculate the number of bounding boxes
    bb_total = 0
    for n in range(teeth_data.size):
        boxes = teeth_data.get_boxes(n)
        bb_total += len(boxes)
    print("Total BB: " + str(bb_total))

    bb_per_im = bb_total / teeth_data.size
    print("Avg BB per image: " + str(bb_per_im))

    return


if __name__ == '__main__':
    main()
