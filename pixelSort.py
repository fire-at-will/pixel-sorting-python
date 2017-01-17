from PIL import Image
from PIL import ImageFile
import sys
import os
import argparse
from argparse import RawTextHelpFormatter
from constants import *
import time
import random
import math
import imageio
import re
from subprocess import call
import shutil


# User defined constants
global SORT_BY_ROWS
SORT_BY_ROWS = True
SORT_BY_COLUMNS = False
SORT_BY_CIRCLE = False
global ALGORITHM
global iterations
global HEURISTIC
HEURISTIC = RGBSUM

global image
global imagePixels
global width, height
global imageOutput
global fileName

global rowCount, columnCount

global interval
global randomInterval

global pixelSnapshots
global currentRowOrCol
global snapshotsPerRowOrCol

global iterationCount
iterationCount = 0

################################################################################
#
#                                   Main()
#
################################################################################

def main():
    global KEEP_TEMP_FILES
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Get start time
    startTime = time.time()

    getSettings()

    if createGIF:
        createGIFTemporaryFiles()

    sortImage()

    if createGIF:
        print "Pixel sort complete. Creating GIF..."
        createImagesFromSnapshot()
        createGIFFromImages()
        if not KEEP_TEMP_FILES:
            removeGIFTemporaryFiles()


    print "Done! Pixel sort took %i seconds.               " % (time.time() - startTime)

    image.show()

    saveImage("_output", False)

################################################################################
#
#                                   I/O
#
################################################################################

def createGIFFromImages():
    # Images to mp4
    os.system("ffmpeg -r 60 -i temp/%8d.png -c:v libx264 out.mp4")
    # ffmpeg -r 60 -i temp/%3d.png -c:v libx264 out.mp4

    # Create a palatte
    # ffmpeg -y -i out.mp4 -vf fps=60,scale=320:-1:flags=lanczos,palettegen palette.png
    os.system("ffmpeg -y -i out.mp4 -vf fps=60,scale=600:-1:flags=lanczos,palettegen palette.png")

    # Create GIF
    # ffmpeg -i out.mp4 -filter_complex "fps=60,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse" output.gif
    os.system("ffmpeg -i out.mp4 -i palette.png -filter_complex \"fps=60,scale=600:-1:flags=lanczos[x];[x][1:v]paletteuse\" out.gif")

def createGIFTemporaryFiles():
    global height, width, pixelSnapshots, SORT_BY_ROWS, SORT_BY_COLUMNS
    print "Creating temporary data structures for GIF creation."

    if not os.path.exists(TEMP_FILE_LOCATION):
        os.makedirs(TEMP_FILE_LOCATION)

    if SORT_BY_COLUMNS:
        pixelSnapshots = [0] * height
        for ii in range(0, height):
            pixelSnapshots[ii] = []

    elif SORT_BY_ROWS:
        pixelSnapshots = [0] * width
        for ii in range(0, width):
            pixelSnapshots[ii] = []
    
def removeGIFTemporaryFiles():
    shutil.rmtree(TEMP_FILE_LOCATION)


def saveSnapshot(pixelLine):
    global pixelSnapshots#, SORT_BY_COLUMNS, SORT_BY_ROWS, height, width, imagePixels, currentRowOrCol

    pixelSnapshots[currentRowOrCol].append(pixelLine)

def createImagesFromSnapshot():
    global frameCount, width, height, pixelSnapshots

    # Determine largest number of frames
    frameCount = 0
    for i in range(0, len(pixelSnapshots)):
        if len(pixelSnapshots[i]) > frameCount:
            frameCount = len(pixelSnapshots)

    # For each frame...
    for frameNumber in range(0, len(pixelSnapshots[0])):
        # Create an image
        frame = Image.new("RGB", (width, height))
        framePixels = frame.load()

        # Color the image

        # For each row
        for row in range(0, height):
            # For each column
            try:
                rowOfPixels = pixelSnapshots[row][frameNumber]
            except:
                rowOfPixels = pixelSnapshots[row][len(pixelSnapshots[row]) - 1]
            for col in range(0, width):
                framePixels[col, row] = rowOfPixels[col]

        # Save image
        sys.stdout.write("Saving frame: %i/%i   \r" % (frameNumber, frameCount) )
        sys.stdout.flush()
        saveName = TEMP_FILE_LOCATION + "/%08i" % frameNumber + ".png"
        frame.save(saveName)

def saveImage(fileSuffixName, isTemp):
    global fileName, image
    baseName = os.path.splitext(fileName)[0]
    saveName = ""

    if isTemp:
        saveName = TEMP_FILE_LOCATION + fileSuffixName + ".png"
    else:
        saveName = baseName + fileSuffixName + ".png"

    try:
        image.save(saveName)
    except:
        print "Error: Cannot save as png. Attempting to save as jpg."
        if isTemp:
            saveName = TEMP_FILE_LOCATION + fileSuffixName + ".jpg"
        else:
            saveName = baseName + fileSuffixName + ".jpg"
        image.save(saveName)


################################################################################
#
#                             Sort Logic
#
################################################################################

def sortImage():
    global imagePixels, image, height, width, SORT_BY_ROWS, SORT_BY_COLUMNS, SORT_BY_CIRCLE, ALGORITHM, interval, randomInterval, currentRowOrCol

    if SORT_BY_ROWS:
        # Sort by rows
        print "Sorting by rows."
        for i in range(height):
            currentRowOrCol = i
            iterationCount = 0
            sys.stdout.write("Processing row: %i/%i   \r" % (i, height) )
            sys.stdout.flush()

            rowInterval = interval
            if randomInterval:
                # Create random interval for row
                rowInterval = random.randrange(2, interval)

            endOfLine = width

            # How many sections we have to sort in each row
            loops = int(width / rowInterval)

            for k in range(0, loops):
                array = []

                # Get the next pixels to sort in the interval
                for j in range(rowInterval):
                    array.append(imagePixels[k * rowInterval + j, i])

                # Sort the interval
                sort(array)

                # Replace image pixels with the sorted pixels
                for j in range(rowInterval):
                    imagePixels[(k * rowInterval + j), i] = array[j]

    elif SORT_BY_COLUMNS:
        # Sort by columns
        print "Sorting by columns."
        for i in range(width):
            currentRowOrCol = i
            iterationCount = 0
            sys.stdout.write("Processing column: %i/%i   \r" % (i, width) )
            sys.stdout.flush()

            columnInterval = interval
            if randomInterval:
                columnInterval = random.randrange(2, interval)

            endOfLine = height

            loops = int(height / columnInterval)

            for k in range(0, loops):
                array = []

                for j in range(columnInterval):
                    array.append(imagePixels[i, k * columnInterval + j])

                sort(array)

                for j in range(columnInterval):
                    imagePixels[i, (k * columnInterval + j)] = array[j]

    elif SORT_BY_CIRCLE:
        # Sort by circle
        print "Sorting in circle pattern."

        # Circle specs
        radius = 1
        centerX = round(width / 2)
        centerY = round(height / 2)

        while radius < centerX:
            # For each radius size, do:

            minX = int(round(centerX - radius))
            maxY = int(round(centerX + radius))

            array = []

            # Get the next pixels to sort

            # Top Half of circle
            try:
                for x in range(minX, maxY):
                    array.append(imagePixels[x, centerY + circleEquation("positive", x, centerX, radius)])

                # Bottom Half of circle
                for x in range(minX, maxY):
                    array.append(imagePixels[x, centerY + circleEquation("negative", x, centerX, radius)])

                # Sort the interval
                sort(array)

                # Replace image pixels with the sorted pixels

                arrayIndex = 0
                # Top Half of circle
                for x in range(minX, maxY):
                    imagePixels[x, centerY + circleEquation("positive", x, centerX, radius)] = array[arrayIndex]
                    arrayIndex = arrayIndex + 1

                # Bottom Half of circle
                for x in range(minX, maxY):
                    imagePixels[x, centerY + circleEquation("negative", x, centerX, radius)] = array[arrayIndex]
                    arrayIndex = arrayIndex + 1

                # Increase radius by 1 and repeat
                radius += 1
            except:
                return 

def sort(array):
    global ALGORITHM, width, height, iterations

    if ALGORITHM == QUICK:
        quickSort(array, width)
    elif ALGORITHM == BUBBLE:
        bubbleSort(array, iterations, width)
    elif ALGORITHM == COCKTAIL:
        cocktailSort(array, iterations)

def circleEquation(mode, x, h, r):

    y = math.sqrt( (r * r) - ( (x - h) * (x - h) ) )

    if mode == "positive":
        # Return positive version
        return abs(y)
    else:
        # Return negative version
        if y > 0:
            y = y - y - y

        return y


################################################################################
#
#                             Bubble Sort
#
#   Modified to only perform a certain number of iterations
#
################################################################################
def bubbleSort(pixelLine, iterations, maxIterations):
    global iterationCount, snapshotsPerRowOrCol
    iteration = 0

    while (iteration < iterations) and (iteration < len(pixelLine) - 1):
        # Do an iteration
        for index, pixel in enumerate(pixelLine):
            nextIndex = index + 1
            if nextIndex >= maxIterations:
                break

            if valueOfPixel(pixel) > valueOfPixel(pixelLine[nextIndex]):
                temp = pixelLine[index]
                pixelLine[index] = pixelLine[index + 1]
                pixelLine[index + 1] = temp
                if createGIF:
                    if iterationCount % snapshotsPerRowOrCol == 0:
                        saveSnapshot(list(pixelLine))
                    iterationCount = iterationCount + 1

        iteration = iteration + 1



################################################################################
#
#                             Cocktail Sort
#
#   Modified to only perform a certain number of iterations
#
################################################################################
def cocktailSort(pixelLine, numIterations):
    iteration = 0
    up = range(len(pixelLine)-1)
    while True:

        if iteration >= iterations:
            return

        for indices in (up, reversed(up)):
            swapped = False
            for i in indices:
                if valueOfPixel(pixelLine[i]) > valueOfPixel(pixelLine[i+1]):
                    pixelLine[i], pixelLine[i+1] =  pixelLine[i+1], pixelLine[i]
                    swapped = True
            if not swapped:
                return

################################################################################
#
#                             Quick Sort
#
################################################################################
def quickSort(pixelLine, maxIterations):
    quickSortAlgorithm(pixelLine, 0, len(pixelLine) - 1, 0, maxIterations)

def quickSortAlgorithm(pixelLine, low, high, iteration, maxIterations):
   
   if iteration >= maxIterations:
       return;

   iteration = iteration + 1

   if low < high:

       midpoint = partition(pixelLine, low, high)

       quickSortAlgorithm(pixelLine, low, midpoint - 1, iteration, maxIterations)
       quickSortAlgorithm(pixelLine, midpoint + 1, high, iteration, maxIterations)


def partition(pixelLine,low,high):
    global iterationCount, snapshotsPerRowOrCol, createGIF
    
    pivotvalue = valueOfPixel(pixelLine[low])

    leftmark = low + 1
    rightmark = high

    done = False
    while not done:

       while leftmark <= rightmark and valueOfPixel(pixelLine[leftmark]) <= pivotvalue:
           leftmark = leftmark + 1

       while valueOfPixel(pixelLine[rightmark]) >= pivotvalue and rightmark >= leftmark:
           rightmark = rightmark - 1

       if rightmark < leftmark:
           done = True
       else:
           temp = pixelLine[leftmark]
           pixelLine[leftmark] = pixelLine[rightmark]
           pixelLine[rightmark] = temp

    temp = pixelLine[low]
    pixelLine[low] = pixelLine[rightmark]
    pixelLine[rightmark] = temp

    if createGIF:
        if iterationCount % snapshotsPerRowOrCol == 0:
            saveSnapshot(list(pixelLine))
        iterationCount = iterationCount + 1

    return rightmark

################################################################################
#
#                             Value of Pixel
#
################################################################################

def sumRGB(pixel):
    return pixel[0] + pixel[1] + pixel[2]

def valueOfPixel(pixel):
    global HEURISTIC

    if HEURISTIC == RGBSUM:
        # Return sum of RGB
        return sumRGB(pixel)
    elif HEURISTIC == R:
        # Return red
        return pixel[0]
    elif HEURISTIC == G:
        # Return green
        return pixel[1]
    elif HEURISTIC == B:
        # Return blue
        return pixel[2]
    else:
        # Just in case return RGB sum
        return sumRGB(pixel)

################################################################################
#
#                             Settings functions
#
#   Gets user parameters from command line and
#
################################################################################

def getSettings():
    global image, imagePixels, height, width, fileName, SORT_BY_ROWS, SORT_BY_COLUMNS, SORT_BY_CIRCLE, iterations, ALGORITHM, HEURISTIC, interval, randomInterval, createGIF, frameCount, snapshotsPerRowOrCol, KEEP_TEMP_FILES

    # Configure parser
    parser = argparse.ArgumentParser(description = "Pixel sort an image!", formatter_class=RawTextHelpFormatter)
    parser.add_argument("image_name", help="Name of the image to sort.")
    parser.add_argument("-v", "--vertical", help="Sort image vertically. Default is to sort horizontally.", action="store_true")
    parser.add_argument("-c", "--circle", help="Sort image in a circle pattern, going out from the center of the image.", action="store_true")

    algorithmString = """Specifies which sort algorithm to use. Defaults to quick sort.
    * bubble - Bubble Sort. Slow, but produces good results. Use -i in the 200-400 range.
    * cocktail - Cocktail Sort. Similar to bubble, but faster.
    * quick - Quick Sort. Very fast with dramatic results. Use -i in the 1-10 range.
    """

    parser.add_argument("-a", "--algorithm", help=algorithmString)
    parser.add_argument("-i", "--iterations", help="Sets the number of iterations to run the sort. The higher this value, the more extreme the result.")

    parser.add_argument("-si", "--sortInterval", help="Sets the interval length to sort by. Use -r to make this random for each row/line.")
    parser.add_argument("-r", "--random", help="Sets the interval length for each row/column to be random integer between 2 and i.", action="store_true")

    parser.add_argument("-g", "--gif", help="Creates a GIF animation of the pixel sort. Slows down computation time considerably due to IO.", action="store_true")
    parser.add_argument("-f", "--frames", help="The number of frames that the created GIF will contain. Defaults to 50.")
    parser.add_argument("-k", "--keepTempFiles", help="Keeps the temporary files created by the GIF creation process. Default is false.", action="store_true")

    args = parser.parse_args()

    # Parse image
    try:
        fileName = args.image_name

        image = Image.open(fileName)
        imagePixels = image.load()

        width, height = image.size

    except :
        print "Error: %s is not present in the current working directory. Please enter a valid file name." % fileName
        sys.exit(0)

    # Are we sorting vertically or horizontally?
    if args.vertical:
        print "Vert"
        SORT_BY_ROWS    = False
        SORT_BY_COLUMNS = True

    if args.circle:
        SORT_BY_CIRCLE  = True
        SORT_BY_ROWS    = False

    if args.algorithm:
        algorithm = args.algorithm
        if algorithm == "bubble":
            print "Using bubble sort. This might take a while."
            ALGORITHM = BUBBLE
        elif algorithm == "cocktail":
            print "Using cocktail sort."
            ALGORITHM = COCKTAIL
        elif algorithm == "quick":
            print "Using quick sort."
            ALGORITHM = QUICK
    else:
        print "No sorting algorithm specified. Using quick sort."
        ALGORITHM = QUICK

    if args.iterations:
        iterations = int(args.iterations)
        print "Performing %i iterations." % iterations
    else:
        if SORT_BY_ROWS:
            iterations = width
        else:
            iterations = height

    if args.sortInterval:
        interval = int(args.sortInterval)
    else:
        if SORT_BY_ROWS:
            interval = width
        elif SORT_BY_COLUMNS:
            interval = height
    if args.random:
        randomInterval = True
    else:
        randomInterval = False

    if args.gif:

        if SORT_BY_CIRCLE:
            print "ERROR: Cannot create a GIF of a circle sort."
            exit()

        createGIF = True
        if args.frames:
            frameCount = int(args.frames)
        else:
            frameCount = 100

        if SORT_BY_ROWS:
            snapshotsPerRowOrCol = math.floor(width / frameCount)
            print "Num of iterations per row: %i" % snapshotsPerRowOrCol
        else:
            snapshotsPerRowOrCol = math.floor(height / frameCount)

        if args.keepTempFiles:
            KEEP_TEMP_FILES = args.keepTempFiles
        else:
            KEEP_TEMP_FILES = False
    else:
        createGIF = False


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

################################################################################
#
#                             Program Bootstrap
#
################################################################################

if __name__ == "__main__":
    main()
