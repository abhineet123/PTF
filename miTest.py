import distanceUtils
import numpy as np

if __name__ == '__main__':
    img1 = np.array([1, 1, 4, 4, 7, 7, 6, 4, 5, 4, 3, 2, 3, 1])
    img2 = np.array([1, 0, 2, 3, 4, 7, 6, 7, 5, 4, 3, 2, 3, 1])

    print 'img1:\n', img1
    print 'img2:\n', img2

    no_of_pixels = img1.shape[0]
    no_of_bins = 8

    hist12, hist1, hist2 = distanceUtils.initMI(no_of_pixels, no_of_bins)

    print 'hist12:\n', hist12
    print 'hist1:\n', hist1
    print 'hist2:\n', hist2

    hist12, hist1, hist2 = distanceUtils.getHistograms(img1, img2)

    print 'hist12:\n', hist12
    print 'hist1:\n', hist1
    print 'hist2:\n', hist2

    mi=distanceUtils.getMIPoints(img1, img2)

    print 'mi: ', mi
