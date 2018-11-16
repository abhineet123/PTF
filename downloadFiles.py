import urllib2
import os
'''
- Select the setting before you run the script.

- To download the full dataset, set everything "True"


'''

# Select one or more of the video categories
ORIENTED_TASKS = False
COMPOSITE_TASKS = True
ROBOT_TASKS = False

# Select one or more of the Light Conditions
NORMAL_LIGHT = True
DIFFUSE_LIGHT = False




def download_files(url_base_path, url_gt_path, sequences, speed):

    for i in sequences:
        for j in speed:
            if ORIENTED_TASKS == True:
                url = url_base_path + i + j + '.zip'
                url_gt = url_gt_path + i + j + '.txt'
            else:
                url = url_base_path + i +  '.zip'
                url_gt =  url_gt_path + i + '.txt'
            

            file_name = url.split('/')[-1]
            file_name_gt = url_gt.split('/')[-1]
            

            
            u = urllib2.urlopen(url)
            u_gt = urllib2.urlopen(url_gt)
            

            f = open(file_name, 'wb')
            f_gt = open(file_name_gt, 'wb')

            

            meta = u.info()
            meta_gt = u_gt.info()
            

            file_size = int(meta.getheaders("Content-Length")[0])
            

            print "Downloading: %s Bytes: %s" % (file_name, file_size)



            file_size_dl = 0
            block_sz = 17984 
            while True:
                buffer = u.read(block_sz)
                buffer_gt = u_gt.read(block_sz)
                if not buffer:
                    break
                file_size_dl += len(buffer)
                f.write(buffer)
                f_gt.write(buffer_gt)
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                status = status + chr(8)*(len(status)+1)
                print status,

            f.close()
            f_gt.close()

if ORIENTED_TASKS == True:
    
    speed = ['1', '2', '3', '4', '5']
    if NORMAL_LIGHT == True:
        sequences = ['nl_bookI_s', 'nl_bookII_s', 'nl_bookIII_s', 'nl_mugI_s', 'nl_mugII_s', 'nl_mugIII_s', 'nl_cereal_s', 'nl_juice_s']
        url_base_path = "http://webdocs.cs.ualberta.ca/~vis/trackDB/nl_images/"
        url_gt_path = "http://webdocs.cs.ualberta.ca/~vis/trackDB/nl_tdata/"
        download_files(url_base_path, url_gt_path, sequences, speed)

    if DIFFUSE_LIGHT == True:
        sequences = ['dl_bookI_s', 'dl_bookII_s', 'dl_bookIII_s', 'dl_mugI_s', 'dl_mugII_s', 'dl_mugIII_s', 'dl_cereal_s', 'dl_juice_s']
        url_base_path = "http://webdocs.cs.ualberta.ca/~vis/trackDB/dl_images/"
        url_gt_path = "http://webdocs.cs.ualberta.ca/~vis/trackDB/dl_tdata/"
        download_files(url_base_path, url_gt_path, sequences, speed)

if COMPOSITE_TASKS == True:
    speed = ['1']
    if NORMAL_LIGHT == True:
        sequences = ['nl_bus', 'nl_newspaper', 'nl_letter', 'nl_highlighting']
        url_base_path = "http://webdocs.cs.ualberta.ca/~vis/trackDB/nl_images/"
        url_gt_path = "http://webdocs.cs.ualberta.ca/~vis/trackDB/nl_tdata/"
        download_files(url_base_path, url_gt_path, sequences, speed)
    if DIFFUSE_LIGHT == True:
        url_base_path = "http://webdocs.cs.ualberta.ca/~vis/trackDB/dl_images/"
        url_gt_path = "http://webdocs.cs.ualberta.ca/~vis/trackDB/dl_tdata/"
        download_files(url_base_path, url_gt_path, sequences, speed)

if ROBOT_TASKS == True:
    speed = ['1']
    sequences = ['robot_cereal', 'robot_juice', 'robot_mugI', 'robot_mugII', 'robot_mugIII', 'robot_bookI', 'robot_bookII', 'robot_bookIII']
    if NORMAL_LIGHT == True:
        url_base_path = "http://webdocs.cs.ualberta.ca/~vis/trackDB/nl_robotImages/"
        url_gt_path = "http://webdocs.cs.ualberta.ca/~vis/trackDB/robot_tdata/"
        download_files(url_base_path, url_gt_path, sequences, speed)
    if DIFFUSE_LIGHT == True:
        print 'No Diffuse Light Data'




