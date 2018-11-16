__author__ = 'Tommy'
playlist_name = 'C:\Videos\#songs_new.pls'
list_filename = 'list.txt'

new_files = open(list_filename).readlines()
playlist = open(playlist_name, 'r')
header0 = playlist.readline()
header1 = playlist.readline().split('=')
rem_lines = playlist.readlines()
playlist.close()

print 'header1=', header1

no_of_entries = int(header1[1])
print 'no_of_entries=', no_of_entries

playlist = open(playlist_name, 'w')
playlist.write(header0)
playlist.write(header1[0] + '=' + str(no_of_entries + len(new_files))+'\n')
for line in rem_lines:
    playlist.write(line)

file_id = no_of_entries + 1
for new_file in new_files:
    playlist.write('File' + str(file_id) + '=\\' + new_file)
    file_id += 1

playlist.close()
print 'no_of_entries now=', file_id - 1
raw_input("Press Enter to continue...")