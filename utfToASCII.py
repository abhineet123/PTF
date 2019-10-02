filePath = 'The Stuff of Thought_ Language - Steven Pinker.txt'

fichier = open(filePath, "rb")
contentOfFile = fichier.read()
contentOfFile=contentOfFile.decode("utf-8")
asciidata=contentOfFile.encode("ascii","ignore")

fichier.close()

out_filePath = filePath + '.ascii'

fichierTemp = open(out_filePath, "w")
fichierTemp.write(asciidata)
fichierTemp.close()


