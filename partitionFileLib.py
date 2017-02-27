'''Partitioning a file into chunkFiles, chunkSize words long'''

from nltk.tokenize import word_tokenize
import os

def chunkFile(filePath, filename, chunkPath, chunkSize):

    with open(filePath + filename,'r') as file :
        #i = 0
        content = file.read()
        tokenized = word_tokenize(content)
		#tokenized = word_tokenize(content.decode('latin-1'))
		
        if not len(tokenized) < chunkSize :
            content = content.splitlines()
            chunk = ''
            chunkTokenized = []
            chunkIndex = 0
            i = 0
			
            while i < len(content):
                line = content[i]
                i += 1
                chunk += line + '\n'
                chunkTokenized += word_tokenize(line)
				#chunkTokenized += word_tokenize(line.decode('latin-1'))
				
                if len(chunkTokenized) >= chunkSize or i == len(content): #change 250 in chunkSize
                    chunkFilename = '.'.join(filename.split('.')[0:len(filename.split('.'))-1]) + '[' + str(chunkIndex) + '].txt'
                    chunkIndex += 1
                    with open(chunkPath+chunkFilename, 'w') as chunkFile:
                        chunkFile.write(chunk)
                    chunk = ''
                    chunkTokenized = []
				
					
#chunkFile("C:/Users/Spéro/Documents/Research Project Data Science/Unmasking Project/Comparison Set-selected/", 'test.txt', './WithmanChunks/', 250)

# Creating the questioned chunk directory
# if not 'ChunkQuestioned BDT 1857' in os.listdir('./'):
#     os.mkdir('ChunkQuestioned BDT 1857')
# Split the Withman known text into pieces
# chunkFile("./", 'BDE 46.48.txt', './WithmanChunks/', 250)
# # Now split the anonymous files
# months = os.listdir('Questioned BDT 1857')
# i = 0
# for month in months:
#     if not month == '.DS_Store':
#         if not month in os.listdir('ChunkQuestioned BDT 1857'):
#             os.mkdir('ChunkQuestioned BDT 1857/'+month)
#         for filename in os.listdir('Questioned BDT 1857/'+month):
#             if not filename == '.DS_Store' and not filename == 'chunks':
#                 i += 1
#                 chunkFile('Questioned BDT 1857/'+month+'/', filename, 'ChunkQuestioned BDT 1857/'+month+'/', 250)