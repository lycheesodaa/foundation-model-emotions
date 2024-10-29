#Praat skeleton code for audio feature extraction

# [female]

# Modify these directories as required
directory$ = "C:/Users/caedm/Documents/Emotion-Forecasting/Processed/Female/"
outdir$ = "C:/Users/caedm/Documents/Emotion-Forecasting/Processed/Features_50_25/"

extension$ = ".wav"

 
# Create a strings list to store folder names
Create Strings as directory list: "folderList", directory$
numberOfFolders = Get number of strings

# Iterate through each folder
for i from 1 to numberOfFolders
    selectObject: "Strings folderList"
    folderName$ = Get string: i
    folderPath$ = directory$ + "/" + folderName$
    
    # Process the folder here
    appendInfoLine: "Processing folder: ", folderName$

    
    Create Strings as file list: "list", folderPath$ + "/*" + extension$

    number_files = Get number of strings
    appendInfoLine: "  Found ", number_files, " wav files"

    for a from 1 to number_files

        select Strings list

        current_file$ = Get string... 'a'
        appendInfoLine: "    Processing ", current_file$

        Read from file... 'folderPath$'/'current_file$'

        object_name$ = selected$("Sound")



        To Intensity... 100 0.025 no

        Down to Matrix

        Transpose

        Write to matrix text file... 'outdir$'/'object_name$'.intensity

        Remove

        select Matrix 'object_name$'

        Remove

        select Intensity 'object_name$'

        Remove



        select Sound 'object_name$'

        To Pitch (ac)... 0.025 100 15 no 0.03 0.45 0.01 0.35 0.14 600

        select Pitch 'object_name$'

        Smooth... 10

        Interpolate

        To Matrix

        Transpose

        Write to matrix text file... 'outdir$'/'object_name$'.pitch

        Remove

        select Matrix 'object_name$'

        Remove

        select Pitch 'object_name$'

        Remove

        select Pitch 'object_name$'

        Remove

        select Pitch 'object_name$'

        Remove



        select Sound 'object_name$'

        To Harmonicity (ac)... 0.025 100 0.1 4.5

        To Matrix

        Transpose

        Write to matrix text file... 'outdir$'/'object_name$'.hnr

        Remove

        select Matrix 'object_name$'

        Remove

        select Harmonicity 'object_name$'

        Remove



        select Sound 'object_name$'

        To MFCC... 13 0.050 0.025 100 100 0

        To Matrix

        Transpose

        Write to matrix text file... 'outdir$'/'object_name$'.mfcc

        Remove

        select Matrix 'object_name$'

        Remove

        select MFCC 'object_name$'

        Remove



        select Sound 'object_name$'

        To MelFilter... 0.050 0.025 100 100 0

        To Matrix

        Transpose

        Write to matrix text file... 'outdir$'/'object_name$'.mfb

        Remove

        select Matrix 'object_name$'

        Remove

        select MelFilter 'object_name$'

        Remove

        select Sound 'object_name$'

        Remove
    endfor
endfor
 

# [male]

# Modify these directories as required
directory$ = "C:/Users/caedm/Documents/Emotion-Forecasting/Processed/Male/"
outdir$ = "C:/Users/caedm/Documents/Emotion-Forecasting/Processed/Features_50_25/"

extension$ = ".wav"

 
# Create a strings list to store folder names
Create Strings as directory list: "folderList", directory$
numberOfFolders = Get number of strings

# Iterate through each folder
for i from 1 to numberOfFolders
    selectObject: "Strings folderList"
    folderName$ = Get string: i
    folderPath$ = directory$ + "/" + folderName$
    
    # Process the folder here
    appendInfoLine: "Processing folder: ", folderName$

    
    Create Strings as file list: "list", folderPath$ + "/*" + extension$

    number_files = Get number of strings
    appendInfoLine: "  Found ", number_files, " wav files"

    for a from 1 to number_files

        select Strings list

        current_file$ = Get string... 'a'
        appendInfoLine: "    Processing ", current_file$

        Read from file... 'folderPath$'/'current_file$'

        object_name$ = selected$("Sound")



        To Intensity... 75 0.025 no

        Down to Matrix

        Transpose

        Write to matrix text file... 'outdir$'/'object_name$'.intensity

        Remove

        select Matrix 'object_name$'

        Remove

        select Intensity 'object_name$'

        Remove



        select Sound 'object_name$'

        To Pitch (ac)... 0.025 75 15 no 0.03 0.45 0.01 0.35 0.14 600

        select Pitch 'object_name$'

        Smooth... 10

        Interpolate

        To Matrix

        Transpose

        Write to matrix text file... 'outdir$'/'object_name$'.pitch

        Remove

        select Matrix 'object_name$'

        Remove

        select Pitch 'object_name$'

        Remove

        select Pitch 'object_name$'

        Remove

        select Pitch 'object_name$'

        Remove



        select Sound 'object_name$'

        To Harmonicity (ac)... 0.025 75 0.1 4.5

        To Matrix

        Transpose

        Write to matrix text file... 'outdir$'/'object_name$'.hnr

        Remove

        select Matrix 'object_name$'

        Remove

        select Harmonicity 'object_name$'

        Remove



        select Sound 'object_name$'

        To MFCC... 13 0.050 0.025 100 100 0

        To Matrix

        Transpose

        Write to matrix text file... 'outdir$'/'object_name$'.mfcc

        Remove

        select Matrix 'object_name$'

        Remove

        select MFCC 'object_name$'

        Remove



        select Sound 'object_name$'

        To MelFilter... 0.050 0.025 100 100 0

        To Matrix

        Transpose

        Write to matrix text file... 'outdir$'/'object_name$'.mfb

        Remove

        select Matrix 'object_name$'

        Remove

        select MelFilter 'object_name$'

        Remove

        select Sound 'object_name$'

        Remove

    endfor
endfor