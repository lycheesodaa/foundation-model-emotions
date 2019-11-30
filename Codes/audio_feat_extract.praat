#Praat skeleton code for audio feature extraction

[female]

directory$ = "Processed/Female/"

outdir$ = "Processed/Features/"

extension$ = ".wav"

 

Create Strings as file list... list 'directory$'*'extension$'

number_files = Get number of strings

 

for a from 1 to number_files

                select Strings list

                current_file$ = Get string... 'a'

                Read from file... 'directory$''current_file$'

                object_name$ = selected$("Sound")

 

                To Intensity... 100 0.01 no

                Down to Matrix

                Transpose

                Write to matrix text file... 'outdir$'/'object_name$'.intensity

                Remove

                select Matrix 'object_name$'

                Remove

                select Intensity 'object_name$'

                Remove

 

                select Sound 'object_name$'

                To Pitch (ac)... 0.01 100 15 no 0.03 0.45 0.01 0.35 0.14 600

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

                To Harmonicity (ac)... 0.01 100 0.1 4.5

                To Matrix

                Transpose

                Write to matrix text file... 'outdir$'/'object_name$'.hnr

                Remove

                select Matrix 'object_name$'

                Remove

                select Harmonicity 'object_name$'

                Remove

 

                select Sound 'object_name$'

                To MFCC... 13 0.025 0.01 100 100 0

                To Matrix

                Transpose

                Write to matrix text file... 'outdir$'/'object_name$'.mfcc

                Remove

                select Matrix 'object_name$'

                Remove

                select MFCC 'object_name$'

                Remove

 

                select Sound 'object_name$'

                To MelFilter... 0.025 0.01 100 100 0

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

 

[male]

# This script measures intensity every 10ms

 

directory$ = "/Users/emilykmp/Documents/Research/IEMOCAP_Data/Male_Speakers/"

outdir$ = "/Users/emilykmp/Documents/Research/IEMOCAP_Data/Features/"

extension$ = ".wav"

 

Create Strings as file list... list 'directory$'*'extension$'

number_files = Get number of strings

 

for a from 1 to number_files

                select Strings list

                current_file$ = Get string... 'a'

                Read from file... 'directory$''current_file$'

                object_name$ = selected$("Sound")

 

                To Intensity... 75 0.01 no

                Down to Matrix

                Transpose

                Write to matrix text file... 'outdir$'/'object_name$'.intensity

                Remove

                select Matrix 'object_name$'

                Remove

                select Intensity 'object_name$'

                Remove

 

                select Sound 'object_name$'

                To Pitch (ac)... 0.01 75 15 no 0.03 0.45 0.01 0.35 0.14 600

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

                To Harmonicity (ac)... 0.01 75 0.1 4.5

                To Matrix

                Transpose

                Write to matrix text file... 'outdir$'/'object_name$'.hnr

                Remove

                select Matrix 'object_name$'

                Remove

                select Harmonicity 'object_name$'

                Remove

 

                select Sound 'object_name$'

                To MFCC... 13 0.025 0.01 100 100 0

                To Matrix

                Transpose

                Write to matrix text file... 'outdir$'/'object_name$'.mfcc

                Remove

                select Matrix 'object_name$'

                Remove

                select MFCC 'object_name$'

                Remove

 

                select Sound 'object_name$'

                To MelFilter... 0.025 0.01 100 100 0

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

Endfor