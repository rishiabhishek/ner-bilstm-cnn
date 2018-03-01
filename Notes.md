

# Notes


## Features:


 1. **Character Features Using a CNN** : Words are padded with a number of special PADDING characters on both sides depending on the window size of the CNN.
 2. **Word Embeddings**<br />
	 2.1. https://ronan.collobert.com/senna/ <br />
	 2.2. http://nlp.stanford.edu/projects/glove/ <br />
	 2.3. https://code.google.com/p/word2vec/ <br />
	
 3. **Character Embeddings** : The character set includes all unique characters in the CoNLL-2003 dataset plus the special tokens PADDING and UNKNOWN. The PADDING token is used for the CNN, and the UNKNOWN token is used for all other characters.
 4. **Additional Word-level Features**<br />
	 4.1. **Capitalization Feature** : allCaps, upperInitial, lowercase, mixedCaps, noinfo<br />
	 4.2. **Lexicons** *(Not Used)*<br />
 5. **Character-level Features** : A lookup table was used to output a 4-dimensional vector representing the type of the character (upper case, lower case, punctuation, other).

    


  
