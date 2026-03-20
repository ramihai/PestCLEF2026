# EPOP training and development documents

DOI: https://doi.org/10.57745/YKSEPY


## What is EPOP

The *Epidemiomonitoring Of Plant* (*EPOP*) corpus is a set of documents manually annotated with gold standard named entities, entity linking, binary relations and n-ary relations/events.

The [annotation guidelines are publicly available](https://hal.inrae.fr/MAIAGE/hal-04744299v2).


## Content of this archive

This archive contains the documents of the training parts of the official split.
The documents constitute the input of the PestCLEF @ LifeCLEF 2026.

- `train/*.txt` 110 files, each containing the text of a document of the training set, encoded in UTF-8.
- `dev/*.txt` 55 files, each containing the text of a document of the development set, encoded in UTF-8.
- `test/*.txt` 82 files, each containing the text of a document of the test set, encoded in UTF-8.

The text of these documents constitute the base reference for the text-bound annotations provided in the EPOP corpus. You will find the annotations here: https://doi.org/10.57745/ZDNOGF.

The `train`, `dev` and `test` directories each contain a file named `documents-metadata.csv`.
This file contains metadata associated with the documents during their collection.
It is formatted in three columns: 1/ document identifier, 2/ original language before translation, and 3/ document URL.
Note that the URL might not be available anymore or the content of the URL might have been modified by the site owner.

## Terms of use

See the file named LICENSE included in the archive where you found the documents.
