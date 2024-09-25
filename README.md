# DiSulfiNet3D

a 3D Edge Conditioned Graph Neural Network trained to predict possible disulfide-bridge positions in proteins.

So far, this neural network is trained on the CATH-s40 database.

```
arguments:
  -h, --help            shows help message and exit
  -i, --input           path to the input pdb file
  -o, --ouput           output file name. default=out.csv
```