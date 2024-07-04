In quesa cartella ho fatto dei cambiamenti.
Nella prima versione, facevo il loading del modello pre-addestrato con i pesi di IMAGENET, e lo addestravo direttamente
sul cifar2. Questo innanzitutto non è tecnicamente corretto per due motivi:

1. La VGG16 di IMAGENET è realizzata per predire 1000 classi, dunque bisogna modificare l'ultimo layer, che restituisca 2 outputs;
2. Se si fa il loading di unn modello pre-addestrato, allora si sfruttano i pesi che ha imparato.

Dunque, ho sostituito l'ultimo layer che faceva Tensor([4096,1000]) con un Tensor([4096,2]) e ho freezzato i pesi della rete
addestrata su IMAGENET fino al layer n-1, ed ho addestrato soltanto l'utlimo!
