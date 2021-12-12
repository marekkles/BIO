# Projekt BIO
## Detekcia anatomických štruktúr v sietnicových obrazoch

-----
Roman Čabala (xcabal06)

Marek Vaško  (xvasko16)

-----

Zdrojové kódy projektu sú založené na referenčných zdrojových kódov pre trénovanie konvolúčnych neurónových sietí pre sémantickú segmentáciu v projekte  [PyTorch torchvision](https://github.com/pytorch/vision/tree/main/references/segmentation). Zdrojové kódy sú distribuované pod licenciou BSD-3, ktorá je využitá rovnako pre zdrojové kódy tohto projektu.

Predtrénované modely sú prístupné na stiahnutie z [úložiska Google Drive](). Výsledky pre jednotlivé modely na testovasích dátach sú prístupné v súbore `test_results.txt`.

Pre získanie testovacích výsledkov boli využité príkazy, predpokladá sa umiestnenie datasetu v priečinku `../BIO_data/RetinaDataset/`

`python3 train.py --model deeplabv3_resnet101 --resume Deeplabv3_Resnet10-30epochs/checkpoint.pth --test-only`

`python3 train.py --model fcn_resnet101 --resume ResNet101-30epochs/checkpoint.pth --test-only`

`python3 train.py --model lraspp_mobilenet_v3_large --resume MobileNetV3-50epoch/checkpoint.pth --test-only`

Pre trénovanie iných modelov odporúčame odkaz na torchvision a príkaz `python3 train.py --help` 

Pre ukážku inferencie je možné využiť Python Notebook `inference.ipynb`.

Dataset je prístupný na odkaze [na úložisku Google Drive]()



