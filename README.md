
# Neural Noiseprint Transfer: A Generic Noiseprint-Based Counter Forensics Framework


## License
Copyright (c) 2022 Ahmed Elliethy.

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 

## Installation
The code requires Python 3.x and PyTorch 1.12.

To install Python 3.x for Ubuntu, you can run:

```
apt-get update
apt-get install -y python3.8 python3.8-dev python3-pip python3-venv
```

To install PyTorch, follow the link here https://pytorch.org

## Usage
To extract the noiseprint, run:

```
python3 main_nnt.py --img_forged_filename=<input forged image> --img_auth_filename=<input authentic image> --out_filename=<output file> --method_name=<method that can be 'injection' or 'optimization' (default='injection')> --visualize_results=<True if want to display results (default=true)>
```

### Demo
To execute a demo, run the following

```
python3 main_nnt.py --img_forged_filename='Demo/splicing-01.png' --img_auth_filename='Demo/normal-01.png' --out_filename='output.png'
```

### Output

![alt text](https://i.ibb.co/ZmcyGfW/Demo1.png)PSNR =  34.12433053884067, and SSIM = 0.939936182563455

```
python3 main_nnt.py --img_forged_filename='Demo/splicing-02.png' --img_auth_filename='Demo/normal-02.png' --out_filename='output.png'
```
### Output

![alt text](https://i.ibb.co/1LqvMr5/Demo2.png)PSNR =  31.57684118074469, and SSIM = 0.8961242284468497



## Reference

```
@article{Elliethy2022_NNT,
  title={# Neural Noiseprint Transfer: A Generic Noiseprint-Based Counter Forensics Framework},
  author={Ahmed Elliethy},
  journal={Under review},
} 
```

