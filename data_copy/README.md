1. Copy the .RAW files from your pod (this will ask for the root password you set on your pod)
```
scp -r -P 8822 'root@<POD_IP>:/persistent/*.RAW' <FOLDER_PATH_TO_SAVE_TO_ON_YOUR_COMPUTER>
scp -r -P 8822 'root@192.168.1.50:/persistent/*.RAW' /Users/david/8sleep/raw
```

2. Install required packages
```
pip3 install cbor2 numpy pandas
```

3. Execute python script
```
python3 decode_raw_data.py <FOLDER_PATH_FROM_STEP_1> <FOLDER_PATH_TO_SAVE_TO>
python3 decode_raw_data.py /Users/david/8sleep/raw/ /Users/david/8sleep/
```


Example output
```
$ scp -r -P 8822 'root@192.168.1.50:/persistent/*.RAW' /Users/ds/main/8sleep_biometrics/data_copy/raw
root@192.168.1.50's password: 
003EC8A8.RAW                                                                                                                                                                                                                                                                                                                                    100% 7428KB  18.8MB/s   00:00    
003ED48C.RAW                                                                                                                                                                                                                                                                                                                                    100% 7424KB  22.5MB/s   00:00    
003EE06B.RAW              
...

$ python decode_raw_data.py /Users/ds/main/8sleep_biometrics/data_copy/raw /Users/ds/main/8sleep_biometrics/data_copy
Decoding 2 file(s)...
Loading cbor data from: /Users/ds/main/8sleep_biometrics/data_copy/raw/0012DBFA.RAW
Loading cbor data from: /Users/ds/main/8sleep_biometrics/data_copy/raw/0012D00F.RAW
Decoded 2 file(s)
Saving data to file...
Saved file, please share /Users/ds/main/8sleep_biometrics/data_copy/output.pkl.zip
```
