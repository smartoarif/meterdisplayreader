# meterdisplayreader
This is a test project based on fastapi, yolov7+pytorch, and intended to read numbers of a meter from its 7 segment display area. 

The fastapi endpoint /meterimage accepts a file upload (image of an electricity meter, with 'image' as the key) and return the reading on the meter screen that indicates electricity consumption units

**install**

git clone https://github.com/smartoarif/meterdisplayreader.git<br>
cd meterdisplayreader<br>
pip install -r requirements.txt

**run**

python -m uvicorn app:app --host 0.0.0.0 --port 8000

**Test**

test the URL http://localhost:8000/meterimage with postman. Set 'image' as the key & file as type with selecting the appropiate meter image, under the Body tab
![image](https://github.com/smartoarif/meterdisplayreader/assets/3448147/8ca678c1-a548-490d-a2be-9ba56583527d)

