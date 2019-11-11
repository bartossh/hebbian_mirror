import requests

num_of_iter = 2

data = open('./assets/test.jpg', 'rb').read()
for i in range(0, num_of_iter):
    res = requests.get(
        url='http://0.0.0.0:8000/recognition/object/boxes_names'
    )
    print("\n RESPONSE GET boxes names for test number {}: \n {}"
        .format(i, res.__dict__))
    res = requests.post(url='http://0.0.0.0:8000/recognition/object/boxes',
                        data=data,
                        headers={'Content-Type': 'application/octet-stream'})
    print("\n RESPONSE POST to boxes, test num {} \n Sending buffer length: {},\n Received {}"
        .format(i, len(data), res.__dict__))
    res = requests.post(url='http://0.0.0.0:8000/recognition/object/image',
                        data=data,
                        headers={'Content-Type': 'application/octet-stream'})
    print("\n RESPONSE POST to image, test num {} \n Sending buffer length: {},\n Received {}"
        .format(i, len(data), res))
