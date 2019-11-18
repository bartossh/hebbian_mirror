END=10
STARTTIME=$(date +%s)
x=$END 
while [ $x -gt 0 ]; 
do 
  echo "----> NEW REQUESTS - COUNTING DOWN: " $x
  curl -X GET -H 'Content-Type: application/json' -i 'http://0.0.0.0:8000/recognition/object/boxes_names'
  curl -X POST -H 'Content-Type: application/octet-stream' -i 'http://0.0.0.0:8000/recognition/object/image' --data-binary  @test.jpg
  curl -X POST -H 'Content-Type: application/octet-stream' -i 'http://0.0.0.0:8000/recognition/object/boxes' --data-binary  @test.jpg
  x=$(($x-1))
  echo -e " \n"
  echo -e " \n"
  sleep .1
done
ENDTIME=$(date +%s)
echo "It takes $(($ENDTIME - $STARTTIME)) seconds"