sudo docker-compose up --build

python3 -m http.server



sudo docker-compose down

sudo docker rm $(sudo docker ps -a -q)

sudo docker rmi $(sudo docker images -q)

sudo docker volume prune -f
