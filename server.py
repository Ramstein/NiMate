# first of all import the socket library
import socket

# create an INET, STREAMing socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Socket successfully created")
port = 33433

# Next bind to the port
# we have not typed any ip in the ip field
# instead we have inputted an empty string
# this makes the server listen to requests
# coming from other computers on the network
s.bind(('', port))
print("socket binded to %s" %(port) )

# put the socket into listening mode
s.listen(5)
print("socket is listening")

# a forever loop until we interrupt it or
# an error occurs
while True:

    # Establish connection with client.
    conn, addr = s.accept()
    print('Got connection from', addr)
    # send a thank you message to the client.
    for i in range(10):
        conn.send(bytes('Thank you for connecting.'+str(i), encoding='utf-8'))
    # Close the connection with the client
    conn.close()
