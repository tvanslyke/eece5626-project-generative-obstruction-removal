import socket
import sys


def cmdline(lookup, addr='timspc.local', port=12000):
	""" Simple socket server to inspect stuff remotely. """
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	with server_socket:
		server_socket.bind(('timspc.local', 12000))
		while True:
			message, address = server_socket.recvfrom(1024)
			for line in message.decode().split('\n'):
				value = lookup.get(line.strip()) or "?"
				server_socket.sendto("{}\n".format(value).encode(), address)
		
		



